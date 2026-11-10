from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .metrics import EnrichedRow
from .models import Check, MarketStats, ScenarioResult


def _check(name: str, condition: bool | None, actual: str, rule: str) -> Check:
    return Check(name=name, passed=condition, actual=actual, rule=rule)


def _between(value: float | None, low: float, high: float) -> bool | None:
    return None if value is None else low <= value <= high


def _ge(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value >= threshold


def _le(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value <= threshold


def _gt(value: float | None, threshold: float = 0.0) -> bool | None:
    return None if value is None else value > threshold


def _lt(value: float | None, threshold: float = 0.0) -> bool | None:
    return None if value is None else value < threshold


def _no_flags(flags: list[str]) -> bool:
    return not flags


def _symbol_base(symbol: str) -> str:
    value = str(symbol or "").upper().strip().split(".", 1)[0]
    for suffix in ("_PERPETUAL", "-PERPETUAL", "_PERP", "-PERP", "PERPETUAL", "PERP", "_SWAP", "-SWAP"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "BNB", "EUR"):
        if value.endswith(quote) and len(value) > len(quote):
            value = value[:-len(quote)]
            break
    return value


def _oi_growth_min(symbol: str, thresholds: dict[str, Any]) -> float:
    # P1's lower major-asset threshold applies to BTC and ETH only. Prefix
    # matching incorrectly classified ETHFI/ETHW/BTCDOM as majors.
    base_asset = _symbol_base(symbol)
    is_major = base_asset in {"BTC", "ETH"}
    key = "oi_4h_major_pct" if is_major else "oi_4h_alt_pct"
    default = 0.3 if is_major else 1.0
    return float(thresholds.get(key, default))


def _result(row: EnrichedRow, scenario: str, direction: str, checks: list[Check], status: str = "NOT_MET", reason: str = "", technical: dict[str, Any] | None = None) -> ScenarioResult:
    # A PFR OI_W sign reversal is qualitative in P1: it does not cancel an
    # otherwise confirmed setup, but it must remain visible as a warning.
    market = row.market
    funding_sign_reversal = (
        market.fr_oiw_pct is not None
        and market.pfr_oiw_pct is not None
        and market.fr_oiw_pct * market.pfr_oiw_pct < 0
    )
    if status == "CONFIRMED_SIGNAL" and funding_sign_reversal:
        status = "CONFIRMED_SIGNAL_WITH_WARNING"
        warning = "PFR OI_W changes sign versus current FR OI_W; qualitative warning only (P1 has no numeric cutoff)."
        reason = f"{reason}; {warning}" if reason else warning

    return ScenarioResult(
        symbol=row.market.symbol,
        name=row.market.name,
        scenario=scenario,
        direction=direction,
        checks=checks,
        quality_flags=list(row.quality_flags),
        status=status,
        reason=reason,
        technical=technical or {},
    )


def _is_cascade(row: EnrichedRow) -> bool | None:
    """Conservative SHORT prohibition for a possible long-liquidation flush.

    The global P1 prohibition is broader than the explicit SHORT D signature:
    a high long-liquidation share plus falling OI on both reported horizons is
    enough to block fresh SHORT candidates. F/S negativity remains a separate
    required condition for the SHORT D scenario itself. The prompt does not
    quantify "near the 24H low", so this guard intentionally errs on the side
    of blocking shorts until a 1H bounce/rejection is established.
    """
    m = row.market
    required = [row.long_liq_share, m.oi_chg24_pct, m.oi_chg4h_pct]
    if any(v is None for v in required):
        return None
    return bool(row.long_liq_share >= 0.80 and m.oi_chg24_pct < 0 and m.oi_chg4h_pct < 0)


def evaluate_scenarios(
    row: EnrichedRow,
    stats: MarketStats,
    p1: dict[str, Any],
    *,
    technical: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    current_snapshot_available: bool = True,
) -> list[ScenarioResult]:
    m = row.market
    t = p1.get("thresholds", {})
    s = p1.get("scenario", {})
    technical = technical or {}
    history = history or []
    no_flags = _no_flags(row.quality_flags)
    oi_min = _oi_growth_min(m.symbol, t)
    out: list[ScenarioResult] = []

    # LONG A — Trend Continuation
    flow_ok = None
    if m.futures_imbalance_4h is not None and m.spot_imbalance_4h is not None:
        flow_ok = bool(m.futures_imbalance_4h > 0 and m.spot_imbalance_4h > 0 and (row.f_strong_buy or row.s_strong_buy))
    checks = [
        _check("F/S positive; at least one strong", flow_ok, f"F={m.futures_imbalance_4h}; S={m.spot_imbalance_4h}; strong_F={row.f_strong_buy}; strong_S={row.s_strong_buy}", "F>0 and S>0 and (F or S in top 15%)"),
        _check("OI 4H growth", _ge(m.oi_chg4h_pct, oi_min), f"{m.oi_chg4h_pct}% (threshold +{oi_min}%)", f">={oi_min}%"),
        _check("Price change 24H", _between(m.chg24_pct, 1.0, 12.0), f"{m.chg24_pct}%", "+1%..+12%"),
        _check("Funding OI_W", _lt(m.fr_oiw_pct, float(t.get("funding_hot_pct", 0.03))), f"{m.fr_oiw_pct}%", "< +0.03%"),
        _check("L/S 1D", _lt(m.ls_ratio_1d, float(t.get("ls_long_extreme", 2.5))), str(m.ls_ratio_1d), "< 2.5"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    status, reason = "NOT_MET", ""
    if all(c.passed is True for c in checks):
        retest = technical.get("trend_retest_long", {})
        if retest.get("confirmed") is True:
            status = "CONFIRMED_SIGNAL"
            reason = "All P1 table conditions and the closed 1H breakout/retest are confirmed; verify execution risk before entry"
        else:
            status = "WAITING_FOR_1H_BREAKOUT_RETEST"
            reason = "P1 table conditions are met, but the closed 1H breakout and retest are not confirmed"
    out.append(_result(row, "LONG A — Trend Continuation", "LONG", checks, status, reason, {"trend_retest_long": technical.get("trend_retest_long")}))

    # LONG B — Spot Accumulation
    spot_seq = technical.get("spot_sequence_ok")
    higher_low = technical.get("higher_low", {})
    checks = [
        _check("Strong positive Spot", bool(row.s_strong_buy) if m.spot_imbalance_4h is not None else None, f"S={m.spot_imbalance_4h}; rank={row.s_rank_desc}", "S>0 and top 15%"),
        _check("F range", _between(m.futures_imbalance_4h, -10.0, 2.0), f"{m.futures_imbalance_4h}", "-10..+2"),
        _check("OI 4H range", _between(m.oi_chg4h_pct, -2.0, 1.0), f"{m.oi_chg4h_pct}%", "-2%..+1%"),
        _check("Price change 24H", _between(m.chg24_pct, -4.0, 6.0), f"{m.chg24_pct}%", "-4%..+6%"),
        _check("Funding OI_W", _le(m.fr_oiw_pct, 0.01), f"{m.fr_oiw_pct}%", "<= +0.01%"),
        _check("L/S 1D", _lt(m.ls_ratio_1d, 1.8), str(m.ls_ratio_1d), "< 1.8"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    if all(c.passed is True for c in checks):
        if spot_seq is True and higher_low.get("confirmed") is True and no_flags:
            status, reason = "CONFIRMED_SIGNAL", "Three separate positive Spot snapshots and the 1H higher low are confirmed; verify execution risk before entry"
        else:
            status, reason = "WAITING_FOR_HISTORY_OR_HIGHER_LOW", "Table conditions met; still need 3 positive Spot snapshots spaced about 4h apart and a 1H higher low"
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "LONG B — Spot Accumulation", "LONG", checks, status, reason, {"spot_sequence_ok": spot_seq, "higher_low": higher_low}))

    # LONG C — Short Squeeze, two-stage rule. The phrase 'significant short liquidations' is not quantified in P1.
    short_liq_min = float(s.get("short_squeeze_min_short_liqs_usd", 0.0))
    prep_checks = [
        _check("Funding preparation", _le(m.fr_oiw_pct, float(t.get("funding_negative_pct", -0.005))), f"{m.fr_oiw_pct}%", "<= -0.005%"),
        _check("L/S preparation", _le(m.ls_ratio_1d, float(t.get("ls_short_extreme", 0.9))), str(m.ls_ratio_1d), "<= 0.9"),
    ]
    trigger_checks = prep_checks + [
        _check("F positive", _gt(m.futures_imbalance_4h), str(m.futures_imbalance_4h), "> 0"),
        _check("1H price rising", technical.get("rising_1h"), str(technical.get("rising_1h")), "Latest closed 1H close > previous closed 1H close"),
        _check("OI 4H cap", _le(m.oi_chg4h_pct, 0.5), f"{m.oi_chg4h_pct}%", "<= +0.5%"),
        _check("Short liquidations significant", _ge(m.liq_short24_usd, short_liq_min) if short_liq_min > 0 else None, f"${m.liq_short24_usd:,.0f}" if m.liq_short24_usd is not None else "n/a", f">= ${short_liq_min:,.0f}" if short_liq_min > 0 else "UNDEFINED: configure a USD threshold"),
        _check("Spot not below -3", _ge(m.spot_imbalance_4h, -3.0), str(m.spot_imbalance_4h), ">= -3"),
        _check("No vertical pump", None if m.chg24_pct is None or m.vol_mktcap_ratio is None else not (m.chg24_pct > 12.0 or m.vol_mktcap_ratio > 2.0), f"price24={m.chg24_pct}%; VOL24H/MktCap={m.vol_mktcap_ratio}", "price24 <= +12% and VOL24H/MktCap <= 2"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    if all(c.passed is True for c in trigger_checks):
        status, reason = "CONFIRMED_SIGNAL", "Both LONG C stages and the rising 1H trigger are confirmed; verify execution risk before entry"
        visible_checks = trigger_checks
    elif all(c.passed is True for c in prep_checks):
        status, reason = "PREPARATION", "Preparation stage passed (2/2); squeeze trigger is not confirmed"
        visible_checks = prep_checks
    else:
        status, reason = "NOT_MET", ""
        visible_checks = trigger_checks
    trigger_details = [
        {"name": check.name, "passed": check.passed, "actual": check.actual, "rule": check.rule}
        for check in trigger_checks[len(prep_checks):]
    ]
    out.append(_result(row, "LONG C — Short Squeeze", "LONG", visible_checks, status, reason, {
        "preparation_checks": [c.passed for c in prep_checks],
        "trigger_checks": trigger_details,
    }))

    # LONG D — Post-Long-Flush Reversal
    checks = [
        _check("Long liquidation share", _ge(row.long_liq_share, 0.80), f"{row.long_liq_share:.2%}" if row.long_liq_share is not None else "n/a", ">= 80%"),
        _check("OI 24H", _le(m.oi_chg24_pct, -5.0), f"{m.oi_chg24_pct}%", "<= -5%"),
        _check("Price 24H", _le(m.chg24_pct, -4.0), f"{m.chg24_pct}%", "<= -4%"),
        _check("Spot positive", _gt(m.spot_imbalance_4h), str(m.spot_imbalance_4h), "> 0"),
        _check("F not below universe median", None if m.futures_imbalance_4h is None or stats.median_f is None else m.futures_imbalance_4h >= stats.median_f, f"{m.futures_imbalance_4h} vs median {stats.median_f}", "F >= median F"),
    ]
    if all(c.passed is True for c in checks):
        if not no_flags:
            status, reason = "BLOCKED_QUALITY", "Data pattern matches but market-quality flags exclude this market"
        elif higher_low.get("confirmed") is True:
            status, reason = "CONFIRMED_SIGNAL", "All P1 table conditions and the 1H higher low are confirmed; verify execution risk before entry"
        else:
            status, reason = "WAITING_FOR_HIGHER_LOW", "P1 table conditions are met, but LONG D is not confirmed until a 1H higher low forms"
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "LONG D — Post-Long-Flush Reversal", "LONG", checks, status, reason, {"higher_low": higher_low, "liq_coverage_of_oi_drop_pct": row.liquidation_coverage_of_oi_drop_pct}))

    # SHORT A — Trend Continuation
    flow_ok = None
    if m.futures_imbalance_4h is not None and m.spot_imbalance_4h is not None:
        flow_ok = bool(m.futures_imbalance_4h < 0 and m.spot_imbalance_4h < 0 and (row.f_strong_sell or row.s_strong_sell))
    cascade = _is_cascade(row)
    checks = [
        _check("F/S negative; at least one strong", flow_ok, f"F={m.futures_imbalance_4h}; S={m.spot_imbalance_4h}; strong_F={row.f_strong_sell}; strong_S={row.s_strong_sell}", "F<0 and S<0 and (F or S in bottom 15%)"),
        _check("OI 4H growth", _ge(m.oi_chg4h_pct, oi_min), f"{m.oi_chg4h_pct}% (threshold +{oi_min}%)", f">={oi_min}%"),
        _check("Price change 24H", _between(m.chg24_pct, -12.0, -1.0), f"{m.chg24_pct}%", "-12%..-1%"),
        _check("Funding OI_W", _ge(m.fr_oiw_pct, float(t.get("funding_negative_pct", -0.005))), f"{m.fr_oiw_pct}%", ">= -0.005%"),
        _check("L/S 1D", _ge(m.ls_ratio_1d, float(t.get("ls_short_extreme", 0.9))), str(m.ls_ratio_1d), ">= 0.9"),
        _check("No cascade", None if cascade is None else not cascade, str(cascade), "SHORT D cascade signs must not all be present"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    if all(c.passed is True for c in checks):
        retest = technical.get("trend_retest_short", {})
        if retest.get("confirmed") is True:
            status, reason = "CONFIRMED_SIGNAL", "All P1 table conditions and the closed 1H breakdown/retest are confirmed; verify execution risk before entry"
        else:
            status, reason = "WAITING_FOR_1H_BREAKDOWN_RETEST", "P1 table conditions are met, but the closed 1H breakdown and retest are not confirmed"
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "SHORT A — Trend Continuation", "SHORT", checks, status, reason, {"trend_retest_short": technical.get("trend_retest_short"), "cascade": cascade}))

    # SHORT B — Spot Distribution
    checks = [
        _check("Strong negative Spot", bool(row.s_strong_sell) if m.spot_imbalance_4h is not None else None, f"S={m.spot_imbalance_4h}; rank={row.s_rank_desc}", "S<0 and bottom 15%"),
        _check("F range", _between(m.futures_imbalance_4h, -2.0, 10.0), str(m.futures_imbalance_4h), "-2..+10"),
        _check("OI 4H range", _between(m.oi_chg4h_pct, -1.0, 1.5), f"{m.oi_chg4h_pct}%", "-1%..+1.5%"),
        _check("Price change 24H", _ge(m.chg24_pct, 2.0), f"{m.chg24_pct}%", ">= +2%"),
        _check("Funding OI_W", _ge(m.fr_oiw_pct, 0.01), f"{m.fr_oiw_pct}%", ">= +0.01%"),
        _check("L/S 1D", _ge(m.ls_ratio_1d, 1.3), str(m.ls_ratio_1d), ">= 1.3"),
        _check("No liquidation-cascade prohibition", None if cascade is None else not cascade, str(cascade), "SHORT forbidden during active long-liquidation cascade"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    if all(c.passed is True for c in checks):
        retest = technical.get("trend_retest_short", {})
        if retest.get("confirmed") is True:
            status = "CONFIRMED_SIGNAL"
            reason = "SHORT B Spot distribution conditions and the closed 1H breakdown/retest are confirmed; verify execution risk before entry"
        else:
            status = "WAITING_FOR_1H_BREAKDOWN_RETEST"
            reason = "SHORT B table conditions are met, but the closed 1H breakdown and retest are not confirmed"
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "SHORT B — Spot Distribution", "SHORT", checks, status, reason, {"trend_retest_short": technical.get("trend_retest_short")}))

    # SHORT C — Long Squeeze / Reversal
    checks = [
        _check("Funding hot", _ge(m.fr_oiw_pct, float(t.get("funding_hot_pct", 0.03))), f"{m.fr_oiw_pct}%", ">= +0.03%"),
        _check("L/S 1D crowded long", _ge(m.ls_ratio_1d, float(t.get("ls_long_extreme", 2.5))), str(m.ls_ratio_1d), ">= 2.5"),
        _check("Price change 24H", _ge(m.chg24_pct, 8.0), f"{m.chg24_pct}%", ">= +8%"),
        _check("OI change 24H", _ge(m.oi_chg24_pct, 10.0), f"{m.oi_chg24_pct}%", ">= +10%"),
        _check("F reversed down", _le(m.futures_imbalance_4h, -2.0), str(m.futures_imbalance_4h), "<= -2"),
        _check("S not positive excess", _le(m.spot_imbalance_4h, 2.0), str(m.spot_imbalance_4h), "<= +2"),
        _check("OI 4H cap", _le(m.oi_chg4h_pct, 0.5), f"{m.oi_chg4h_pct}%", "<= +0.5%"),
        _check("No liquidation-cascade prohibition", None if cascade is None else not cascade, str(cascade), "SHORT forbidden during active long-liquidation cascade"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    if all(c.passed is True for c in checks):
        structure = technical.get("bearish_structure_break", {})
        retest = technical.get("trend_retest_short", {})
        if structure.get("confirmed") is True and retest.get("confirmed") is True and no_flags:
            status = "CONFIRMED_SIGNAL"
            reason = "SHORT C table conditions, the closed 1H bearish structure break and successful breakdown retest are confirmed"
        else:
            status = "WAITING_FOR_1H_STRUCTURE_BREAK"
            reason = "SHORT C table conditions are met, but the 1H breakdown and successful retest are not both confirmed"
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "SHORT C — Long Squeeze / Reversal", "SHORT", checks, status, reason, {
        "bearish_structure_break": technical.get("bearish_structure_break"),
        "trend_retest_short": technical.get("trend_retest_short"),
    }))

    # SHORT D — Long Liquidation Cascade. Wait status only; never alert as an entry.
    checks = [
        _check("Long liquidation share", _ge(row.long_liq_share, 0.80), f"{row.long_liq_share:.2%}" if row.long_liq_share is not None else "n/a", ">= 80%"),
        _check("OI 24H negative", _lt(m.oi_chg24_pct), f"{m.oi_chg24_pct}%", "< 0%"),
        _check("OI 4H negative", _lt(m.oi_chg4h_pct), f"{m.oi_chg4h_pct}%", "< 0%"),
        _check("F negative", _lt(m.futures_imbalance_4h), str(m.futures_imbalance_4h), "< 0"),
        _check("S negative", _lt(m.spot_imbalance_4h), str(m.spot_imbalance_4h), "< 0"),
        _check(
            "Liquidations cover at least 10% of OI decline",
            _ge(row.liquidation_coverage_of_oi_drop_pct, 10.0),
            f"{row.liquidation_coverage_of_oi_drop_pct:.2f}%" if row.liquidation_coverage_of_oi_drop_pct is not None else "n/a",
            ">= 10%; below 10% P1 attributes most OI decline to voluntary closing, not a liquidation cascade",
        ),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    cascade_inputs_pass = all(c.passed is True for c in checks[:5])
    if cascade_inputs_pass and not no_flags:
        status, reason = "BLOCKED_QUALITY", "Cascade inputs are present but quality flags exclude this market"
    elif cascade_inputs_pass and checks[5].passed is not True:
        status, reason = "BLOCKED_CAUSALITY", "Liquidations explain less than 10% of OI decline or coverage data is missing; P1 does not support calling this a liquidation cascade"
    elif all(c.passed is True for c in checks):
        rejection = technical.get("weak_bounce_rejection", {})
        if rejection.get("confirmed"):
            status, reason = "CONFIRMED_SIGNAL", "Liquidation-cascade filters, weak bounce and rejection at resistance are confirmed; verify execution risk before entry and never short the low"
        else:
            status, reason = "WAIT_BOUNCE", "Cascade conditions present; SHORT is prohibited at the low. Wait for weak bounce and rejection at resistance."
    else:
        status, reason = "NOT_MET", ""
    out.append(_result(row, "SHORT D — Long Liquidation Cascade", "SHORT", checks, status, reason, {"weak_bounce_rejection": technical.get("weak_bounce_rejection"), "liq_coverage_of_oi_drop_pct": row.liquidation_coverage_of_oi_drop_pct}))

    # SHORT E — Failed Short Squeeze. Use the latest three consecutive P1
    # observations only; a changed but unsaved poll is not a new snapshot.
    sequence_cfg = s
    interval_seconds = max(0.0, float(sequence_cfg.get("snapshot_interval_hours", 4.0))) * 3600.0
    tolerance_seconds = max(0, int(sequence_cfg.get("snapshot_interval_tolerance_minutes", 10))) * 60.0
    latest_three = history[-3:] if len(history) >= 3 else []
    sequence_times: list[datetime] = []
    if current_snapshot_available and len(latest_three) == 3:
        for item in latest_three:
            try:
                stamp = datetime.fromisoformat(str(item["snapshot_at"]))
            except (KeyError, TypeError, ValueError):
                sequence_times = []
                break
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            sequence_times.append(stamp.astimezone(timezone.utc))
    spacing_ok = len(sequence_times) == 3 and all(
        interval_seconds <= (later - earlier).total_seconds() <= interval_seconds + tolerance_seconds
        for earlier, later in zip(sequence_times, sequence_times[1:])
    )
    initial = latest_three[0] if spacing_ok else None
    previous = latest_three[1] if spacing_ok else None
    history_ready = bool(spacing_ok)
    initial_short_dominant = None
    initial_oi_falling = None
    prior_flow_not_down = None
    if initial:
        long_liqs = initial.get("liq_long24_usd")
        short_liqs = initial.get("liq_short24_usd")
        initial_short_dominant = short_liqs > long_liqs if short_liqs is not None and long_liqs is not None else None
        oi24 = initial.get("oi_chg24_pct")
        initial_oi_falling = oi24 < 0 if oi24 is not None else None
    if previous:
        prev_f = previous.get("futures_imbalance_4h")
        prev_s = previous.get("spot_imbalance_4h")
        prior_flow_not_down = (prev_f >= 0 and prev_s >= 0) if prev_f is not None and prev_s is not None else None
    failed_breakout = technical.get("failed_breakout_rejection", {}).get("confirmed")
    current_flow_down = None if m.futures_imbalance_4h is None or m.spot_imbalance_4h is None else (m.futures_imbalance_4h < 0 and m.spot_imbalance_4h < 0)
    flow_turned_down = None if prior_flow_not_down is None or current_flow_down is None else (prior_flow_not_down and current_flow_down)
    checks = [
        _check("Three consecutive 4H snapshots available", history_ready, f"latest committed snapshots={len(latest_three)}; spacing valid={spacing_ok}", f"3 consecutive snapshots spaced {interval_seconds / 3600:g}h ± {tolerance_seconds / 60:g}m; current data must be saved/unchanged"),
        _check("Earlier short liquidations dominated", initial_short_dominant, str(initial_short_dominant), "Snapshot t−2: Short Liqs > Long Liqs"),
        _check("Earlier OI was falling", initial_oi_falling, str(initial_oi_falling), "Snapshot t−2: OI Chg 24H < 0"),
        _check("F/S turned down", flow_turned_down, f"previous F/S not both negative={prior_flow_not_down}; current F={m.futures_imbalance_4h}; S={m.spot_imbalance_4h}", "Snapshot t−1: F/S >= 0; current snapshot: F<0 and S<0"),
        _check("Breakout failed on 1H", failed_breakout, str(failed_breakout), "Recent high above swing high; latest 1H close back below"),
        _check("No liquidation-cascade prohibition", None if cascade is None else not cascade, str(cascade), "SHORT forbidden during active long-liquidation cascade"),
        _check("Quality flags", no_flags, ", ".join(row.quality_flags) or "none", "none"),
    ]
    status = "CONFIRMED_SIGNAL" if all(c.passed is True for c in checks) and no_flags else "NOT_MET"
    reason = "Three consecutive 4H observations, the F/S turn-down and the 1H failed-breakout are confirmed; verify execution risk before entry" if status == "CONFIRMED_SIGNAL" else "Requires three consecutive 4H snapshots with valid spacing, earlier short liquidation/OI decline, F/S turn-down and 1H failed-breakout structure"

    out.append(_result(row, "SHORT E — Failed Short Squeeze", "SHORT", checks, status, reason, {"failed_breakout_rejection": technical.get("failed_breakout_rejection")}))

    return out
