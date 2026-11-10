from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .models import MarketRow, MarketStats


@dataclass(slots=True)
class EnrichedRow:
    market: MarketRow
    futures_buy_share_pct: float | None
    spot_buy_share_pct: float | None
    spread: float | None
    delta_funding_oiw_pp: float | None
    fr_venue_dispersion_pp: float | None
    long_liq_share: float | None
    liq_imbalance: float | None
    rel_strength_vs_btc_pp: float | None
    liquidation_coverage_of_oi_drop_pct: float | None
    oi_drop_mostly_voluntary: bool | None
    f_adj: float | None
    s_adj: float | None
    f_rank_desc: float | None
    s_rank_desc: float | None
    f_strong_buy: bool
    f_strong_sell: bool
    s_strong_buy: bool
    s_strong_sell: bool
    quality_flags: list[str]
    diagnostic_notes: list[str]

    @property
    def symbol(self) -> str:
        return self.market.symbol


def _median(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(clean) if clean else None


def _rank_map(rows: list[MarketRow], attr: str) -> tuple[dict[str, float], int]:
    """Return descending average ranks so tied flows are treated identically."""
    values = [(r.symbol, float(getattr(r, attr))) for r in rows if getattr(r, attr) is not None]
    values.sort(key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][1] == values[start][1]:
            end += 1
        # Positions are 1-based; the mean position is the shared rank for a tie group.
        average_rank = ((start + 1) + end) / 2.0
        for symbol, _ in values[start:end]:
            ranks[symbol] = average_rank
        start = end
    return ranks, len(values)


def _rank_tail_count(valid_count: int, quantile: float) -> int:
    if valid_count <= 0:
        return 0
    return max(1, int(math.floor(valid_count * quantile + 0.5)))


def analyze_universe(rows: list[MarketRow], p1: dict[str, Any]) -> tuple[list[EnrichedRow], MarketStats]:
    if not rows:
        raise ValueError("Cannot analyze an empty Coinalyze universe")

    thresholds = p1.get("thresholds", {})
    quantile = float(thresholds.get("strong_flow_quantile", 0.15))
    quantile = min(max(quantile, 0.01), 0.49)
    median_f = _median([r.futures_imbalance_4h for r in rows])
    median_s = _median([r.spot_imbalance_4h for r in rows])
    median_oi24 = _median([r.oi_chg24_pct for r in rows])
    median_oi4h = _median([r.oi_chg4h_pct for r in rows])
    median_oi24_usd = _median([r.oi_chg24_usd for r in rows])
    median_oi4h_usd = _median([r.oi_chg4h_usd for r in rows])
    f_rank, f_n = _rank_map(rows, "futures_imbalance_4h")
    s_rank, s_n = _rank_map(rows, "spot_imbalance_4h")
    # P1's "top/bottom 15%" is rounded to the nearest whole row. For the
    # reference universe this yields 11 of 75 Futures rows and 11 of 72
    # non-null Spot rows; ceil() incorrectly admitted a 12th Futures row.
    f_tail = _rank_tail_count(f_n, quantile)
    s_tail = _rank_tail_count(s_n, quantile)

    btc = next((r for r in rows if r.symbol in {"BTC", "BTCUSDT", "BTCUSDT_PERP", "BTCUSDT_PERP.A"}), None)
    btc_change = btc.chg24_pct if btc else None

    enriched: list[EnrichedRow] = []
    for row in rows:
        f = row.futures_imbalance_4h
        s = row.spot_imbalance_4h
        spread = (s - f) if s is not None and f is not None else None
        fut_buy = (100.0 + f) / 2.0 if f is not None else None
        spot_buy = (100.0 + s) / 2.0 if s is not None else None
        delta_funding = (row.pfr_oiw_pct - row.fr_oiw_pct) if row.pfr_oiw_pct is not None and row.fr_oiw_pct is not None else None
        fr_disp = (row.fr_avg_pct - row.fr_oiw_pct) if row.fr_avg_pct is not None and row.fr_oiw_pct is not None else None
        liq_sum = None
        if row.liq_long24_usd is not None and row.liq_short24_usd is not None:
            liq_sum = row.liq_long24_usd + row.liq_short24_usd
        long_share = (row.liq_long24_usd / liq_sum) if liq_sum and row.liq_long24_usd is not None else None
        liq_imb = ((row.liq_short24_usd - row.liq_long24_usd) / liq_sum) if liq_sum and row.liq_short24_usd is not None and row.liq_long24_usd is not None else None
        rel_strength = (row.chg24_pct - btc_change) if row.chg24_pct is not None and btc_change is not None else None

        liq_total = row.liq_total24_usd if row.liq_total24_usd is not None else liq_sum
        liq_coverage = None
        voluntary = None
        if row.oi_chg24_usd is not None and row.oi_chg24_usd < 0 and liq_total is not None:
            liq_coverage = (liq_total / abs(row.oi_chg24_usd)) if row.oi_chg24_usd else None
            voluntary = liq_coverage is not None and liq_coverage < 0.10

        f_r = f_rank.get(row.symbol)
        s_r = s_rank.get(row.symbol)
        f_strong_buy = bool(f is not None and f > 0 and f_r is not None and f_r <= f_tail)
        f_strong_sell = bool(f is not None and f < 0 and f_r is not None and f_r >= f_n - f_tail + 1)
        s_strong_buy = bool(s is not None and s > 0 and s_r is not None and s_r <= s_tail)
        s_strong_sell = bool(s is not None and s < 0 and s_r is not None and s_r >= s_n - s_tail + 1)

        oi_vol = row.oi_vol_ratio
        if oi_vol is None and row.oi_usd is not None and row.volume24_usd and row.volume24_usd > 0:
            oi_vol = row.oi_usd / row.volume24_usd
        vol_mc = row.vol_mktcap_ratio
        if vol_mc is None and row.volume24_usd is not None and row.market_cap_usd and row.market_cap_usd > 0:
            vol_mc = row.volume24_usd / row.market_cap_usd
        # Persist derived ratios on the row so downstream analysis uses one source of truth.
        if row.oi_vol_ratio is None:
            row.oi_vol_ratio = oi_vol
        if row.vol_mktcap_ratio is None:
            row.vol_mktcap_ratio = vol_mc
        if row.oi_mktcap_ratio is None and row.oi_usd is not None and row.market_cap_usd and row.market_cap_usd > 0:
            row.oi_mktcap_ratio = row.oi_usd / row.market_cap_usd

        flags: list[str] = []
        notes: list[str] = []
        if row.price is None:
            flags.append("PRICE_UNAVAILABLE_OR_ROUNDED_ZERO")
        quality_inputs_missing = False
        if vol_mc is None:
            quality_inputs_missing = True
        elif vol_mc > float(thresholds.get("quality_vol_mktcap_max", 2.0)):
            flags.append(f"VOL24H/MktCap={vol_mc:.3f}>2")
        if oi_vol is None:
            quality_inputs_missing = True
        elif oi_vol < float(thresholds.get("quality_oi_vol_min", 0.1)):
            flags.append(f"OI/VOL24H={oi_vol:.3f}<0.1")
        if row.oi_usd is None:
            quality_inputs_missing = True
        elif row.oi_usd < float(thresholds.get("quality_oi_min_usd", 40_000_000.0)):
            flags.append(f"OI=${row.oi_usd:,.0f}<40m")
        if quality_inputs_missing:
            flags.append("QUALITY_DATA_INCOMPLETE")
        if s is None:
            flags.append("SPOT_IMBALANCE_NA")
        if delta_funding is not None:
            if row.fr_oiw_pct is not None and row.pfr_oiw_pct is not None and row.fr_oiw_pct * row.pfr_oiw_pct < 0:
                notes.append(
                    "PFR OI_W changes sign versus current FR OI_W; P1 has no numeric divergence threshold, warning only"
                )
            else:
                notes.append("PFR-vs-FR divergence has no numeric P1 threshold; not auto-flagged")
        base_symbol = row.symbol.upper().split(".", 1)[0].replace("_PERP", "").replace("-PERP", "")
        is_bitcoin = base_symbol in {"BTC", "BTCUSDT", "BTCUSDC", "BTCUSD"}
        if not is_bitcoin and row.btc_corr7d is not None and row.btc_corr7d > 0.7:
            notes.append("BTC Correlation 7D > 0.7: treat as BTC-correlated exposure, not an independent position")
        if row.oi_chg4h_usd is not None and abs(row.oi_chg4h_usd) < float(thresholds.get("oi_4h_small_usd", 2_000_000.0)):
            notes.append("OI 4H change is small in USD")
        if voluntary is True:
            notes.append("Liquidations explain <10% of absolute OI 24H decline; decline likely mostly voluntary")
        # P1 explicitly asks to surface all three divergences: F vs S,
        # price vs Spot, and price vs OI (same 24H window only).
        if f is not None and s is not None and f * s < 0:
            notes.append("Futures and Spot Imbalance have opposite signs; F/S divergence, no mutual flow confirmation")
        if row.chg24_pct is not None and s is not None:
            if row.chg24_pct > 0 and s < 0:
                notes.append("Price 24H is positive while Spot Imbalance is negative; price/Spot divergence")
            elif row.chg24_pct < 0 and s > 0:
                notes.append("Price 24H is negative while Spot Imbalance is positive; price/Spot divergence")
        if row.chg24_pct is not None and row.oi_chg24_pct is not None:
            if row.chg24_pct > 0 and row.oi_chg24_pct < 0:
                notes.append("Price 24H up while OI 24H down: possible short closing, not proof of a sustainable new LONG")
            elif row.chg24_pct < 0 and row.oi_chg24_pct < 0:
                notes.append("Price 24H down while OI 24H down: possible long closing, not proof of fresh SHORT opening")
            elif row.chg24_pct < 0 and row.oi_chg24_pct > 0:
                notes.append("Price 24H down while OI 24H up: positions are building during the decline; direction is not identified by OI alone")
        if (
            row.oi_chg4h_pct is not None
            and f is not None and s is not None
            and ((row.oi_chg4h_pct > 0 and f < 0 and s < 0)
                 or (row.oi_chg4h_pct < 0 and f > 0 and s > 0))
        ):
            notes.append("F/S flow and OI 4H move in different directions; lower confidence, observation only")
        if rel_strength is not None and btc_change is not None:
            if rel_strength > 0 and btc_change < 0:
                notes.append("Relative strength positive while BTC is down")
            elif rel_strength < 0 and btc_change > 0:
                notes.append("Relative weakness while BTC is up")

        enriched.append(EnrichedRow(
            market=row,
            futures_buy_share_pct=fut_buy,
            spot_buy_share_pct=spot_buy,
            spread=spread,
            delta_funding_oiw_pp=delta_funding,
            fr_venue_dispersion_pp=fr_disp,
            long_liq_share=long_share,
            liq_imbalance=liq_imb,
            rel_strength_vs_btc_pp=rel_strength,
            liquidation_coverage_of_oi_drop_pct=liq_coverage * 100.0 if liq_coverage is not None else None,
            oi_drop_mostly_voluntary=voluntary,
            f_adj=(f - median_f) if f is not None and median_f is not None else None,
            s_adj=(s - median_s) if s is not None and median_s is not None else None,
            f_rank_desc=f_r,
            s_rank_desc=s_r,
            f_strong_buy=f_strong_buy,
            f_strong_sell=f_strong_sell,
            s_strong_buy=s_strong_buy,
            s_strong_sell=s_strong_sell,
            quality_flags=flags,
            diagnostic_notes=notes,
        ))

    stats = MarketStats(
        row_count=len(rows),
        f_count=sum(r.futures_imbalance_4h is not None for r in rows),
        s_count=sum(r.spot_imbalance_4h is not None for r in rows),
        median_f=median_f,
        median_s=median_s,
        median_oi24_pct=median_oi24,
        median_oi4h_pct=median_oi4h,
        median_oi24_usd=median_oi24_usd,
        median_oi4h_usd=median_oi4h_usd,
        f_negative_count=sum(r.futures_imbalance_4h is not None and r.futures_imbalance_4h < 0 for r in rows),
        s_negative_count=sum(r.spot_imbalance_4h is not None and r.spot_imbalance_4h < 0 for r in rows),
        oi24_negative_count=sum(r.oi_chg24_pct is not None and r.oi_chg24_pct < 0 for r in rows),
        oi24_count=sum(r.oi_chg24_pct is not None for r in rows),
        price_positive_count=sum(r.chg24_pct is not None and r.chg24_pct > 0 for r in rows),
        price_negative_count=sum(r.chg24_pct is not None and r.chg24_pct < 0 for r in rows),
        price_count=sum(r.chg24_pct is not None for r in rows),
        btc_chg24_pct=btc_change,
        warnings=["Less than 20 rows: P1 top/bottom 15% ranks are unreliable"] if len(rows) < 20 else [],
    )
    return enriched, stats
