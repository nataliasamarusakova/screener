from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any

from .binance import BinanceClient
from .coinalyze import IncompleteUniverseError, ScrapeResult, fetch_table
from .config import Settings
from .metrics import EnrichedRow, analyze_universe
from .models import Candle, ScenarioResult
from .scenarios import evaluate_scenarios
from .storage import Store
from .structure import (
    bearish_structure_break,
    breakout_and_retest,
    failed_breakout_rejection,
    higher_low,
    rising_1h,
    weak_bounce_rejection,
    confirmed_pivots,
    scenario_invalidation,
)
from .telegram import (
    TelegramClient, format_alert,
    format_cancellation_message, format_reversal_message,
)

log = logging.getLogger(__name__)


def _recipient_storage_key(chat_id: str, settings: Settings) -> str | None:
    """Create a stable, one-way pseudonym for a Telegram recipient.

    Prefer a dedicated secret so rotating the bot token does not change the
    storage identity. The bot token is a backward-compatible fallback.
    """
    secret = str(
        getattr(settings, "telegram_storage_hash_key", "")
        or getattr(settings, "telegram_bot_token", "")
        or ""
    )
    if not secret:
        return None
    digest = hmac.new(secret.encode("utf-8"), str(chat_id).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def _migrate_notification_recipients(store: Store, settings: Settings) -> int:
    """Migrate raw legacy chat IDs before the notification store is queried."""
    migrate = getattr(store, "migrate_notification_recipient_ids", None)
    if not callable(migrate):
        return 0
    secret = str(
        getattr(settings, "telegram_storage_hash_key", "")
        or getattr(settings, "telegram_bot_token", "")
        or ""
    )
    if not secret:
        return 0
    recipient_keys = [
        key for chat_id in getattr(settings, "telegram_chat_ids", [])
        if (key := _recipient_storage_key(str(chat_id), settings)) is not None
    ]
    migrated = migrate(lambda chat_id: _recipient_storage_key(chat_id, settings))
    expanded = 0
    expand_legacy = getattr(store, "migrate_redacted_broadcast_notifications", None)
    if callable(expand_legacy):
        expanded = expand_legacy(recipient_keys)
    if migrated:
        log.info("Migrated %d stored Telegram recipient identifier(s) to HMAC keys", migrated)
    if expanded:
        log.warning("Expanded %d redacted legacy notification state(s) across configured recipients", expanded)
    return migrated + expanded


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    result = statistics.mean(values[:period])
    for value in values[period:]:
        result = value * alpha + result * (1.0 - alpha)
    return result


def _context_4h(
    candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60
) -> dict[str, Any]:
    closes = [c.close for c in candles]
    highs, lows = confirmed_pivots(candles, left=left, right=right, lookback=lookback)
    ema20, ema50 = _ema(closes, 20), _ema(closes, 50)
    if ema20 is None or ema50 is None:
        trend = "insufficient candles"
    elif ema20 > ema50:
        trend = "EMA20 > EMA50 (bullish context)"
    elif ema20 < ema50:
        trend = "EMA20 < EMA50 (bearish context)"
    else:
        trend = "EMA20 = EMA50 (neutral context)"
    return {
        "trend": trend,
        "ema20": ema20,
        "ema50": ema50,
        "last_close": closes[-1] if closes else None,
        "last_open_time_ms": candles[-1].open_time_ms if candles else None,
        "last_close_time_ms": candles[-1].close_time_ms if candles else None,
        "last_confirmed_resistance": highs[-1][1] if highs else None,
        "last_confirmed_support": lows[-1][1] if lows else None,
    }


def _technical_for_symbol(client: BinanceClient, symbol: str, settings: Settings) -> dict[str, Any]:
    structure_cfg = settings.p1.get("structure", {})
    left = int(structure_cfg.get("swing_left_bars", 2))
    right = int(structure_cfg.get("swing_right_bars", 2))
    lookback = int(structure_cfg.get("pivot_lookback_bars", 60))
    retest_lookback = int(structure_cfg.get("retest_lookback_bars", 8))
    atr_tolerance = float(structure_cfg.get("retest_atr_tolerance", 0.25))

    result: dict[str, Any] = {}
    try:
        result["binance_symbol"] = client.resolve_symbol(symbol)
    except Exception as exc:
        result["binance_symbol"] = None
        result["binance_symbol_error"] = str(exc)
    if not result.get("binance_symbol"):
        result["chart_error"] = result.get("binance_symbol_error", "No active Binance USDT perpetual contract")
        return result
    # Use the exact tradable pair resolved through Binance exchangeInfo for every
    # market-data request; Coinalyze's contract labels are not Binance symbols.
    fetch_symbol = str(result["binance_symbol"]).upper()
    try:
        candles4h = client.fetch_candles(fetch_symbol, "4h", settings.binance_kline_limit)
        result["context_4h"] = _context_4h(candles4h, left=left, right=right, lookback=lookback)
    except Exception as exc:
        result["context_4h"] = {"trend": f"unavailable: {exc}"}
    try:
        candles1h = client.fetch_candles(fetch_symbol, "1h", settings.binance_kline_limit)
        result["higher_low"] = higher_low(candles1h, left, right, lookback)
        result["trend_retest_long"] = breakout_and_retest(
            candles1h, "LONG", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
        )
        result["trend_retest_short"] = breakout_and_retest(
            candles1h, "SHORT", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
        )
        short_retest = result["trend_retest_short"]
        short_reference_level = (
            short_retest.get("level") if short_retest.get("confirmed") is True else None
        )
        short_reference_pivot_time = (
            short_retest.get("pivot_open_time_ms") if short_retest.get("confirmed") is True else None
        )
        short_reference_break_time = (
            short_retest.get("breakout_open_time_ms") if short_retest.get("confirmed") is True else None
        )
        result["bearish_structure_break"] = bearish_structure_break(
            candles1h, left, right, lookback,
            reference_level=short_reference_level,
            reference_pivot_open_time_ms=short_reference_pivot_time,
            reference_break_open_time_ms=short_reference_break_time,
        )
        result["weak_bounce_rejection"] = weak_bounce_rejection(
            candles1h,
            left=left,
            right=right,
            lookback=lookback,
            wick_body_ratio=float(structure_cfg.get("weak_rejection_wick_body_ratio", 0.75)),
            atr_tolerance=float(structure_cfg.get("weak_rejection_atr_tolerance", 0.5)),
        )
        result["failed_breakout_rejection"] = failed_breakout_rejection(candles1h, left, right, lookback)
        result["rising_1h"] = rising_1h(candles1h)
        result["last_closed_1h"] = {
            "open_time_ms": candles1h[-1].open_time_ms,
            "close_time_ms": candles1h[-1].close_time_ms,
            "open": candles1h[-1].open,
            "high": candles1h[-1].high,
            "low": candles1h[-1].low,
            "close": candles1h[-1].close,
        }
        # The Coinalyze table has no 24H low column. Surface the low derived
        # from the last 24 *closed* Binance 1H candles, without inventing a
        # numeric definition of "near the low".
        last_24h = candles1h[-24:]
        if last_24h:
            low_24h = min(candle.low for candle in last_24h)
            high_24h = max(candle.high for candle in last_24h)
            last_close = candles1h[-1].close
            result["range_24h_1h"] = {
                "low": low_24h,
                "high": high_24h,
                "last_closed_close": last_close,
                "distance_from_low_pct": ((last_close / low_24h) - 1.0) * 100.0 if low_24h > 0 else None,
                "distance_from_high_pct": ((high_24h / last_close) - 1.0) * 100.0 if last_close > 0 else None,
                "source": "last 24 closed Binance 1H candles",
            }
    except Exception as exc:
        result["candle_error_1h"] = str(exc)
        result["higher_low"] = {"confirmed": False, "reason": f"1H candle data unavailable: {exc}"}
    if settings.binance_fetch_orderbook:
        try:
            result["orderbook"] = client.fetch_orderbook_diagnostics(
                fetch_symbol,
                limit=settings.binance_orderbook_limit,
                notional_usd=settings.execution_notional_usd,
            )
        except Exception as exc:
            result["orderbook"] = {"error": str(exc)}
    result["execution_cost_assumptions"] = {"taker_fee_bps_per_side": settings.trading_fee_bps_per_side}
    if bool(settings.p1.get("scanner", {}).get("fetch_15m_context", True)):
        try:
            candles15m = client.fetch_candles(fetch_symbol, "15m", settings.binance_kline_limit)
            latest = candles15m[-1]
            candle_change = ((latest.close / latest.open) - 1.0) * 100.0 if latest.open else None
            result["context_15m"] = {
                "last_closed_open_time_ms": latest.open_time_ms,
                "last_closed_close_time_ms": latest.close_time_ms,
                "open": latest.open,
                "high": latest.high,
                "low": latest.low,
                "close": latest.close,
                "last_closed_change_pct": candle_change,
                "direction": "BULLISH" if latest.close > latest.open else "BEARISH" if latest.close < latest.open else "DOJI",
                "source": "last closed Binance 15m candle",
                "use": "entry-timing context only; P1 defines no separate numeric 15m trigger threshold",
            }
        except Exception as exc:
            result["context_15m"] = {"error": str(exc)}
    return result


def _preliminary_candidate_symbols(
    enriched: list[EnrichedRow], stats: Any, settings: Settings, store: Store,
    *, contract_type: str | None = None, as_of: datetime | None = None,
) -> set[str]:
    symbols: set[str] = set()
    # Continue monitoring 1H invalidation for every previously delivered active
    # setup, even after its Coinalyze table filters stop passing.
    recipient_keys = [
        key for chat_id in settings.telegram_chat_ids
        if (key := _recipient_storage_key(str(chat_id), settings)) is not None
    ]
    active_symbols = {
        str(item.get("event_group") or "").split("|", 1)[0]
        for item in store.active_notification_states(recipient_keys=recipient_keys)
        if str(item.get("event_group") or "").split("|", 1)[0]
    }
    # Active alerts remain chart candidates even when a coin disappears from
    # the current Coinalyze universe. Absence from that table is not cancellation.
    symbols.update(active_symbols)
    for row in enriched:
        history = store.recent_symbol_history(row.symbol, limit=100, contract_type=contract_type)
        spot_sequence = store.spot_sequence_status(
            row.symbol,
            required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
            interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
            tolerance_minutes=int(settings.p1.get("scenario", {}).get("snapshot_interval_tolerance_minutes", 10)),
            as_of=as_of,
            contract_type=contract_type,
        )
        draft = evaluate_scenarios(
            row, stats, settings.p1,
            technical={"spot_sequence_ok": spot_sequence},
            history=history,
            as_of=as_of,
        )
        if any(r.near_candidate or r.status in {"PREPARATION", "WAIT_BOUNCE"} for r in draft):
            symbols.add(row.symbol)
    max_candidates = int(settings.p1.get("scanner", {}).get("max_chart_candidates", 40))
    if len(symbols) > max_candidates:
        # Active positions/events must stay monitored; only draft candidates are
        # subject to the cap.
        active_present = symbols & active_symbols
        remaining = symbols - active_present
        oi_by_symbol = {r.symbol: (r.market.oi_usd or 0.0) for r in enriched}
        room = max(0, max_candidates - len(active_present))
        selected = set(sorted(remaining, key=lambda s: oi_by_symbol.get(s, 0.0), reverse=True)[:room])
        symbols = active_present | selected
        log.warning("Chart candidates capped: %d -> %d (active setups prioritized)", len(active_present | remaining), len(symbols))
    return symbols


def _can_send_alerts(scrape: ScrapeResult, settings: Settings) -> tuple[bool, str]:
    expected = str(settings.p1.get("universe", {}).get("expected_contract_type", "PERPETUAL")).upper()
    if scrape.contract_type == "ALL_CONTRACTS":
        return False, "Coinalyze page is All Contracts, P1 requires Perpetual contracts"
    if scrape.contract_type == "UNKNOWN" and not bool(settings.p1.get("universe", {}).get("allow_unknown_contract_type_for_alerts", False)):
        return False, "Could not verify Perpetual contracts view; notifications blocked"
    if scrape.contract_type != expected and scrape.contract_type != "UNKNOWN":
        return False, f"Contract type {scrape.contract_type} does not match expected {expected}"
    if scrape.missing_required_columns:
        return False, "Required P1 table columns missing: " + ", ".join(scrape.missing_required_columns)
    return True, "OK"


def _snapshot_deltas(
    store: Store, rows: list[EnrichedRow], fetched_at: datetime, unchanged: bool = False
) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}
    for enriched in rows:
        history = store.recent_symbol_history(enriched.symbol, limit=2)
        if not history:
            continue
        if unchanged:
            previous = history[-1]
            current = enriched.market.to_dict()
            current["snapshot_at"] = fetched_at.isoformat()
        else:
            if len(history) < 2:
                continue
            previous, current = history[-2], history[-1]
        try:
            previous_at = datetime.fromisoformat(previous["snapshot_at"])
            current_at = datetime.fromisoformat(current["snapshot_at"])
            gap_hours = (current_at - previous_at).total_seconds() / 3600.0
        except (KeyError, TypeError, ValueError):
            continue

        def difference(key: str) -> float | None:
            old, new = previous.get(key), current.get(key)
            return float(new) - float(old) if old is not None and new is not None else None

        price_delta = None
        old_price, new_price = previous.get("price"), current.get("price")
        if old_price is not None and new_price is not None and float(old_price) != 0:
            price_delta = (float(new_price) / float(old_price) - 1.0) * 100.0
        oi_delta_usd = difference("oi_usd")
        oi_delta_pct = None
        old_oi, new_oi = previous.get("oi_usd"), current.get("oi_usd")
        if old_oi is not None and new_oi is not None and float(old_oi) != 0:
            oi_delta_pct = (float(new_oi) / float(old_oi) - 1.0) * 100.0
        if unchanged:
            note = "значения таблицы полностью совпадают с предыдущим снимком: новых данных Coinalyze нет"
        elif gap_hours < 4.0:
            note = "окна 4H перекрываются; только предварительное наблюдение, не независимые наблюдения"
        elif gap_hours > 4.0:
            note = "интервал >4h; окна уже несопоставимы как одно последовательное 4H наблюдение"
        else:
            note = "интервал ровно 4h; приблизительно сопоставимые последовательные окна"

        price_x_oi = "unknown"
        if unchanged:
            price_x_oi = "no_new_table_data"
        elif price_delta is not None and oi_delta_pct is not None:
            if price_delta > 0 and oi_delta_pct > 0:
                price_x_oi = "price_up_oi_up"
            elif price_delta < 0 and oi_delta_pct > 0:
                price_x_oi = "price_down_oi_up"
            elif price_delta > 0 and oi_delta_pct < 0:
                price_x_oi = "price_up_oi_down_possible_short_closing"
            elif price_delta < 0 and oi_delta_pct < 0:
                price_x_oi = "price_down_oi_down_possible_long_closing"
            else:
                price_x_oi = "mixed_or_flat"
            if gap_hours < 4.0:
                price_x_oi = "preliminary_overlapping_windows:" + price_x_oi
            elif gap_hours > 4.0:
                price_x_oi = "not_comparable_gap_over_4h:" + price_x_oi
            else:
                price_x_oi = "comparable_4h_interval:" + price_x_oi
        deltas[enriched.symbol] = {
            "interval_hours": round(gap_hours, 3),
            "window_note": note,
            "price_change_pct": round(price_delta, 6) if price_delta is not None else None,
            "oi_change_usd": round(oi_delta_usd, 2) if oi_delta_usd is not None else None,
            "oi_change_pct": round(oi_delta_pct, 6) if oi_delta_pct is not None else None,
            "price_x_oi_classification": price_x_oi,
            "futures_imbalance_delta": round(difference("futures_imbalance_4h"), 6) if difference("futures_imbalance_4h") is not None else None,
            "spot_imbalance_delta": round(difference("spot_imbalance_4h"), 6) if difference("spot_imbalance_4h") is not None else None,
            "funding_oiw_delta_pp": round(difference("fr_oiw_pct"), 6) if difference("fr_oiw_pct") is not None else None,
        }
    return deltas


def _scenario_code(scenario: str) -> str:
    return str(scenario or "").split(" — ", 1)[0].strip().upper()


def _confirmation_anchor(result: ScenarioResult) -> int | None:
    technical = result.technical or {}
    code = _scenario_code(result.scenario)
    fields = {
        "LONG A": ("trend_retest_long", "retest_open_time_ms"),
        "LONG B": ("higher_low", "latest_open_time_ms"),
        "LONG D": ("higher_low", "latest_open_time_ms"),
        "SHORT A": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT B": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT C": ("bearish_structure_break", "break_open_time_ms"),
        "SHORT D": ("weak_bounce_rejection", "rejection_open_time_ms"),
        "SHORT E": ("failed_breakout_rejection", "failure_open_time_ms"),
    }.get(code)
    if fields is None:
        return None
    value = (technical.get(fields[0]) or {}).get(fields[1])
    return value if isinstance(value, int) and value > 0 else None


def _has_required_confirmation(result: ScenarioResult, row: EnrichedRow) -> bool:
    """Fail closed: P1 checks, a specific chart trigger and an invalidation level are mandatory."""
    if result.status not in {"CONFIRMED_SIGNAL", "CONFIRMED_SIGNAL_WITH_WARNING"}:
        return False
    if not result.complete or result.quality_flags:
        return False
    technical = result.technical or {}
    pair = str(technical.get("binance_symbol") or "").upper()
    if not pair.endswith("USDT") or not pair[:-4].isalnum():
        return False
    last = technical.get("last_closed_1h") or {}
    close = last.get("close")
    open_ms = last.get("open_time_ms")
    close_ms = last.get("close_time_ms")
    if not isinstance(close, (int, float)) or not math.isfinite(float(close)):
        return False
    if not isinstance(open_ms, int) or not isinstance(close_ms, int) or close_ms < open_ms:
        return False
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if close_ms > now_ms + 5_000 or now_ms - close_ms > 75 * 60_000:
        return False
    if close_ms - open_ms != 60 * 60_000 - 1:
        return False

    code = _scenario_code(result.scenario)
    gates = {
        "LONG A": ("trend_retest_long", "confirmed"),
        "LONG B": ("higher_low", "confirmed"),
        "LONG D": ("higher_low", "confirmed"),
        "SHORT A": ("trend_retest_short", "confirmed"),
        "SHORT B": ("trend_retest_short", "confirmed"),
        "SHORT C": ("bearish_structure_break", "confirmed"),
        "SHORT D": ("weak_bounce_rejection", "confirmed"),
        "SHORT E": ("failed_breakout_rejection", "confirmed"),
    }
    gate = gates.get(code)
    if gate is None:  # LONG C has an undefined "significant short liquidation" threshold in P1.
        return False
    state = technical.get(gate[0]) or {}
    if state.get(gate[1]) is not True:
        return False
    if _confirmation_anchor(result) is None:
        return False
    if code == "LONG B" and technical.get("spot_sequence_ok") is not True:
        return False
    if code == "SHORT C" and (technical.get("trend_retest_short") or {}).get("confirmed") is not True:
        return False
    if code == "SHORT D":
        coverage = row.liquidation_coverage_of_oi_drop_pct
        if coverage is None or coverage < 10.0:
            return False

    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    level = invalidation.get("level")
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not math.isfinite(float(level)) or level <= 0:
        return False
    breached = close < level if result.direction.upper() == "LONG" else close > level
    if breached:
        return False

    context = technical.get("context_4h") or {}
    close_4h = context.get("last_close")
    close_4h_ms = context.get("last_close_time_ms")
    if not isinstance(close_4h, (int, float)) or not isinstance(close_4h_ms, int):
        return False
    if close_4h_ms > now_ms + 5_000 or now_ms - close_4h_ms > 4 * 60 * 60_000 + 15 * 60_000:
        return False
    if result.direction.upper() == "LONG":
        support = context.get("last_confirmed_support")
        if not isinstance(support, (int, float)) or close_4h < support:
            return False
    else:
        resistance = context.get("last_confirmed_resistance")
        if not isinstance(resistance, (int, float)) or close_4h > resistance:
            return False
    return True


def _notification_group(result: ScenarioResult) -> str:
    # Compatible with the old v1.0.9 group key so existing notification history
    # remains active after deploying the lifecycle change.
    return f"{result.symbol}|{result.scenario}"


def _notification_identity(result: ScenarioResult) -> tuple[str, str, str, float] | None:
    technical = result.technical or {}
    pair = str(technical.get("binance_symbol") or "").upper()
    anchor = _confirmation_anchor(result)
    if not pair.endswith("USDT") or anchor is None:
        return None
    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    level = invalidation.get("level")
    if not isinstance(level, (int, float)) or not math.isfinite(float(level)) or float(level) <= 0:
        return None
    group = _notification_group(result)
    event_key = f"{group}|{anchor}"
    fingerprint = hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:20]
    return group, event_key, fingerprint, float(level)


def _event_group_parts(event_group: str) -> tuple[str, str, str]:
    parts = str(event_group or "").split("|", 1)
    symbol = parts[0] if parts else ""
    scenario = parts[1] if len(parts) > 1 else ""
    code = _scenario_code(scenario)
    direction = "LONG" if code.startswith("LONG") else "SHORT" if code.startswith("SHORT") else ""
    return symbol, scenario, direction


def _event_invalidated(last_closed_1h: dict[str, Any], active: dict[str, Any]) -> tuple[bool, float | None, float | None]:
    level = active.get("invalidation_level")
    direction = str(active.get("direction") or "").upper()
    close = last_closed_1h.get("close")
    open_ms = last_closed_1h.get("open_time_ms")
    close_ms = last_closed_1h.get("close_time_ms")
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not math.isfinite(float(level)):
        return False, None, float(close) if isinstance(close, (int, float)) else None
    if direction not in {"LONG", "SHORT"} or not isinstance(close, (int, float)) or not math.isfinite(float(close)) or float(close) <= 0:
        return False, float(level), None
    if not isinstance(open_ms, int) or not isinstance(close_ms, int) or close_ms - open_ms != 60 * 60_000 - 1:
        return False, float(level), float(close)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if close_ms > now_ms + 5_000 or now_ms - close_ms > 75 * 60_000:
        return False, float(level), float(close)
    crossed = close < level if direction == "LONG" else close > level
    return bool(crossed), float(level), float(close)


def _send_one(
    telegram: TelegramClient, message: str, chat_id: str,
) -> dict[str, Any]:
    delivery = telegram.send(message, chat_ids=[chat_id])
    return delivery[0] if delivery else {"chat_id": chat_id, "sent": False, "error": "No delivery response"}


def _collect_notifications(
    results: list[ScenarioResult],
    rows: list[EnrichedRow],
    settings: Settings,
    store: Store,
    contract_alerts_allowed: bool,
    dry_run: bool,
    *,
    fetched_at: datetime | None = None,
    contract_type: str = "PERPETUAL",
    technical_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Deliver new confirmations in a batch; cancellations/reversals as standalone messages."""
    by_symbol = {row.symbol: row for row in rows}
    technical_by_symbol = technical_by_symbol or {}
    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_ids)
    actual_recipients = list(dict.fromkeys(str(item) for item in settings.telegram_chat_ids))
    recipient_to_chat = {
        key: chat_id
        for chat_id in actual_recipients
        if (key := _recipient_storage_key(chat_id, settings)) is not None
    }
    recipients = list(recipient_to_chat)
    _migrate_notification_recipients(store, settings)
    if not contract_alerts_allowed:
        log.warning(
            "New confirmations and reversals blocked by contract/schema guard; "
            "existing alerts will still be checked for confirmed 1H invalidation"
        )
    if not recipients or dry_run or not telegram.configured:
        reason = "dry-run" if dry_run else "telegram not configured"
        log.info("Telegram lifecycle notifications skipped: %s", reason)
        return []

    results_by_group = {_notification_group(item): item for item in results}
    confirmed: dict[str, list[tuple[ScenarioResult, EnrichedRow, tuple[str, str, str, float]]]] = {}
    confirmed_opposite: dict[str, list[tuple[ScenarioResult, EnrichedRow, tuple[str, str, str, float]]]] = {}
    for result in results if contract_alerts_allowed else []:
        row = by_symbol.get(result.symbol)
        if row is None or not _has_required_confirmation(result, row):
            continue
        identity = _notification_identity(result)
        if identity is None:
            continue
        confirmed.setdefault(result.symbol, []).append((result, row, identity))
        confirmed_opposite.setdefault(result.symbol, []).append((result, row, identity))

    active_states = store.active_notification_states(recipient_keys=recipients)
    # Legacy v1.0.9 records did not have a direction or invalidation_level. Derive
    # those from the current scenario's chart only once, then persist the upgrade.
    normalized_active: list[dict[str, Any]] = []
    for active in active_states:
        group = str(active.get("event_group") or "")
        symbol, scenario, inferred_direction = _event_group_parts(group)
        direction = str(active.get("direction") or inferred_direction).upper()
        level = active.get("invalidation_level")
        current_result = results_by_group.get(group)
        chart_technical = (
            current_result.technical or {}
            if current_result is not None
            else technical_by_symbol.get(symbol, {})
        )
        if not isinstance(level, (int, float)) or not math.isfinite(float(level)):
            invalidation_scenario = current_result.scenario if current_result is not None else scenario
            invalidation_direction = current_result.direction if current_result is not None else direction
            inferred = scenario_invalidation(
                invalidation_direction, invalidation_scenario, chart_technical
            ).get("level")
            if isinstance(inferred, (int, float)) and math.isfinite(float(inferred)) and inferred > 0:
                level = float(inferred)
                migrated_key = f"MIGRATED|{active.get('event_key')}|{level:.12g}"
                store.record_notification(
                    migrated_key, active.get("telegram_message_id"),
                    recipient_key=str(active.get("recipient_key") or active.get("chat_id")),
                    event_group=group, fingerprint=str(active.get("fingerprint") or "legacy"),
                    event_state="ACTIVE", direction=direction, invalidation_level=level,
                    pair=str(active.get("pair") or chart_technical.get("binance_symbol") or symbol),
                    scenario_code=_scenario_code(scenario),
                    contract_type=str(active.get("contract_type") or contract_type),
                )
                active = {**active, "event_key": migrated_key, "invalidation_level": level, "direction": direction}
        if not active.get("direction") and direction:
            active["direction"] = direction
        normalized_active.append(active)

    # A confirmed opposite setup is a reversal. Aggregate all previously active
    # same-side scenarios for the pair into one standalone reversal notification.
    reversed_groups: set[tuple[str, str]] = set()
    outcomes: list[dict[str, Any]] = []
    active_by_pair_chat: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for active in normalized_active:
        group = str(active.get("event_group") or "")
        symbol, _, _ = _event_group_parts(group)
        chat_id = str(active.get("chat_id") or "")
        if symbol and chat_id:
            active_by_pair_chat.setdefault((symbol, chat_id), []).append(active)

    for (symbol, chat_id), pair_actives in active_by_pair_chat.items():
        opposite_new = []
        for candidate in confirmed_opposite.get(symbol, []):
            new_direction = candidate[0].direction.upper()
            old_side_actives = [
                active for active in pair_actives
                if str(active.get("direction") or _event_group_parts(str(active.get("event_group") or ""))[2]).upper() != new_direction
            ]
            if not old_side_actives:
                continue
            last_closed = (candidate[0].technical or {}).get("last_closed_1h") or {}
            # A true reversal requires both: a fully confirmed opposite setup
            # and the old setup's own persisted 1H invalidation level to be broken.
            # This prevents contradictory LONG/SHORT alerts while the old setup
            # is still structurally valid.
            if all(_event_invalidated(last_closed, active)[0] for active in old_side_actives):
                opposite_new.append(candidate)
        if not opposite_new:
            continue
        new_result, new_row, new_identity = max(opposite_new, key=lambda item: item[2][1])
        new_group, new_event_key, new_fingerprint, new_level = new_identity
        new_direction = new_result.direction.upper()
        old_actives = [
            item for item in pair_actives
            if str(item.get("direction") or _event_group_parts(str(item.get("event_group") or ""))[2]).upper() != new_direction
        ]
        old_codes = sorted({_scenario_code(_event_group_parts(str(item.get("event_group") or ""))[1]) for item in old_actives})
        old_direction = old_actives[0].get("direction") or _event_group_parts(str(old_actives[0].get("event_group") or ""))[2]
        reversal_key = f"REVERSAL|{symbol}|{old_direction}|{new_direction}|{new_event_key}"
        if store.notification_exists(reversal_key, chat_id):
            for old in old_actives:
                old_group = str(old.get("event_group") or "")
                latest_old = store.latest_notification_state(old_group, chat_id) or old
                if str(latest_old.get("event_state") or "ACTIVE").upper() == "ACTIVE":
                    store.record_notification(
                        f"{reversal_key}|RECOVER_OLD_STATE|{old_group}",
                        latest_old.get("telegram_message_id"),
                        recipient_key=chat_id, event_group=old_group,
                        fingerprint=str(latest_old.get("fingerprint") or "legacy"),
                        event_state="REVERSED",
                        direction=str(latest_old.get("direction") or _event_group_parts(old_group)[2]),
                        invalidation_level=(float(latest_old["invalidation_level"]) if isinstance(latest_old.get("invalidation_level"), (int, float)) else None),
                        pair=str(latest_old.get("pair") or symbol),
                        scenario_code=_scenario_code(_event_group_parts(old_group)[1]),
                        contract_type=contract_type,
                        delivery_status=str(latest_old.get("delivery_status") or "SENT"),
                    )
                reversed_groups.add((old_group, chat_id))
            latest_new = store.latest_notification_state(new_group, chat_id)
            if latest_new is None or str(latest_new.get("event_state") or "").upper() != "ACTIVE":
                store.record_notification(
                    new_event_key, None, recipient_key=chat_id,
                    event_group=new_group, fingerprint=new_fingerprint,
                    event_state="ACTIVE", direction=new_direction,
                    invalidation_level=new_level, pair=str((new_result.technical or {}).get("binance_symbol") or symbol),
                    scenario_code=_scenario_code(new_result.scenario), contract_type=contract_type,
                    delivery_status="UNKNOWN",
                )
            continue
        if len(old_codes) > 1:
            old_scenario_label = ", ".join(old_codes)
        else:
            old_scenario_label = old_codes[0] if old_codes else "предыдущий сценарий"
        pair = str((new_result.technical or {}).get("binance_symbol") or old_actives[0].get("pair") or symbol)
        message = format_reversal_message(
            pair, str(old_direction), old_scenario_label, new_direction, _scenario_code(new_result.scenario)
        )
        delivery = _send_one(telegram, message, recipient_to_chat[chat_id])
        if delivery.get("sent") or delivery.get("delivery_unknown"):
            delivery_status = "SENT" if delivery.get("sent") else "UNKNOWN"
            store.record_notification(
                reversal_key, delivery.get("message_id"), recipient_key=chat_id,
                event_group=f"REVERSAL|{symbol}|{old_direction}|{new_direction}",
                fingerprint=new_fingerprint, event_state="SENT", direction=new_direction,
                pair=pair, scenario_code=_scenario_code(new_result.scenario), contract_type=contract_type,
                delivery_status=delivery_status, error=str(delivery.get("error") or "") or None,
            )
            for old in old_actives:
                old_group = str(old.get("event_group") or "")
                store.record_notification(
                    f"{reversal_key}|OLD_STATE|{old_group}", delivery.get("message_id"), recipient_key=chat_id,
                    event_group=old_group, fingerprint=str(old.get("fingerprint") or "legacy"),
                    event_state="REVERSED", direction=str(old_direction),
                    invalidation_level=float(old["invalidation_level"]) if isinstance(old.get("invalidation_level"), (int, float)) else None,
                    pair=str(old.get("pair") or pair), scenario_code=_event_group_parts(old_group)[1].split(" — ", 1)[0],
                    contract_type=contract_type, delivery_status=delivery_status,
                )
                reversed_groups.add((old_group, chat_id))
            store.record_notification(
                new_event_key, delivery.get("message_id"), recipient_key=chat_id,
                event_group=new_group, fingerprint=new_fingerprint, event_state="ACTIVE",
                direction=new_direction, invalidation_level=new_level, pair=pair,
                scenario_code=_scenario_code(new_result.scenario), contract_type=contract_type,
                delivery_status=delivery_status,
            )
            outcomes.append({"type": "reversal", "symbol": symbol, "sent": bool(delivery.get("sent")), "delivery": delivery})

    # Invalidate remaining ACTIVE scenarios by the exact 1H level saved when
    # their original message was sent. Each cancellation is its own Telegram post.
    for active in normalized_active:
        group = str(active.get("event_group") or "")
        symbol, scenario, direction = _event_group_parts(group)
        chat_id = str(active.get("chat_id") or "")
        if not chat_id or (group, chat_id) in reversed_groups:
            continue
        current_result = results_by_group.get(group)
        technical = (current_result.technical or {}) if current_result is not None else {}
        if not technical.get("last_closed_1h"):
            technical = technical_by_symbol.get(symbol, {})
        last = technical.get("last_closed_1h") or {}
        if not last:
            # No current Coinalyze row is not proof of invalidation. Only a
            # fresh closed Binance 1H candle crossing the persisted level can cancel.
            continue
        close_time = last.get("close_time_ms")
        close_time = close_time if isinstance(close_time, int) else "unknown"
        invalidated, level, _ = _event_invalidated(last, active)
        if not invalidated or level is None:
            continue
        cancel_key = f"CANCEL|{active.get('event_key')}|{close_time}"
        if store.notification_exists(cancel_key, chat_id):
            continue
        pair = str(active.get("pair") or technical.get("binance_symbol") or symbol)
        message = format_cancellation_message(pair, direction, _scenario_code(scenario), level)
        delivery = _send_one(telegram, message, recipient_to_chat[chat_id])
        if delivery.get("sent") or delivery.get("delivery_unknown"):
            active_contract_type = str(active.get("contract_type") or contract_type).upper()
            store.record_notification(
                cancel_key, delivery.get("message_id"), recipient_key=chat_id,
                event_group=group, fingerprint=str(active.get("fingerprint") or "legacy"),
                event_state="CANCELLED", direction=direction, invalidation_level=level,
                pair=pair, scenario_code=_scenario_code(scenario), contract_type=active_contract_type,
                delivery_status="SENT" if delivery.get("sent") else "UNKNOWN",
                error=str(delivery.get("error") or "") or None,
            )
        if delivery.get("sent"):
            outcomes.append({"type": "cancellation", "symbol": symbol, "sent": True, "delivery": delivery})

    # New confirmations are sent as separate Telegram posts, one signal per
    # message and recipient. An already ACTIVE group is never repeated because
    # its trigger timestamp/metrics changed.
    active_groups_by_chat = {
        (str(item.get("event_group") or ""), str(item.get("chat_id") or ""))
        for item in store.active_notification_states(recipient_keys=recipients)
    }
    normal_by_chat: dict[str, list[tuple[ScenarioResult, EnrichedRow, tuple[str, str, str, float]]]] = {chat: [] for chat in recipients}
    for symbol, items in confirmed.items():
        for result, row, identity in items:
            group, event_key, fingerprint, level = identity
            for chat_id in recipients:
                if store.notification_exists(event_key, chat_id):
                    continue
                if (group, chat_id) in active_groups_by_chat or (group, chat_id) in reversed_groups:
                    continue
                has_opposite_active = any(
                    _event_group_parts(active_group)[0] == symbol
                    and _event_group_parts(active_group)[2] != result.direction.upper()
                    and active_chat == chat_id
                    for active_group, active_chat in active_groups_by_chat
                )
                if has_opposite_active:
                    continue
                # A reversal message already carried this new scenario for this pair.
                reversal_group = f"REVERSAL|{symbol}|"
                if any(g.startswith(reversal_group) and c == chat_id for g, c in active_groups_by_chat):
                    continue
                normal_by_chat[chat_id].append((result, row, identity))

    for chat_id, items in normal_by_chat.items():
        for result, row, identity in items:
            group, event_key, fingerprint, level = identity
            try:
                message = format_alert(result, row, result.technical, fetched_at=fetched_at)
            except (TypeError, ValueError) as exc:
                log.error("Cannot format confirmed signal %s %s: %s", result.symbol, result.scenario, exc)
                continue

            # One call to sendMessage per signal. Do not combine messages,
            # prepend a batch heading, or place separator lines between signals.
            delivery = _send_one(telegram, message, recipient_to_chat[chat_id])
            if delivery.get("sent") or delivery.get("delivery_unknown"):
                store.record_notification(
                    event_key, delivery.get("message_id"), recipient_key=chat_id,
                    event_group=group, fingerprint=fingerprint, event_state="ACTIVE",
                    direction=result.direction, invalidation_level=level,
                    pair=str((result.technical or {}).get("binance_symbol") or row.symbol),
                    scenario_code=_scenario_code(result.scenario), contract_type=contract_type,
                    delivery_status="SENT" if delivery.get("sent") else "UNKNOWN",
                    error=str(delivery.get("error") or "") or None,
                )
            outcomes.append({
                "type": "confirmed_signal",
                "sent": bool(delivery.get("sent")),
                "symbol": result.symbol,
                "scenario": result.scenario,
                "delivery": delivery,
            })
    return outcomes

def run_scan_once(settings: Settings, *, dry_run: bool | None = None) -> dict[str, Any]:
    store = Store(settings.data_dir)
    _migrate_notification_recipients(store, settings)
    pruned = store.prune(snapshot_retention_days=30, run_retention_days=90, notification_retention_days=180)
    if any(pruned.values()):
        log.info("Persistent state pruned: %s", pruned)
    run_id = store.start_run()
    use_dry_run = settings.dry_run if dry_run is None else dry_run
    try:
        scrape = fetch_table(
            url=settings.coinalyze_url,
            p_sid=settings.coinalyze_p_sid,
            chat_sid=settings.coinalyze_chat_sid,
            max_pages=settings.coinalyze_max_pages,
            headless=settings.coinalyze_headless,
            timeout_ms=settings.coinalyze_timeout_ms,
        )
        unchanged = store.snapshot_unchanged(scrape.contract_type, scrape.rows)
        snapshot_cfg = settings.p1.get("scenario", {})
        snapshot_interval_hours = float(snapshot_cfg.get("snapshot_interval_hours", 4.0))
        snapshot_tolerance_minutes = int(snapshot_cfg.get("snapshot_interval_tolerance_minutes", 10))
        snapshot_saved = False
        if not unchanged and store.snapshot_due(
            scrape.fetched_at,
            interval_hours=snapshot_interval_hours,
            tolerance_minutes=snapshot_tolerance_minutes,
            contract_type=scrape.contract_type,
        ):
            store.add_snapshot(scrape.fetched_at, scrape.contract_type, scrape.rows)
            snapshot_saved = True
        enriched, stats = analyze_universe(scrape.rows, settings.p1)
        if unchanged:
            stats.warnings.append("No new Coinalyze table data: parsed values fully match the previous saved snapshot")
            log.info("Coinalyze values unchanged vs the previous saved snapshot")
        elif not snapshot_saved:
            stats.warnings.append(
                f"Market values changed, but durable history is sampled every ~{snapshot_interval_hours:g}h; "
                "this poll is not counted as an independent P1 snapshot"
            )
        if scrape.missing_columns:
            stats.warnings.append("Missing recognized optional/required columns: " + ", ".join(scrape.missing_columns))
        if scrape.missing_required_columns:
            stats.warnings.append("Missing REQUIRED P1 columns: " + ", ".join(scrape.missing_required_columns))
        if scrape.contract_type == "UNKNOWN":
            stats.warnings.append("Perpetual vs All Contracts could not be verified")
        candidate_symbols = _preliminary_candidate_symbols(
            enriched, stats, settings, store, contract_type=scrape.contract_type, as_of=scrape.fetched_at
        )
        binance = BinanceClient(
            settings.binance_base_url,
            settings.binance_request_interval_sec,
            http_proxy=settings.binance_http_proxy,
            vpn_enabled=settings.binance_vpn_enabled,
        )
        technical_by_symbol: dict[str, dict[str, Any]] = {}
        for index, symbol in enumerate(sorted(candidate_symbols), start=1):
            log.info("Fetching candles for chart candidate %d/%d: %s", index, len(candidate_symbols), symbol)
            technical_by_symbol[symbol] = _technical_for_symbol(binance, symbol, settings)

        results: list[ScenarioResult] = []
        for row in enriched:
            # Use only durable observations for multi-snapshot patterns. An
            # unchanged Coinalyze poll is deliberately not a new independent sample.
            history = store.recent_symbol_history(row.symbol, limit=100, contract_type=scrape.contract_type)
            spot_sequence = store.spot_sequence_status(
                row.symbol,
                required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
                interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
                tolerance_minutes=int(settings.p1.get("scenario", {}).get("snapshot_interval_tolerance_minutes", 10)),
                as_of=scrape.fetched_at,
                contract_type=scrape.contract_type,
            )
            technical = {"spot_sequence_ok": spot_sequence, **technical_by_symbol.get(row.symbol, {})}
            scenario_results = evaluate_scenarios(
                row, stats, settings.p1, technical=technical, history=history, as_of=scrape.fetched_at
            )
            for scenario_result in scenario_results:
                # Preserve both global diagnostics (order book/context) and scenario-specific predicates.
                scenario_result.technical = {**technical, **scenario_result.technical}
            results.extend(scenario_results)

        allowed, guard_reason = _can_send_alerts(scrape, settings)
        entry_allowed = allowed and not unchanged
        if unchanged:
            guard_reason = f"{guard_reason}; unchanged Coinalyze table: new position openings blocked"
        if not entry_allowed:
            log.warning("New position-entry notifications blocked: %s", guard_reason)
        notifications = _collect_notifications(
            results,
            enriched,
            settings,
            store,
            entry_allowed,
            use_dry_run,
            fetched_at=scrape.fetched_at,
            contract_type=scrape.contract_type,
            technical_by_symbol=technical_by_symbol,
        )
        visible_results = [r for r in results if r.near_candidate or r.status in {"PREPARATION", "WAIT_BOUNCE", "OBSERVATION_AFTER_BOUNCE"}]
        store.finish_run(run_id, len(scrape.rows), len(visible_results), "OK")
        log.info("Scan complete: rows=%d near-scenarios=%d contract=%s", len(scrape.rows), len(visible_results), scrape.contract_type)
        return {
            "status": "OK",
            "rows": len(scrape.rows),
            "contract_type": scrape.contract_type,
            "missing_columns": scrape.missing_columns,
            "missing_required_columns": scrape.missing_required_columns,
            "no_new_coinalyze_data": unchanged,
            "snapshot_saved": snapshot_saved,
            "stats": stats,
            "results": visible_results,
            "notifications": notifications,
            "alert_guard": guard_reason,
        }
    except IncompleteUniverseError as exc:
        store.finish_run(run_id, 0, 0, "INCOMPLETE_UNIVERSE", str(exc))
        log.exception("Coinalyze universe incomplete; no analysis/notifications sent")
        raise
    except Exception as exc:
        store.finish_run(run_id, 0, 0, "FAILED", str(exc))
        log.exception("P1 scan failed")
        raise
