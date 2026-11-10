from __future__ import annotations

import hashlib
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
from .telegram import TelegramClient, format_alert, format_market_summary

log = logging.getLogger(__name__)


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
        candles4h = client.fetch_candles(symbol, "4h", settings.binance_kline_limit)
        result["context_4h"] = _context_4h(candles4h, left=left, right=right, lookback=lookback)
    except Exception as exc:
        result["context_4h"] = {"trend": f"unavailable: {exc}"}
    try:
        candles1h = client.fetch_candles(symbol, "1h", settings.binance_kline_limit)
        result["higher_low"] = higher_low(candles1h, left, right, lookback)
        result["trend_retest_long"] = breakout_and_retest(
            candles1h, "LONG", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
        )
        result["trend_retest_short"] = breakout_and_retest(
            candles1h, "SHORT", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
        )
        result["bearish_structure_break"] = bearish_structure_break(candles1h, left, right, lookback)
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
                symbol,
                limit=settings.binance_orderbook_limit,
                notional_usd=settings.execution_notional_usd,
            )
        except Exception as exc:
            result["orderbook"] = {"error": str(exc)}
    result["execution_cost_assumptions"] = {"taker_fee_bps_per_side": settings.trading_fee_bps_per_side}
    if bool(settings.p1.get("scanner", {}).get("fetch_15m_context", True)):
        try:
            candles15m = client.fetch_candles(symbol, "15m", settings.binance_kline_limit)
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


def _spot_sequence_for_current_poll(
    store: Store,
    symbol: str,
    settings: Settings,
    *,
    current_snapshot_available: bool,
    contract_type: str | None = None,
) -> bool | None:
    """Only evaluate a P1 Spot sequence when the poll maps to a committed observation.

    A changed but unsaved poll is useful for live diagnostics, but it cannot
    replace the latest element of a multi-snapshot history sequence.
    """
    if not current_snapshot_available:
        return None
    scenario_cfg = settings.p1.get("scenario", {})
    return store.spot_sequence_status(
        symbol,
        required=int(scenario_cfg.get("spot_accumulation_required_snapshots", 3)),
        interval_hours=float(scenario_cfg.get("snapshot_interval_hours", 4.0)),
        tolerance_minutes=int(scenario_cfg.get("snapshot_interval_tolerance_minutes", 10)),
        contract_type=contract_type,
    )


def _preliminary_candidate_symbols(
    enriched: list[EnrichedRow],
    stats: Any,
    settings: Settings,
    store: Store,
    *,
    current_snapshot_available: bool = True,
    contract_type: str | None = None,
) -> set[str]:
    symbols: set[str] = set()
    for row in enriched:
        history = store.recent_symbol_history(row.symbol, limit=100, contract_type=contract_type)
        spot_sequence = _spot_sequence_for_current_poll(
            store,
            row.symbol,
            settings,
            current_snapshot_available=current_snapshot_available,
            contract_type=contract_type,
        )
        draft = evaluate_scenarios(
            row, stats, settings.p1,
            technical={"spot_sequence_ok": spot_sequence},
            history=history,
            current_snapshot_available=current_snapshot_available,
        )
        if any(r.near_candidate or r.status in {"PREPARATION", "WAIT_BOUNCE"} for r in draft):
            symbols.add(row.symbol)
    max_candidates = int(settings.p1.get("scanner", {}).get("max_chart_candidates", 40))
    if len(symbols) > max_candidates:
        log.warning("Chart candidates capped: %d -> %d", len(symbols), max_candidates)
        # Stable deterministic cut: most relevant by OI USD descending.
        oi_by_symbol = {r.symbol: (r.market.oi_usd or 0.0) for r in enriched}
        symbols = set(sorted(symbols, key=lambda s: oi_by_symbol.get(s, 0.0), reverse=True)[:max_candidates])
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
    store: Store,
    rows: list[EnrichedRow],
    fetched_at: datetime,
    unchanged: bool = False,
    snapshot_saved: bool = False,
    snapshot_interval_hours: float = 4.0,
    snapshot_interval_tolerance_minutes: int = 10,
    contract_type: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare the current poll with the correct last committed observation.

    When the poll was saved, compare the two newest committed snapshots. When it
    was not saved, compare the live poll with the latest committed snapshot.
    An unchanged poll is explicitly marked as no new data and produces zero
    deltas rather than pretending the polling timestamp is a new observation.
    """
    deltas: dict[str, dict[str, Any]] = {}
    for enriched in rows:
        history = store.recent_symbol_history(enriched.symbol, limit=2, contract_type=contract_type)
        if not history:
            continue
        if snapshot_saved:
            if len(history) < 2:
                continue
            previous, current = history[-2], history[-1]
        elif unchanged:
            previous = history[-1]
            current = dict(previous)
        else:
            previous = history[-1]
            current = enriched.market.to_dict()
            current["snapshot_at"] = fetched_at.isoformat()

        try:
            previous_at = datetime.fromisoformat(previous["snapshot_at"])
            current_at = datetime.fromisoformat(current["snapshot_at"])
            if previous_at.tzinfo is None:
                previous_at = previous_at.replace(tzinfo=timezone.utc)
            if current_at.tzinfo is None:
                current_at = current_at.replace(tzinfo=timezone.utc)
            gap_hours = (current_at.astimezone(timezone.utc) - previous_at.astimezone(timezone.utc)).total_seconds() / 3600.0
        except (KeyError, TypeError, ValueError):
            continue

        def difference(key: str) -> float | None:
            old, new = previous.get(key), current.get(key)
            if old is None or new is None:
                return None
            try:
                old_num, new_num = float(old), float(new)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(old_num) or not math.isfinite(new_num):
                return None
            return new_num - old_num

        price_delta = None
        old_price, new_price = previous.get("price"), current.get("price")
        try:
            old_price_num, new_price_num = float(old_price), float(new_price)
            if old_price_num != 0 and math.isfinite(old_price_num) and math.isfinite(new_price_num):
                price_delta = (new_price_num / old_price_num - 1.0) * 100.0
        except (TypeError, ValueError, OverflowError):
            pass
        oi_delta_usd = difference("oi_usd")
        oi_delta_pct = None
        old_oi, new_oi = previous.get("oi_usd"), current.get("oi_usd")
        try:
            old_oi_num, new_oi_num = float(old_oi), float(new_oi)
            if old_oi_num != 0 and math.isfinite(old_oi_num) and math.isfinite(new_oi_num):
                oi_delta_pct = (new_oi_num / old_oi_num - 1.0) * 100.0
        except (TypeError, ValueError, OverflowError):
            pass
        expected_hours = max(0.0, float(snapshot_interval_hours))
        tolerance_hours = max(0, int(snapshot_interval_tolerance_minutes)) / 60.0
        if unchanged:
            note = "значения таблицы полностью совпадают с последним снимком: новых данных Coinalyze нет"
        elif gap_hours < 0:
            note = "время текущего опроса раньше последнего снимка; последовательность несопоставима"
        elif gap_hours < expected_hours:
            note = f"окна {expected_hours:g}H перекрываются; только предварительное наблюдение, не независимые наблюдения"
        elif gap_hours > expected_hours + tolerance_hours:
            note = f"интервал больше {expected_hours:g}H с учётом допуска; окна несопоставимы как одно последовательное наблюдение"
        elif gap_hours > expected_hours:
            note = (
                f"интервал {gap_hours:.2f}h: поздний опрос в пределах допуска "
                f"{tolerance_hours * 60:g} мин; последовательность считается приблизительно сопоставимой"
            )
        else:
            note = f"интервал ровно {expected_hours:g}h; последовательные окна сопоставимы"

        price_x_oi = "no_new_table_data" if unchanged else "unknown"
        if not unchanged and price_delta is not None and oi_delta_pct is not None:
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
            if gap_hours < 0:
                price_x_oi = "not_comparable_timestamp_reversed:" + price_x_oi
            elif gap_hours < expected_hours:
                price_x_oi = "preliminary_overlapping_windows:" + price_x_oi
            elif gap_hours > expected_hours + tolerance_hours:
                price_x_oi = "not_comparable_gap_outside_tolerance:" + price_x_oi
            elif gap_hours > expected_hours:
                price_x_oi = "comparable_late_interval_within_tolerance:" + price_x_oi
            else:
                price_x_oi = "comparable_target_interval:" + price_x_oi
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


def _apply_1h_structure_invalidation(result: ScenarioResult) -> None:
    """Prevent an already invalidated 1H setup from remaining a positive status."""
    code = result.scenario.split(" — ", 1)[0].strip().upper()
    technical = result.technical or {}
    sources = {
        "LONG A": "trend_retest_long",
        "LONG B": "higher_low",
        "LONG D": "higher_low",
        "SHORT A": "trend_retest_short",
        "SHORT D": "weak_bounce_rejection",
    }
    source_name = sources.get(code)
    if not source_name:
        return
    active_statuses = {
        "LONG A": {"OBSERVATION", "OBSERVATION_WITH_WARNING"},
        "LONG B": {"WAITING_FOR_HISTORY_OR_HIGHER_LOW", "OBSERVATION", "OBSERVATION_WITH_WARNING"},
        "LONG D": {"OBSERVATION", "OBSERVATION_WITH_WARNING"},
        "SHORT A": {"OBSERVATION", "OBSERVATION_WITH_WARNING"},
        "SHORT D": {"WAIT_BOUNCE", "OBSERVATION_AFTER_BOUNCE"},
    }
    if result.status not in active_statuses.get(code, set()):
        return
    state = technical.get(source_name)
    if not isinstance(state, dict) or state.get("invalidated") is not True:
        return
    if code == "SHORT D":
        # The liquidation-cascade context may still exist, but the rejection
        # has failed: revert to waiting for a fresh bounce/rejection.
        result.status = "WAIT_BOUNCE"
        result.reason = "The prior 1H resistance rejection was invalidated by a later close above resistance; wait for a new bounce/rejection."
    else:
        result.status = "INVALIDATED_1H_STRUCTURE"
        result.reason = f"The {source_name} 1H structure was invalidated by a later closed 1H candle; this setup is no longer active."

def _fingerprint_number(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return None


def _scenario_event_fingerprint(result: ScenarioResult) -> str:
    """Hash only meaningful scenario/structure state, not every market tick."""
    technical = result.technical or {}
    tracked = (
        "higher_low",
        "trend_retest_long",
        "trend_retest_short",
        "bearish_structure_break",
        "weak_bounce_rejection",
        "failed_breakout_rejection",
    )
    technical_state: dict[str, Any] = {}
    for key in tracked:
        value = technical.get(key)
        if not isinstance(value, dict):
            continue
        technical_state[key] = {
            field: _fingerprint_number(value.get(field))
            for field in (
                "confirmed", "invalidated", "level", "latest_low", "latest_open_time_ms",
                "breakout_open_time_ms", "retest_open_time_ms", "resistance",
                "candidate_resistance", "rejection_open_time_ms", "failed_level",
                "last_close", "invalidation_open_time_ms", "invalidation_close",
            )
            if field in value
        }
    context_4h = technical.get("context_4h") or {}
    if isinstance(context_4h, dict):
        ema20, ema50 = context_4h.get("ema20"), context_4h.get("ema50")
        technical_state["context_4h"] = {
            "trend": context_4h.get("trend"),
            "close_vs_support": (
                "below" if isinstance(context_4h.get("last_close"), (int, float))
                and isinstance(context_4h.get("last_confirmed_support"), (int, float))
                and context_4h["last_close"] < context_4h["last_confirmed_support"]
                else "above_or_unknown"
            ),
            "close_vs_resistance": (
                "above" if isinstance(context_4h.get("last_close"), (int, float))
                and isinstance(context_4h.get("last_confirmed_resistance"), (int, float))
                and context_4h["last_close"] > context_4h["last_confirmed_resistance"]
                else "below_or_unknown"
            ),
            "ema_relation": (
                "bullish" if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 > ema50
                else "bearish" if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 < ema50
                else "unknown"
            ),
        }
    context_15m = technical.get("context_15m") or {}
    if isinstance(context_15m, dict):
        # Directional change is meaningful for timing; candle timestamp is not
        # included, avoiding one repeated alert for every new 15m candle.
        technical_state["context_15m"] = {"direction": context_15m.get("direction")}
    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    payload = {
        "status": result.status,
        "checks": [(check.name, check.passed) for check in result.checks],
        "technical": technical_state,
        "invalidation": {
            "level": _fingerprint_number(invalidation.get("level")),
            "rule": invalidation.get("rule"),
            "source": invalidation.get("source"),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _market_summary_event_key(contract_type: str, fetched_at: datetime) -> str:
    timestamp = fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return f"MARKET_SUMMARY|{contract_type}|{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _collect_notifications(
    results: list[ScenarioResult],
    rows: list[EnrichedRow],
    settings: Settings,
    store: Store,
    contract_alerts_allowed: bool,
    dry_run: bool,
    no_new_data: bool = False,
    *,
    stats: Any | None = None,
    fetched_at: datetime | None = None,
    contract_type: str = "PERPETUAL",
    snapshot_deltas: dict[str, dict[str, Any]] | None = None,
    missing_columns: list[str] | None = None,
    emit_market_summary: bool = True,
) -> list[dict[str, Any]]:
    by_symbol = {row.symbol: row for row in rows}
    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_ids)
    recipients = list(dict.fromkeys(str(chat_id) for chat_id in settings.telegram_chat_ids))
    outcomes: list[dict[str, Any]] = []

    if not contract_alerts_allowed:
        log.warning("Notification delivery blocked by contract/schema guard")
        return [{"sent": False, "reason": "contract/schema guard"}]
    if not recipients:
        log.warning("No Telegram recipients configured; analysis remains available in dry-run logs")

    def deliver(
        event_key: str,
        message: str,
        *,
        kind: str,
        symbol: str | None = None,
        event_group: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        if event_group is not None and fingerprint is not None:
            pending = []
            for chat_id in recipients:
                last_state = store.latest_notification_state(event_group, chat_id)
                if last_state is None or last_state.get("fingerprint") != fingerprint:
                    # A new structural state is delivered immediately, even if
                    # this exact fingerprint appeared before and the cooldown
                    # has not elapsed (e.g. A -> B -> A).
                    pending.append(chat_id)
                elif not no_new_data and store.notification_due(
                    event_key, settings.telegram_cooldown_hours, chat_id
                ):
                    pending.append(chat_id)
            already_sent_without_recipient_tracking = not recipients and store.notification_exists(event_key)
            if already_sent_without_recipient_tracking:
                return
        elif no_new_data:
            pending = [chat_id for chat_id in recipients if not store.notification_exists(event_key, chat_id)]
            already_sent_without_recipient_tracking = not recipients and store.notification_exists(event_key)
            if already_sent_without_recipient_tracking:
                return
        else:
            pending = [
                chat_id for chat_id in recipients
                if store.notification_due(event_key, settings.telegram_cooldown_hours, chat_id)
            ]

        if recipients and not pending:
            return
        if not recipients or dry_run or not telegram.configured:
            reason = "dry-run" if dry_run else "telegram not configured"
            plain = message.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", "")
            log.info("[%s / %s]\n%s", kind.upper(), reason.upper(), plain)
            outcomes.append({"type": kind, "symbol": symbol, "sent": False, "reason": reason, "text": message})
            return

        delivery = telegram.send(message, chat_ids=pending)
        successes = [item for item in delivery if item.get("sent")]
        # Commit each confirmed delivery separately. A failure for another chat
        # must not cause a duplicate to a recipient who already received the alert.
        for item in successes:
            store.record_notification(
                event_key,
                item.get("message_id"),
                chat_id=str(item.get("chat_id")),
                event_group=event_group,
                fingerprint=fingerprint,
            )
        outcomes.append({
            "type": kind,
            "symbol": symbol,
            "sent": bool(delivery) and len(successes) == len(delivery),
            "delivery": delivery,
        })

    if emit_market_summary and stats is not None and fetched_at is not None and rows:
        # One whole-market digest per P1 4H observation window; do not create a
        # new summary event for every five-minute table change.
        summary_key = _market_summary_event_key(contract_type, fetched_at)
        strong_flow_quantile = float(settings.p1.get("thresholds", {}).get("strong_flow_quantile", 0.15))
        summary = format_market_summary(
            rows, stats, fetched_at, contract_type, missing_columns,
            strong_flow_quantile=strong_flow_quantile,
        )
        deliver(summary_key, summary, kind="market_summary")

    eligible = {"OBSERVATION", "OBSERVATION_WITH_WARNING", "OBSERVATION_AFTER_BOUNCE", "TRIGGER", "INVALIDATED_1H_STRUCTURE"}
    if settings.telegram_send_wait_states:
        eligible.update({
            "WAIT_BOUNCE",
            "PREPARATION",
            "WAITING_FOR_HISTORY_OR_HIGHER_LOW",
            "WAITING_FOR_1H_STRUCTURE_BREAK",
        })

    for result in results:
        if result.status not in eligible or result.quality_flags:
            continue
        row = by_symbol.get(result.symbol)
        if row is None:
            log.error("Cannot format Telegram notification: market row missing for %s", result.symbol)
            continue
        fingerprint = _scenario_event_fingerprint(result)
        key = f"{result.symbol}|{result.scenario}|{fingerprint}"
        message = format_alert(
            result,
            row,
            result.technical,
            fetched_at=fetched_at,
            snapshot_delta=(snapshot_deltas or {}).get(result.symbol),
        )
        event_group = f"{result.symbol}|{result.scenario}"
        deliver(
            key, message, kind="scenario", symbol=result.symbol,
            event_group=event_group, fingerprint=fingerprint,
        )
    return outcomes


def run_scan_once(settings: Settings, *, dry_run: bool | None = None) -> dict[str, Any]:
    store = Store(settings.data_dir)
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
            enriched, stats, settings, store,
            current_snapshot_available=(unchanged or snapshot_saved),
            contract_type=scrape.contract_type,
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
            # An identical table poll does not advance the saved observation time.
            # Keep the original timestamp so it cannot masquerade as a new snapshot.
            history = store.recent_symbol_history(
                row.symbol, limit=100, contract_type=scrape.contract_type
            )
            spot_sequence = _spot_sequence_for_current_poll(
                store,
                row.symbol,
                settings,
                current_snapshot_available=(unchanged or snapshot_saved),
                contract_type=scrape.contract_type,
            )
            technical = {"spot_sequence_ok": spot_sequence, **technical_by_symbol.get(row.symbol, {})}
            scenario_results = evaluate_scenarios(
                row, stats, settings.p1, technical=technical, history=history,
                current_snapshot_available=(unchanged or snapshot_saved),
            )
            for scenario_result in scenario_results:
                # Preserve both global diagnostics (order book/context) and scenario-specific predicates.
                scenario_result.technical = {**technical, **scenario_result.technical}
                _apply_1h_structure_invalidation(scenario_result)
            results.extend(scenario_results)

        allowed, guard_reason = _can_send_alerts(scrape, settings)
        if not allowed:
            log.warning("Telegram alert guard: %s", guard_reason)
        deltas_by_symbol = _snapshot_deltas(
            store,
            enriched,
            scrape.fetched_at,
            unchanged=unchanged,
            snapshot_saved=snapshot_saved,
            snapshot_interval_hours=snapshot_interval_hours,
            snapshot_interval_tolerance_minutes=snapshot_tolerance_minutes,
            contract_type=scrape.contract_type,
        )
        notifications = _collect_notifications(
            results,
            enriched,
            settings,
            store,
            allowed,
            use_dry_run,
            no_new_data=unchanged,
            stats=stats,
            fetched_at=scrape.fetched_at,
            contract_type=scrape.contract_type,
            snapshot_deltas=deltas_by_symbol,
            missing_columns=scrape.missing_columns,
            emit_market_summary=snapshot_saved,
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
