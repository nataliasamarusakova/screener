from __future__ import annotations

import logging
import statistics
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
)
from .telegram import TelegramClient, format_alert

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
            result["context_15m"] = {
                "last_closed_open_time_ms": latest.open_time_ms,
                "last_closed_change_pct": ((latest.close / latest.open) - 1.0) * 100.0 if latest.open else None,
            }
        except Exception as exc:
            result["context_15m"] = {"error": str(exc)}
    return result


def _preliminary_candidate_symbols(enriched: list[EnrichedRow], stats: Any, settings: Settings, store: Store) -> set[str]:
    symbols: set[str] = set()
    for row in enriched:
        history = store.recent_symbol_history(row.symbol, limit=100)
        spot_sequence = store.spot_sequence_status(
            row.symbol,
            required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
            interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
            tolerance_minutes=int(settings.p1.get("scenario", {}).get("snapshot_interval_tolerance_minutes", 10)),
        )
        draft = evaluate_scenarios(
            row, stats, settings.p1,
            technical={"spot_sequence_ok": spot_sequence},
            history=history,
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


def _collect_notifications(
    results: list[ScenarioResult],
    rows: list[EnrichedRow],
    settings: Settings,
    store: Store,
    contract_alerts_allowed: bool,
    dry_run: bool,
    no_new_data: bool = False,
) -> list[dict[str, Any]]:
    by_symbol = {row.symbol: row for row in rows}
    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_ids)
    eligible = {"OBSERVATION", "OBSERVATION_AFTER_BOUNCE", "TRIGGER"}
    if settings.telegram_send_wait_states:
        eligible.update({
            "WAIT_BOUNCE",
            "PREPARATION",
            "WAITING_FOR_HISTORY_OR_HIGHER_LOW",
            "WAITING_FOR_1H_STRUCTURE_BREAK",
        })
    outcomes: list[dict[str, Any]] = []
    for result in results:
        if result.status not in eligible or result.quality_flags:
            continue
        if not contract_alerts_allowed:
            outcomes.append({"symbol": result.symbol, "scenario": result.scenario, "sent": False, "reason": "contract/schema guard"})
            continue
        key = f"{result.symbol}|{result.scenario}|{result.status}"
        message = format_alert(result, by_symbol[result.symbol], result.technical)
        if dry_run or not telegram.configured:
            if no_new_data and store.notification_exists(key):
                # Preserve the old broadcast-level behavior for legacy state and dry runs.
                continue
            if not store.notification_due(key, settings.telegram_cooldown_hours):
                continue
            log.info("[DRY RUN / NO TELEGRAM]\n%s", message.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))
            outcomes.append({"symbol": result.symbol, "scenario": result.scenario, "sent": False, "reason": "dry-run" if dry_run else "telegram not configured", "text": message})
            continue

        recipients = list(dict.fromkeys(str(chat_id) for chat_id in settings.telegram_chat_ids))
        if no_new_data:
            pending = [
                chat_id for chat_id in recipients
                if not store.notification_exists(key, chat_id)
            ]
        else:
            pending = [
                chat_id for chat_id in recipients
                if store.notification_due(key, settings.telegram_cooldown_hours, chat_id)
            ]
        if not pending:
            continue

        delivery = telegram.send(message, chat_ids=pending)
        successes = [item for item in delivery if item.get("sent")]
        # Persist each recipient's confirmed success even when another recipient fails.
        for item in successes:
            store.record_notification(
                key, item.get("message_id"), chat_id=str(item["chat_id"])
            )
        all_success = bool(delivery) and len(successes) == len(delivery)
        outcomes.append({
            "symbol": result.symbol, "scenario": result.scenario,
            "sent": all_success, "delivery": delivery,
        })
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
        if not unchanged:
            store.add_snapshot(scrape.fetched_at, scrape.contract_type, scrape.rows)
        enriched, stats = analyze_universe(scrape.rows, settings.p1)
        if unchanged:
            stats.warnings.append("No new Coinalyze table data: parsed values fully match the previous saved snapshot")
            log.info("Coinalyze values unchanged vs the previous saved snapshot")
        if scrape.missing_columns:
            stats.warnings.append("Missing recognized optional/required columns: " + ", ".join(scrape.missing_columns))
        if scrape.missing_required_columns:
            stats.warnings.append("Missing REQUIRED P1 columns: " + ", ".join(scrape.missing_required_columns))
        if scrape.contract_type == "UNKNOWN":
            stats.warnings.append("Perpetual vs All Contracts could not be verified")
        candidate_symbols = _preliminary_candidate_symbols(enriched, stats, settings, store)
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
            history = store.recent_symbol_history(row.symbol, limit=100)
            spot_sequence = store.spot_sequence_status(
                row.symbol,
                required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
                interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
                tolerance_minutes=int(settings.p1.get("scenario", {}).get("snapshot_interval_tolerance_minutes", 10)),
            )
            technical = {"spot_sequence_ok": spot_sequence, **technical_by_symbol.get(row.symbol, {})}
            scenario_results = evaluate_scenarios(row, stats, settings.p1, technical=technical, history=history)
            for scenario_result in scenario_results:
                # Preserve both global diagnostics (order book/context) and scenario-specific predicates.
                scenario_result.technical = {**technical, **scenario_result.technical}
            results.extend(scenario_results)

        allowed, guard_reason = _can_send_alerts(scrape, settings)
        if not allowed:
            log.warning("Telegram alert guard: %s", guard_reason)
        notifications = _collect_notifications(results, enriched, settings, store, allowed, use_dry_run, no_new_data=unchanged)
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
