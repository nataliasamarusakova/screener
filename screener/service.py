from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .binance import (
    BinanceClient, BinanceCooldownActiveError, BinanceScanAbortError,
    BinanceDataError, SpotDataError, SpotMarketClient,
)
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
    """Create the current stable, one-way pseudonym for a Telegram recipient.

    A dedicated secret is preferred so rotating the bot token does not change
    the storage identity; the bot token is the configured fallback.
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


def _technical_for_symbol(
    client: BinanceClient, symbol: str, settings: Settings, *, include_outcome_history: bool = False
) -> dict[str, Any]:
    structure_cfg = settings.p1.get("structure", {})
    left = int(structure_cfg.get("swing_left_bars", 2))
    right = int(structure_cfg.get("swing_right_bars", 2))
    lookback = int(structure_cfg.get("pivot_lookback_bars", 60))
    retest_lookback = int(structure_cfg.get("retest_lookback_bars", 8))
    atr_tolerance = float(structure_cfg.get("retest_atr_tolerance", 0.25))
    retest_max_extension_atr = float(structure_cfg.get("retest_max_extension_atr", 2.0))

    result: dict[str, Any] = {}
    try:
        result["binance_symbol"] = client.resolve_symbol(symbol)
    except BinanceScanAbortError:
        raise
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
    except BinanceScanAbortError:
        raise
    except Exception as exc:
        result["context_4h"] = {"trend": f"unavailable: {exc}"}
    try:
        candles1h = client.fetch_candles(
            fetch_symbol, "1h", 1000 if include_outcome_history else settings.binance_kline_limit
        )
        # The same closed 1H series feeds both active-signal outcome tracking
        # and the on-demand Futures taker-flow feature. The latter is also
        # required for first-time candidates, not only previously active alerts.
        result["_candles_1h"] = candles1h
        result["higher_low"] = higher_low(
            candles1h, left, right, lookback,
            max_age_bars=int(structure_cfg.get("higher_low_max_age_bars", 24)),
            min_rise_atr=float(structure_cfg.get("higher_low_min_rise_atr", 0.25)),
            max_extension_atr=float(structure_cfg.get("higher_low_max_extension_atr", 2.0)),
        )
        result["trend_retest_long"] = breakout_and_retest(
            candles1h, "LONG", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
            max_extension_atr=retest_max_extension_atr,
        )
        result["trend_retest_short"] = breakout_and_retest(
            candles1h, "SHORT", left=left, right=right, lookback=lookback,
            retest_lookback=retest_lookback, atr_tolerance=atr_tolerance,
            max_extension_atr=retest_max_extension_atr,
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
        result["failed_breakout_rejection"] = failed_breakout_rejection(
            candles1h, left, right, lookback,
            max_age_bars=int(structure_cfg.get("failed_breakout_max_age_bars", 3)),
        )
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
    except BinanceScanAbortError:
        raise
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
        except BinanceScanAbortError:
            raise
        except Exception as exc:
            result["orderbook"] = {"error": str(exc)}
    maker_fee_bps = float(getattr(settings, "maker_fee_bps", 2.0))
    taker_fee_bps = float(getattr(settings, "taker_fee_bps", 5.0))
    legacy_fee_bps = getattr(settings, "trading_fee_bps_per_side", None)
    if legacy_fee_bps is not None:
        # Settings loaded from the legacy environment variable map that value
        # to both maker/taker only when the new variables were not specified.
        maker_fee_bps = float(getattr(settings, "maker_fee_bps", legacy_fee_bps))
        taker_fee_bps = float(getattr(settings, "taker_fee_bps", legacy_fee_bps))
    result["execution_cost_assumptions"] = {
        "maker_fee_bps_per_side": maker_fee_bps,
        "taker_fee_bps_per_side": taker_fee_bps,
        "entry_execution_type": str(getattr(settings, "shadow_entry_execution_type", "TAKER")),
        "exit_execution_type": str(getattr(settings, "shadow_exit_execution_type", "TAKER")),
        "shadow_notional_usd": float(getattr(settings, "shadow_notional_usd", 1000.0)),
        "execution_type_status": "ASSUMED",
        "notional_status": "ASSUMED_FIXED_NOTIONAL",
        "funding_included": False,
        "slippage_included": False,
    }
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
        except BinanceScanAbortError:
            raise
        except Exception as exc:
            result["context_15m"] = {"error": str(exc)}
    return result


def _taker_flow_imbalance(candles: list[Candle]) -> float | None:
    """Return quote-volume taker imbalance only for fully covered candles."""
    if not candles:
        return None
    valid = [
        candle for candle in candles
        if candle.quote_volume > 0
        and candle.taker_buy_quote_volume is not None
        and math.isfinite(candle.taker_buy_quote_volume)
        and 0 <= candle.taker_buy_quote_volume <= candle.quote_volume
    ]
    # Do not silently turn partial candle coverage into a one-hour aggregate.
    if len(valid) != len(candles):
        return None
    quote_volume = sum(item.quote_volume for item in valid)
    buy_quote = sum(float(item.taker_buy_quote_volume) for item in valid)
    if quote_volume <= 0:
        return None
    return 2.0 * (buy_quote / quote_volume) - 1.0


def _taker_flow_status(
    candles: list[Candle], *, now_ms: int, max_age_ms: int,
    expected_count: int | None = None, interval_ms: int | None = None,
) -> str:
    """Explain why a source candle set can or cannot yield taker flow."""
    if not candles:
        return "NO_CANDLES"
    latest = candles[-1]
    age_ms = int(now_ms) - int(latest.close_time_ms)
    if age_ms < 0:
        return "FUTURE_CANDLE"
    if age_ms > max_age_ms:
        return "STALE"
    if expected_count is not None and len(candles) < expected_count:
        return "INSUFFICIENT_CANDLES"
    if interval_ms is not None and any(
        later.open_time_ms - earlier.open_time_ms != interval_ms
        for earlier, later in zip(candles, candles[1:])
    ):
        return "DATA_GAP"
    for candle in candles:
        if candle.taker_buy_quote_volume is None:
            return "MISSING_TAKER_BUY_QUOTE_VOLUME"
        if not math.isfinite(candle.taker_buy_quote_volume) or not math.isfinite(candle.quote_volume):
            return "INVALID_TAKER_VOLUME"
        if candle.quote_volume <= 0:
            return "ZERO_QUOTE_VOLUME"
        if not 0 <= candle.taker_buy_quote_volume <= candle.quote_volume:
            return "INVALID_TAKER_BUY_QUOTE_VOLUME"
    return "AVAILABLE"


def _oi_change_1h(records: list[dict[str, Any]], field: str) -> float | None:
    """Compare the newest OI observation to the closest point about one hour earlier."""
    if len(records) < 2:
        return None
    latest = records[-1]
    try:
        latest_time = int(latest["timestamp"])
        latest_value = float(latest[field])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    target = latest_time - 60 * 60_000
    candidates: list[tuple[int, float]] = []
    for item in records[:-1]:
        try:
            timestamp = int(item["timestamp"])
            value = float(item[field])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if timestamp <= target and math.isfinite(value) and value > 0:
            candidates.append((timestamp, value))
    if not candidates or latest_value <= 0 or not math.isfinite(latest_value):
        return None
    timestamp, previous_value = max(candidates, key=lambda pair: pair[0])
    # A sparse history is UNKNOWN, not an inferred unchanged OI value.
    if target - timestamp > 15 * 60_000:
        return None
    return (latest_value / previous_value - 1.0) * 100.0


def _market_context_for_symbol(
    futures_client: BinanceClient,
    spot_client: SpotMarketClient,
    symbol: str,
    futures_candles_1h: list[Candle],
    settings: Settings,
) -> dict[str, Any]:
    """Collect bounded on-demand Spot/OI/Funding context for one finalist."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    futures_flow_candles = futures_candles_1h[-1:]
    futures_flow_status = _taker_flow_status(
        futures_flow_candles, now_ms=now_ms, max_age_ms=70 * 60_000,
    )
    context: dict[str, Any] = {
        "futures_symbol": symbol.upper(),
        "futures_flow_1h": (
            _taker_flow_imbalance(futures_flow_candles)
            if futures_flow_status == "AVAILABLE" else None
        ),
        "futures_flow_status": futures_flow_status,
        "futures_flow_source": "Binance USDⓈ-M Futures last closed 1H candle",
        "futures_flow_last_closed_1h_open_time_ms": (
            futures_flow_candles[-1].open_time_ms if futures_flow_candles else None
        ),
        "spot_status": "UNKNOWN",
        "spot_flow_status": "NOT_CHECKED",
        "spot_symbol": None,
        "base_multiplier": None,
        "spot_flow_1h": None,
        "oi_history_status": "UNKNOWN",
        "oi_change_status": "UNKNOWN",
        "oi_change_base_1h_pct": None,
        "oi_change_usd_1h_pct": None,
        "oi_latest_base": None,
        "oi_latest_usd": None,
        "funding_status": "UNKNOWN",
        "funding_regular_status": "UNKNOWN",
        "funding_latest_regular": None,
        "funding_average_regular": None,
        "funding_latest_special": None,
    }
    oi_limit = int(settings.p1.get("scanner", {}).get("market_context_oi_limit", 500))
    funding_limit = int(settings.p1.get("scanner", {}).get("market_context_funding_limit", 200))
    kline_limit = int(settings.p1.get("scanner", {}).get("market_context_kline_limit", 300))

    try:
        oi_rows = futures_client.fetch_open_interest_history(symbol, period="5m", limit=oi_limit)
        if oi_rows:
            latest_oi_timestamp = int(oi_rows[-1].get("timestamp") or 0)
            oi_is_fresh = 0 <= now_ms - latest_oi_timestamp <= 15 * 60_000
            context.update({
                "oi_history_status": "AVAILABLE" if oi_is_fresh else "STALE",
                "oi_change_base_1h_pct": _oi_change_1h(oi_rows, "sumOpenInterest") if oi_is_fresh else None,
                "oi_change_usd_1h_pct": _oi_change_1h(oi_rows, "sumOpenInterestValue") if oi_is_fresh else None,
                # Preserve raw last values for diagnostics, but consumers must
                # respect oi_history_status before treating them as current.
                "oi_latest_base": oi_rows[-1].get("sumOpenInterest"),
                "oi_latest_usd": oi_rows[-1].get("sumOpenInterestValue"),
                "oi_latest_timestamp_ms": latest_oi_timestamp,
                "oi_observation_count": len(oi_rows),
            })
            context["oi_change_status"] = (
                "STALE" if not oi_is_fresh else
                "AVAILABLE" if context["oi_change_base_1h_pct"] is not None
                and context["oi_change_usd_1h_pct"] is not None else
                "INSUFFICIENT_HISTORY"
            )
        else:
            context["oi_history_status"] = "NO_DATA"
            context["oi_change_status"] = "NO_DATA"
    except BinanceScanAbortError:
        raise
    except BinanceDataError as exc:
        context["oi_history_status"] = "ERROR"
        context["oi_change_status"] = "ERROR"
        context["oi_error"] = str(exc)[:300]

    try:
        funding_rows = futures_client.fetch_funding_rate_history(symbol, limit=funding_limit)
        regular = [item for item in funding_rows if item.get("rateType") == "Regular"]
        special = [item for item in funding_rows if item.get("rateType") == "Special"]
        if funding_rows:
            context["funding_status"] = "AVAILABLE"
            context["funding_latest_regular"] = regular[-1] if regular else None
            context["funding_regular_status"] = "AVAILABLE" if regular else "NO_REGULAR_RATE"
            context["funding_average_regular"] = (
                sum(float(item["fundingRate"]) for item in regular[-3:]) / len(regular[-3:])
                if regular else None
            )
            context["funding_latest_special"] = special[-1] if special else None
            context["funding_unknown_rate_type_count"] = sum(
                item.get("rateType") not in {"Regular", "Special"} for item in funding_rows
            )
            context["funding_observation_count"] = len(funding_rows)
        else:
            context["funding_status"] = "NO_DATA"
            context["funding_regular_status"] = "NO_DATA"
    except BinanceScanAbortError:
        raise
    except BinanceDataError as exc:
        context["funding_status"] = "ERROR"
        context["funding_regular_status"] = "ERROR"
        context["funding_error"] = str(exc)[:300]

    try:
        resolution = spot_client.resolve_symbol(symbol)
    except SpotDataError as exc:
        context["spot_status"] = "ERROR"
        context["spot_flow_status"] = "ERROR"
        context["spot_error"] = str(exc)[:300]
        return context
    if resolution is None:
        context["spot_status"] = "SPOT_UNAVAILABLE"
        context["spot_flow_status"] = "SPOT_UNAVAILABLE"
        return context
    context.update({
        "spot_status": "AVAILABLE",
        "spot_symbol": resolution.spot_symbol,
        "spot_base_asset": resolution.spot_base_asset,
        "futures_base_asset": resolution.futures_base_asset,
        "base_multiplier": resolution.base_multiplier,
    })
    try:
        spot_candles = spot_client.fetch_candles(
            resolution.spot_symbol, "5m", limit=kline_limit
        )
        if len(spot_candles) >= 12:
            last_hour = spot_candles[-12:]
            age_ms = now_ms - spot_candles[-1].close_time_ms
            context["spot_last_closed_5m_open_time_ms"] = spot_candles[-1].open_time_ms
            context["spot_observation_count"] = len(spot_candles)
            context["spot_latest_candle_age_ms"] = age_ms
            context["spot_flow_status"] = _taker_flow_status(
                last_hour, now_ms=now_ms, max_age_ms=7 * 60_000,
                expected_count=12, interval_ms=5 * 60_000,
            )
            if context["spot_flow_status"] == "AVAILABLE":
                context["spot_status"] = "AVAILABLE"
                context["spot_flow_1h"] = _taker_flow_imbalance(last_hour)
                if context["spot_flow_1h"] is None:
                    context["spot_flow_status"] = "INVALID_AGGREGATE"
            elif context["spot_flow_status"] == "STALE":
                context["spot_status"] = "STALE"
            elif context["spot_flow_status"] == "DATA_GAP":
                context["spot_status"] = "DATA_GAP"
            else:
                # Pair exists; data coverage is the problem, not symbol mapping.
                context["spot_status"] = "AVAILABLE"
        else:
            context["spot_status"] = "NO_DATA"
            context["spot_flow_status"] = "INSUFFICIENT_CANDLES"
    except SpotDataError as exc:
        # A stale/de-listed/missing Spot pair is isolated from the Futures scan.
        context["spot_status"] = "SPOT_UNAVAILABLE" if "-1121" in str(exc) else "ERROR"
        context["spot_flow_status"] = context["spot_status"]
        context["spot_error"] = str(exc)[:300]
    return context


def _market_context_quality(context: dict[str, Any] | None) -> dict[str, Any]:
    """Separate strategy-check completeness from optional market-data coverage."""
    if not isinstance(context, dict):
        missing = [
            "futures_flow_1h", "spot_flow_1h", "oi_change_base_1h_pct",
            "oi_change_usd_1h_pct", "funding_latest_regular",
        ]
        return {
            "market_context_status": "NOT_COLLECTED",
            "market_context_missing_fields": missing,
            "market_context_missing_count": len(missing),
            "market_context_completeness_pct": 0.0,
            "market_context_field_statuses": {},
        }

    field_statuses = {
        "futures_flow_1h": context.get("futures_flow_status", "UNKNOWN"),
        "spot_flow_1h": context.get("spot_flow_status", "UNKNOWN"),
        "oi_change_base_1h_pct": context.get("oi_change_status", "UNKNOWN"),
        "oi_change_usd_1h_pct": context.get("oi_change_status", "UNKNOWN"),
        "funding_latest_regular": context.get("funding_regular_status", "UNKNOWN"),
    }
    missing = [
        field for field, status in field_statuses.items()
        if status != "AVAILABLE" or context.get(field) is None
    ]
    total = len(field_statuses)
    complete = total - len(missing)
    return {
        "market_context_status": "COMPLETE" if not missing else "PARTIAL",
        "market_context_missing_fields": missing,
        "market_context_missing_count": len(missing),
        "market_context_completeness_pct": round(100.0 * complete / total, 1),
        "market_context_field_statuses": field_statuses,
    }


def _fee_bps_for_execution(
    execution_type: str,
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> float:
    normalized = str(execution_type or "").upper()
    if normalized == "MAKER":
        return float(maker_fee_bps)
    if normalized == "TAKER":
        return float(taker_fee_bps)
    raise ValueError("Execution type must be MAKER or TAKER")


def _shadow_fee_accounting(
    *,
    entry: float,
    exit_legs: list[tuple[float, float]] | None,
    risk_abs: float,
    direction: str,
    notional_usd: float,
    entry_fee_bps: float,
    exit_fee_bps: float,
    entry_execution_type: str,
    exit_execution_type: str,
) -> dict[str, Any]:
    """Calculate assumed fees and realized PnL from traded notional.

    ``exit_legs`` contains (fraction of original quantity, execution price).
    An empty list means no exit yet; ``None`` means an ambiguous/unresolved
    intrabar path, so no exit fee or PnL is imputed.
    """
    values = (entry, risk_abs, notional_usd, entry_fee_bps, exit_fee_bps)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Shadow fee inputs must be finite")
    if entry <= 0 or risk_abs <= 0 or notional_usd <= 0:
        raise ValueError("Entry, initial risk and fixed notional must be positive")
    if entry_fee_bps < 0 or exit_fee_bps < 0:
        raise ValueError("Fee rates cannot be negative")
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("Direction must be LONG or SHORT")

    quantity = notional_usd / entry
    initial_risk_usd = quantity * risk_abs
    entry_fee_usd = notional_usd * entry_fee_bps / 10_000.0
    base: dict[str, Any] = {
        "entry_execution_type": entry_execution_type,
        "exit_execution_type": exit_execution_type,
        "entry_execution_type_status": "ASSUMED",
        "exit_execution_type_status": "ASSUMED",
        "execution_type_status": "ASSUMED",
        "entry_fee_bps": entry_fee_bps,
        "exit_fee_bps": exit_fee_bps,
        "entry_notional_usd": notional_usd,
        "exit_notional_usd": None,
        "entry_fee_usd": entry_fee_usd,
        "exit_fee_usd": None,
        "total_fee_usd": None,
        "gross_pnl_usd": None,
        "net_pnl_usd": None,
        "initial_risk_usd": initial_risk_usd,
        "position_quantity_assumed": quantity,
        "gross_r": None,
        "net_r": None,
        "notional_assumption": "ASSUMED_FIXED_NOTIONAL",
        "pnl_basis": "REALIZED_ONLY_EXCLUDES_UNREALIZED_PNL",
        "funding_included": False,
        "slippage_included": False,
    }
    if exit_legs is None:
        base["fee_accounting_status"] = "AMBIGUOUS_EXIT_NOT_PRICED"
        return base

    normalized_legs: list[tuple[float, float]] = []
    for fraction, price in exit_legs:
        fraction = float(fraction)
        price = float(price)
        if not math.isfinite(fraction) or not math.isfinite(price):
            raise ValueError("Exit leg values must be finite")
        if fraction < 0 or fraction > 1 or price <= 0:
            raise ValueError("Exit fraction must be in [0, 1] and price positive")
        normalized_legs.append((fraction, price))

    exit_notional_usd = sum(quantity * fraction * price for fraction, price in normalized_legs)
    exit_fee_usd = exit_notional_usd * exit_fee_bps / 10_000.0
    total_fee_usd = entry_fee_usd + exit_fee_usd
    base.update({
        "exit_notional_usd": exit_notional_usd,
        "exit_fee_usd": exit_fee_usd,
        "total_fee_usd": total_fee_usd,
    })
    if not normalized_legs:
        base["fee_accounting_status"] = "ENTRY_FEE_KNOWN_EXIT_PENDING"
        return base

    direction_sign = 1.0 if direction == "LONG" else -1.0
    gross_pnl_usd = sum(
        direction_sign * (price - entry) * quantity * fraction
        for fraction, price in normalized_legs
    )
    net_pnl_usd = gross_pnl_usd - total_fee_usd
    base.update({
        "gross_pnl_usd": gross_pnl_usd,
        "net_pnl_usd": net_pnl_usd,
        "gross_r": gross_pnl_usd / initial_risk_usd,
        "net_r": net_pnl_usd / initial_risk_usd,
        "fee_accounting_status": "REALIZED_TO_DATE",
    })
    return base


def _resolve_shadow_fee_inputs(
    *,
    fee_bps_per_side: float | None,
    maker_fee_bps: float | None,
    taker_fee_bps: float | None,
    entry_execution_type: str,
    exit_execution_type: str,
) -> tuple[float, float, str, str]:
    """Resolve new side-specific rates while preserving the old test/API argument."""
    if fee_bps_per_side is not None:
        # Backwards-compatible legacy mode: the single rate applies to each side.
        entry_bps = exit_bps = float(fee_bps_per_side)
    else:
        maker = 2.0 if maker_fee_bps is None else float(maker_fee_bps)
        taker = 5.0 if taker_fee_bps is None else float(taker_fee_bps)
        entry_bps = _fee_bps_for_execution(
            entry_execution_type, maker_fee_bps=maker, taker_fee_bps=taker,
        )
        exit_bps = _fee_bps_for_execution(
            exit_execution_type, maker_fee_bps=maker, taker_fee_bps=taker,
        )
    return entry_bps, exit_bps, entry_execution_type.upper(), exit_execution_type.upper()


def _estimated_shadow_fee_r(
    entry: float,
    exit_legs: list[tuple[float, float]],
    risk_abs: float,
    fee_bps_per_side: float | None,
) -> float | None:
    """Legacy unit-quantity fee estimate retained for external compatibility."""
    if fee_bps_per_side is None:
        return None
    try:
        fee_bps = float(fee_bps_per_side)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(fee_bps) or fee_bps < 0 or risk_abs <= 0:
        return None
    exit_notional = sum(float(fraction) * float(price) for fraction, price in exit_legs)
    return (fee_bps / 10_000.0) * (entry + exit_notional) / risk_abs


def _record_shadow_exit_outcomes(
    store: Store,
    *,
    signal_id: str,
    active: dict[str, Any],
    candle: Candle,
    entry: float,
    invalidation: float,
    direction: str,
    risk_abs: float,
    fee_bps_per_side: float | None = None,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    entry_execution_type: str = "TAKER",
    exit_execution_type: str = "TAKER",
    notional_usd: float = 1_000.0,
    fee_assumption_source: str = "CURRENT_CONFIG_FOR_LEGACY_SIGNAL",
) -> int:
    """Append conservative 1H-only shadow exits for the three existing models.

    Same-candle TP/SL conflicts remain AMBIGUOUS. All execution types and the
    fixed notional are assumptions; funding and slippage remain separate.
    """
    entry_bps, exit_bps, entry_type, exit_type = _resolve_shadow_fee_inputs(
        fee_bps_per_side=fee_bps_per_side,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        entry_execution_type=entry_execution_type,
        exit_execution_type=exit_execution_type,
    )
    is_long = direction == "LONG"
    stop_hit = candle.low <= invalidation if is_long else candle.high >= invalidation
    event_at = datetime.fromtimestamp(candle.close_time_ms / 1000.0, tz=timezone.utc).isoformat()
    common = {
        "signal_id": signal_id,
        "outcome_event": "SHADOW_EXIT_PROGRESS",
        "evaluation_horizon": "LIVE",
        "as_of_1h_open_time_ms": candle.open_time_ms,
        "event_at": event_at,
        "symbol": str(active.get("pair") or "").upper(),
        "scenario": active.get("scenario_code"),
        "contract_type": str(active.get("contract_type") or "").upper(),
        "direction": direction,
        "entry_reference": entry,
        "invalidation_level": invalidation,
        "risk_abs": risk_abs,
        "risk_pct": risk_abs / entry * 100.0,
        "candle_high": candle.high,
        "candle_low": candle.low,
        "data_quality": {
            "path_resolution": "1H_OHLC",
            "intrabar_order": "UNKNOWN_WHEN_LEVELS_SHARE_BAR",
            "slippage_included": False,
            "funding_included": False,
            "fee_model": "SIDE_SPECIFIC_MAKER_TAKER",
            "execution_type_status": "ASSUMED",
            "notional_status": "ASSUMED_FIXED_NOTIONAL",
            "fee_assumption_source": fee_assumption_source,
        },
        "fee_assumption_source": fee_assumption_source,
        "strategy_version": active.get("strategy_version"),
    }

    def accounting(exit_legs: list[tuple[float, float]] | None) -> dict[str, Any]:
        return _shadow_fee_accounting(
            entry=entry, exit_legs=exit_legs, risk_abs=risk_abs,
            direction=direction, notional_usd=notional_usd,
            entry_fee_bps=entry_bps, exit_fee_bps=exit_bps,
            entry_execution_type=entry_type, exit_execution_type=exit_type,
        )

    written = 0
    for model, target_r in (
        ("FIXED_TP_1_5R_STOP_1R_V1", 1.5),
        ("FIXED_TP_2R_STOP_1R_V1", 2.0),
    ):
        previous = store.latest_signal_outcome(
            signal_id, outcome_event="SHADOW_EXIT_PROGRESS", exit_model_version=model,
        )
        if previous and bool(previous.get("terminal")):
            continue
        target = entry + target_r * risk_abs if is_long else entry - target_r * risk_abs
        target_hit = candle.high >= target if is_long else candle.low <= target
        if target_hit and stop_hit:
            status, terminal, legacy_gross_r, exit_legs = "AMBIGUOUS_SAME_1H_BAR", True, None, None
        elif target_hit:
            status, terminal, legacy_gross_r, exit_legs = "TARGET_HIT", True, target_r, [(1.0, target)]
        elif stop_hit:
            status, terminal, legacy_gross_r, exit_legs = "STOP_HIT", True, -1.0, [(1.0, invalidation)]
        else:
            status, terminal, legacy_gross_r, exit_legs = "OPEN", False, None, []
        costs = accounting(exit_legs)
        fee_r = (
            costs["total_fee_usd"] / costs["initial_risk_usd"]
            if costs.get("total_fee_usd") is not None else None
        )
        record = {
            **common,
            "exit_model_version": model,
            "target_r": target_r,
            "target_price": target,
            "stop_price": invalidation,
            "exit_status": status,
            "terminal": terminal,
            # Kept as a final-outcome-only legacy field.
            "gross_realized_r": legacy_gross_r,
            "estimated_fee_r": fee_r,
            "fee_adjusted_r": legacy_gross_r - fee_r if legacy_gross_r is not None and fee_r is not None else None,
            "same_bar_target_stop_conflict": bool(target_hit and stop_hit),
            **costs,
        }
        written += int(store.record_signal_outcome(record))

    partial_model = "PARTIAL_50_TP_1_5R_BE_V1"
    previous_partial = store.latest_signal_outcome(
        signal_id, outcome_event="SHADOW_EXIT_PROGRESS", exit_model_version=partial_model,
    )
    if not (previous_partial and bool(previous_partial.get("terminal"))):
        target_r = 1.5
        target = entry + target_r * risk_abs if is_long else entry - target_r * risk_abs
        target_hit = candle.high >= target if is_long else candle.low <= target
        be_hit = candle.low <= entry if is_long else candle.high >= entry
        prior_status = str((previous_partial or {}).get("exit_status") or "OPEN")
        if prior_status == "PARTIAL_TP1_5R_BE_ACTIVE":
            if be_hit:
                status, terminal, legacy_gross_r = "BREAKEVEN_STOP_HIT", True, 0.75
                exit_legs = [(0.5, target), (0.5, entry)]
            else:
                status, terminal, legacy_gross_r = "PARTIAL_TP1_5R_BE_ACTIVE", False, 0.75
                exit_legs = [(0.5, target)]
        elif stop_hit:
            if target_hit:
                status, terminal, legacy_gross_r, exit_legs = "AMBIGUOUS_INITIAL_STOP_AND_TARGET", True, None, None
            else:
                status, terminal, legacy_gross_r, exit_legs = "STOP_HIT", True, -1.0, [(1.0, invalidation)]
        elif target_hit and be_hit:
            status, terminal, legacy_gross_r, exit_legs = "AMBIGUOUS_TP_AND_BE_SAME_1H_BAR", True, None, None
        elif target_hit:
            status, terminal, legacy_gross_r = "PARTIAL_TP1_5R_BE_ACTIVE", False, 0.75
            exit_legs = [(0.5, target)]
        else:
            status, terminal, legacy_gross_r, exit_legs = "OPEN", False, None, []

        costs = accounting(exit_legs)
        fee_r = (
            costs["total_fee_usd"] / costs["initial_risk_usd"]
            if costs.get("total_fee_usd") is not None else None
        )
        record = {
            **common,
            "exit_model_version": partial_model,
            "target_r": target_r,
            "target_price": target,
            "initial_stop_price": invalidation,
            "break_even_stop_price": entry,
            "partial_fraction": 0.5,
            "exit_status": status,
            "terminal": terminal,
            "gross_realized_r": legacy_gross_r,
            "estimated_fee_r": fee_r,
            "fee_adjusted_r": legacy_gross_r - fee_r if legacy_gross_r is not None and fee_r is not None else None,
            "same_bar_target_be_conflict": bool(target_hit and be_hit),
            **costs,
        }
        written += int(store.record_signal_outcome(record))

    return written


def _record_signal_progress(
    store: Store,
    active_states: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[Candle]],
    *,
    fee_bps_per_side: float | None = None,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    entry_execution_type: str = "TAKER",
    exit_execution_type: str = "TAKER",
    notional_usd: float = 1_000.0,
) -> int:
    """Append cumulative signal and shadow-exit observations for each new closed 1H bar."""
    written = 0
    seen_signals: set[str] = set()
    for active in active_states:
        event_key = str(active.get("event_key") or "")
        contract_type = str(active.get("contract_type") or "").upper()
        if not event_key or not contract_type:
            continue
        signal_id = hashlib.sha256(f"{contract_type}|{event_key}".encode("utf-8")).hexdigest()
        if signal_id in seen_signals:
            continue
        seen_signals.add(signal_id)
        symbol = str(active.get("pair") or "").upper()
        candles = sorted(candles_by_symbol.get(symbol, []), key=lambda item: item.open_time_ms)
        if not candles:
            continue
        try:
            entry = float(active["entry_reference"])
            level = float(active["invalidation_level"])
            entry_close_ms = int(active["entry_reference_time_ms"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        direction = str(active.get("direction") or "").upper()
        if direction not in {"LONG", "SHORT"} or not all(math.isfinite(x) and x > 0 for x in (entry, level)):
            continue
        risk_pct = (entry - level) / entry * 100.0 if direction == "LONG" else (level - entry) / entry * 100.0
        risk_abs = abs(entry - level)
        if not math.isfinite(risk_pct) or risk_pct <= 0 or not math.isfinite(risk_abs) or risk_abs <= 0:
            continue

        previous = store.latest_signal_outcome(signal_id, outcome_event="SIGNAL_PROGRESS")
        previous_as_of = int(previous.get("as_of_1h_open_time_ms") or 0) if previous else 0
        previous_mfe = float(previous.get("mfe_pct", 0.0)) if previous else 0.0
        previous_mae = float(previous.get("mae_pct", 0.0)) if previous else 0.0
        first_touch = (previous or {}).get("first_touch_invalidation_at_ms")
        first_breach = (previous or {}).get("first_1h_close_breach_at_ms")

        new_candles = [
            candle for candle in candles
            if candle.open_time_ms > entry_close_ms and candle.open_time_ms > previous_as_of
        ]
        for candle in new_candles:
            if direction == "LONG":
                current_mfe_pct = (candle.high / entry - 1.0) * 100.0
                current_mae_pct = (candle.low / entry - 1.0) * 100.0
                touched = candle.low <= level
                breached = candle.close < level
            else:
                current_mfe_pct = (entry / candle.low - 1.0) * 100.0 if candle.low > 0 else 0.0
                current_mae_pct = (entry / candle.high - 1.0) * 100.0 if candle.high > 0 else 0.0
                touched = candle.high >= level
                breached = candle.close > level
            mfe_pct = max(previous_mfe, current_mfe_pct)
            mae_pct = min(previous_mae, current_mae_pct)
            if first_touch is None and touched:
                first_touch = candle.open_time_ms
            if first_breach is None and breached:
                first_breach = candle.close_time_ms
            risk_fraction = risk_pct / 100.0
            mfe_r = (mfe_pct / 100.0) / risk_fraction if risk_fraction > 0 else None
            mae_r = (mae_pct / 100.0) / risk_fraction if risk_fraction > 0 else None
            event_at = datetime.fromtimestamp(candle.close_time_ms / 1000.0, tz=timezone.utc).isoformat()
            record = {
                "signal_id": signal_id,
                "outcome_event": "SIGNAL_PROGRESS",
                "evaluation_horizon": "LIVE",
                "exit_model_version": "STRUCTURAL_1H_V1",
                "as_of_1h_open_time_ms": candle.open_time_ms,
                "event_at": event_at,
                "symbol": symbol,
                "scenario": active.get("scenario_code"),
                "contract_type": contract_type,
                "direction": direction,
                "entry_reference": entry,
                "entry_reference_time_ms": entry_close_ms,
                "invalidation_level": level,
                "risk_pct": risk_pct,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "first_touch_invalidation_at_ms": first_touch,
                "first_touch_invalidation_bar_open_time_ms": first_touch,
                "first_touch_invalidation_time_precision": "WITHIN_1H_BAR_UNKNOWN",
                "first_1h_close_breach_at_ms": first_breach,
                "data_quality": {
                    "path_resolution": "1H_OHLC",
                    "intrabar_order": "UNKNOWN",
                    "first_touch_time_is_bar_open_not_exact_execution_time": True,
                },
                "strategy_version": active.get("strategy_version"),
            }
            if store.record_signal_outcome(record):
                written += 1
            previous_mfe, previous_mae = mfe_pct, mae_pct
            if fee_bps_per_side is not None:
                active_fee_source = "LEGACY_PER_SIDE_CALL_OVERRIDE"
                active_maker_fee = maker_fee_bps
                active_taker_fee = taker_fee_bps
                active_entry_type = entry_execution_type
                active_exit_type = exit_execution_type
                active_notional = notional_usd
            else:
                # SIGNAL_OPENED is immutable under outcome idempotency, unlike a
                # notification row which may be refreshed on a retry/reminder.
                # Always read the first durable fee snapshot from that marker.
                opened_record = store.latest_signal_outcome(
                    signal_id, outcome_event="SIGNAL_OPENED",
                ) or {}
                snapshot_fields = (
                    "shadow_maker_fee_bps", "shadow_taker_fee_bps",
                    "shadow_entry_execution_type", "shadow_exit_execution_type", "shadow_notional_usd",
                )
                has_snapshot = all(opened_record.get(field) is not None for field in snapshot_fields)
                if has_snapshot:
                    active_fee_source = "SNAPSHOTTED_AT_SIGNAL_OPEN"
                    active_maker_fee = float(opened_record["shadow_maker_fee_bps"])
                    active_taker_fee = float(opened_record["shadow_taker_fee_bps"])
                    active_entry_type = str(opened_record["shadow_entry_execution_type"])
                    active_exit_type = str(opened_record["shadow_exit_execution_type"])
                    active_notional = float(opened_record["shadow_notional_usd"])
                else:
                    active_fee_source = "CURRENT_CONFIG_FOR_LEGACY_SIGNAL"
                    active_maker_fee = maker_fee_bps
                    active_taker_fee = taker_fee_bps
                    active_entry_type = entry_execution_type
                    active_exit_type = exit_execution_type
                    active_notional = notional_usd
            _record_shadow_exit_outcomes(
                store, signal_id=signal_id, active=active, candle=candle,
                entry=entry, invalidation=level, direction=direction,
                risk_abs=risk_abs, fee_bps_per_side=fee_bps_per_side,
                maker_fee_bps=active_maker_fee, taker_fee_bps=active_taker_fee,
                entry_execution_type=active_entry_type,
                exit_execution_type=active_exit_type, notional_usd=active_notional,
                fee_assumption_source=active_fee_source,
            )
    return written

def _preliminary_candidate_symbols(
    enriched: list[EnrichedRow], stats: Any, settings: Settings, store: Store,
    *, contract_type: str | None = None, as_of: datetime | None = None,
    current_observations: dict[str, dict[str, Any]] | None = None,
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
    # Only fetch charts when every non-chart P1 condition has passed and a
    # specific 1H confirmation can still change the outcome. Preparation and
    # wait-for-bounce states are context, not entry candidates.
    chart_wait_statuses = {
        "WAITING_FOR_1H_BREAKOUT_RETEST",
        "WAITING_FOR_HIGHER_LOW",
        "WAITING_FOR_HISTORY_OR_HIGHER_LOW",
        "WAITING_FOR_1H_BREAKDOWN_RETEST",
        "WAITING_FOR_1H_STRUCTURE_BREAK",
        "WAITING_FOR_1H_FAILED_BREAKOUT",
    }
    candidate_priority: dict[str, tuple[float, float]] = {}
    for row in enriched:
        history = store.recent_symbol_history(row.symbol, limit=100, contract_type=contract_type)
        if contract_type and not store.symbol_history_is_current(row.symbol, contract_type=contract_type):
            history = []
        spot_sequence = store.spot_sequence_status(
            row.symbol,
            required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
            interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
            tolerance_minutes=int(settings.p1.get("scenario", {}).get("spot_accumulation_snapshot_tolerance_minutes", 30)),
            as_of=as_of,
            contract_type=contract_type,
            current_observation=(current_observations or {}).get(row.symbol),
        )
        draft = evaluate_scenarios(
            row, stats, settings.p1,
            technical={"spot_sequence_ok": spot_sequence},
            history=history,
            as_of=as_of,
        )
        qualifying = [
            result for result in draft
            if not result.quality_flags
            and result.status in chart_wait_statuses
            and (
                result.status != "WAITING_FOR_HISTORY_OR_HIGHER_LOW"
                or result.technical.get("spot_sequence_ok") is True
            )
        ]
        if qualifying:
            symbols.add(row.symbol)
            # All admitted states have already passed their non-chart checks.
            # Prefer directly observable 24H turnover, then OI; do not rank by
            # check-count because every admitted state has only chart work left.
            liquidity_rank = (
                float(row.market.volume24_usd or 0.0),
                float(row.market.oi_usd or 0.0),
            )
            candidate_priority[row.symbol] = max(
                candidate_priority.get(row.symbol, (0.0, 0.0)),
                liquidity_rank,
            )
    max_candidates = int(settings.p1.get("scanner", {}).get("max_chart_candidates", 40))
    if len(symbols) > max_candidates:
        # Active positions/events must stay monitored; only draft candidates are
        # subject to the cap.
        active_present = symbols & active_symbols
        remaining = symbols - active_present
        room = max(0, max_candidates - len(active_present))
        selected = set(sorted(
            remaining,
            key=lambda symbol: candidate_priority.get(symbol, (0.0, 0.0)),
            reverse=True,
        )[:room])
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
    # Missing metrics are evaluated per scenario as UNKNOWN by its own checks.
    # A global Coinalyze column gate would block unrelated scenarios.
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
        "LONG B": ("higher_low", "confirmation_close_time_ms"),
        "LONG D": ("higher_low", "confirmation_close_time_ms"),
        "SHORT A": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT B": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT C": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT D": ("weak_bounce_rejection", "rejection_open_time_ms"),
        "SHORT E": ("failed_breakout_rejection", "failure_open_time_ms"),
    }.get(code)
    if fields is None:
        return None
    value = (technical.get(fields[0]) or {}).get(fields[1])
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _reversal_new_state_needs_recovery(
    latest_state: dict[str, Any] | None,
    new_event_key: str,
    marker_sent_at: str,
) -> bool:
    """Recover a reversal's new ACTIVE row only when its group state predates the marker.

    A missing row means the process may have crashed between persisting the
    Telegram reversal marker and the per-scenario lifecycle state. An older
    row in the same group belongs to an earlier setup and must not mask that
    partial write. A matching event key or a state at/after the marker is a
    later lifecycle and must never be resurrected.
    """
    if latest_state is None:
        return True
    if str(latest_state.get("event_key") or "") == str(new_event_key):
        return False
    try:
        previous_at = datetime.fromisoformat(str(latest_state["sent_at"]))
        marker_at = datetime.fromisoformat(str(marker_sent_at))
    except (KeyError, TypeError, ValueError):
        return False
    if previous_at.tzinfo is None:
        previous_at = previous_at.replace(tzinfo=timezone.utc)
    if marker_at.tzinfo is None:
        marker_at = marker_at.replace(tzinfo=timezone.utc)
    return previous_at < marker_at


def _confirmation_time(result: ScenarioResult) -> int | None:
    """Return when the scenario became confirmable, not merely its pivot identity."""
    technical = result.technical or {}
    if _scenario_code(result.scenario) in {"LONG B", "LONG D"}:
        value = (technical.get("higher_low") or {}).get("confirmation_close_time_ms")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return _confirmation_anchor(result)


def _latest_opposite_candidate(candidates: list[tuple[ScenarioResult, EnrichedRow, tuple[str, str, str, float]]]):
    """Choose the candidate with the newest confirmed trigger candle.

    Candidate identity contains an event-key string, not a sortable timestamp;
    lexicographical event-key ordering must never decide a reversal. For a
    higher-low setup, both ordering and notification identity use the first
    closed bar that made the pivot confirmable. The pivot candle time remains
    available separately for structural diagnostics.
    """
    if not candidates:
        raise ValueError("Cannot choose a reversal from an empty candidate list")
    return max(candidates, key=lambda item: _confirmation_time(item[0]) or 0)


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
    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    level = invalidation.get("level")
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not math.isfinite(float(level)) or level <= 0:
        return False
    breached = close < level if result.direction.upper() == "LONG" else close > level
    if breached:
        return False

    # 4H context is descriptive only; requiring a 4H pivot here creates an
    # undocumented delivery veto on scenarios already confirmed on closed 1H data.
    return True


def _signal_entry_metadata(result: ScenarioResult, settings: Settings | None = None) -> dict[str, Any]:
    """Persist the signal's entry reference and frozen shadow fee assumptions."""
    last = (result.technical or {}).get("last_closed_1h") or {}
    close = last.get("close")
    close_time = last.get("close_time_ms")
    open_time = last.get("open_time_ms")
    metadata: dict[str, Any] = {
        "entry_reference": float(close) if isinstance(close, (int, float)) and math.isfinite(float(close)) and close > 0 else None,
        "entry_reference_time_ms": int(close_time) if isinstance(close_time, int) else None,
        "entry_reference_1h_open_time_ms": int(open_time) if isinstance(open_time, int) else None,
        "entry_reference_type": "BINANCE_FUTURES_LAST_CLOSED_1H_CLOSE",
        "strategy_version": __version__,
    }
    if settings is not None:
        metadata.update({
            "shadow_maker_fee_bps": float(getattr(settings, "maker_fee_bps", 2.0)),
            "shadow_taker_fee_bps": float(getattr(settings, "taker_fee_bps", 5.0)),
            "shadow_entry_execution_type": str(getattr(settings, "shadow_entry_execution_type", "TAKER")).upper(),
            "shadow_exit_execution_type": str(getattr(settings, "shadow_exit_execution_type", "TAKER")).upper(),
            "shadow_notional_usd": float(getattr(settings, "shadow_notional_usd", 1000.0)),
        })
    return metadata


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


def _record_permanent_delivery_failure(
    store: Store, event_key: str, recipient_key: str, contract_type: str, delivery: dict[str, Any],
) -> None:
    """Persist a definite Telegram rejection without altering an active lifecycle group."""
    store.record_notification(
        event_key, None, recipient_key=recipient_key,
        event_state="FAILED", contract_type=contract_type,
        delivery_status="FAILED", error=str(delivery.get("error") or "Permanent Telegram rejection"),
    )


def _has_retryable_notification_failures(outcomes: list[dict[str, Any]]) -> bool:
    """Keep the poll uncommitted when a notification can safely be retried.

    SENT, UNKNOWN and permanent FAILED outcomes are terminal for idempotency.
    A definite transient failure (for example exhausted Telegram 429 retries)
    is not recorded in notification history and must be retried on a later scan.
    """
    for outcome in outcomes:
        if outcome.get("processing_error"):
            return True
        delivery = outcome.get("delivery")
        if not isinstance(delivery, dict):
            continue
        if not delivery.get("sent") and not delivery.get("delivery_unknown") and not delivery.get("permanent_failure"):
            return True
    return False


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
    normalized_active = active_states

    current_contract = str(contract_type).upper()
    contract_active_states = [
        item for item in normalized_active
        if str(item["contract_type"]).upper() == current_contract
    ]

    # A confirmed opposite setup is a reversal. Aggregate all previously active
    # same-side scenarios for the pair into one standalone reversal notification.
    reversed_groups: set[tuple[str, str]] = set()
    pending_reversal_groups: set[tuple[str, str]] = set()
    outcomes: list[dict[str, Any]] = []
    active_by_pair_recipient: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for active in contract_active_states:
        group = str(active["event_group"])
        symbol, _, _ = _event_group_parts(group)
        recipient_key = str(active["recipient_key"])
        if symbol and recipient_key:
            active_by_pair_recipient.setdefault((symbol, recipient_key), []).append(active)

    for (symbol, recipient_key), pair_actives in active_by_pair_recipient.items():
        opposite_new = []
        for candidate in confirmed_opposite.get(symbol, []):
            new_direction = candidate[0].direction.upper()
            old_side_actives = [
                active for active in pair_actives
                if str(active["direction"]).upper() != new_direction
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
        new_result, new_row, new_identity = _latest_opposite_candidate(opposite_new)
        new_group, new_event_key, new_fingerprint, new_level = new_identity
        new_direction = new_result.direction.upper()
        old_actives = [
            item for item in pair_actives
            if str(item["direction"]).upper() != new_direction
        ]
        old_codes = sorted({_scenario_code(_event_group_parts(str(item.get("event_group") or ""))[1]) for item in old_actives})
        old_direction = old_actives[0]["direction"]
        reversal_key = f"REVERSAL|{symbol}|{old_direction}|{new_direction}|{new_event_key}"
        reversal_failure_key = f"FAILED|{reversal_key}"
        if store.notification_exists(
            reversal_failure_key, recipient_key=recipient_key, contract_type=contract_type
        ):
            continue
        if store.notification_exists(reversal_key, recipient_key=recipient_key, contract_type=contract_type):
            marker_group = f"REVERSAL|{symbol}|{old_direction}|{new_direction}"
            marker = store.latest_notification_state(
                marker_group, recipient_key=recipient_key, contract_type=contract_type
            ) or {}
            related_event_keys = set(marker.get("related_event_keys") or [])
            marker_sent_at = str(marker.get("sent_at") or "")
            for old in old_actives:
                if related_event_keys:
                    if str(old.get("event_key") or "") not in related_event_keys:
                        continue
                elif marker_sent_at and str(old.get("sent_at") or "") > marker_sent_at:
                    continue
                old_group = str(old.get("event_group") or "")
                latest_old = store.latest_notification_state(
                    old_group, recipient_key=recipient_key, contract_type=contract_type
                ) or old
                if str(latest_old["event_state"]).upper() == "ACTIVE":
                    store.record_notification(
                        f"{reversal_key}|RECOVER_OLD_STATE|{old_group}",
                        latest_old.get("telegram_message_id"),
                        recipient_key=recipient_key, event_group=old_group,
                        fingerprint=str(latest_old["fingerprint"]),
                        event_state="REVERSED",
                        direction=str(latest_old["direction"]),
                        invalidation_level=float(latest_old["invalidation_level"]),
                        pair=str(latest_old.get("pair") or symbol),
                        scenario_code=_scenario_code(_event_group_parts(old_group)[1]),
                        contract_type=contract_type,
                        delivery_status=str(latest_old["delivery_status"]),
                        related_event_keys=[str(latest_old.get("event_key") or "")],
                        outcome_as_of_1h_open_time_ms=(
                            marker.get("entry_reference_1h_open_time_ms")
                            if isinstance(marker.get("entry_reference_1h_open_time_ms"), int) else None
                        ),
                    )
                reversed_groups.add((old_group, recipient_key))
            latest_new = store.latest_notification_state(
                new_group, recipient_key=recipient_key, contract_type=contract_type
            )
            # A missing row or an older lifecycle from a previous setup is a
            # partial-write recovery case. Never resurrect a state at/after marker.
            if _reversal_new_state_needs_recovery(latest_new, new_event_key, marker_sent_at):
                marker_status = str((marker or {}).get("delivery_status") or "UNKNOWN").upper()
                if marker_status not in {"SENT", "UNKNOWN"}:
                    marker_status = "UNKNOWN"
                store.record_notification(
                    new_event_key, (marker or {}).get("telegram_message_id"),
                    recipient_key=recipient_key, event_group=new_group,
                    fingerprint=new_fingerprint, event_state="ACTIVE", direction=new_direction,
                    invalidation_level=new_level,
                    pair=str((new_result.technical or {}).get("binance_symbol") or symbol),
                    scenario_code=_scenario_code(new_result.scenario), contract_type=contract_type,
                    delivery_status=marker_status,
                    **_signal_entry_metadata(new_result, settings),
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
        delivery = _send_one(telegram, message, recipient_to_chat[recipient_key])
        if delivery.get("permanent_failure"):
            _record_permanent_delivery_failure(
                store, reversal_failure_key, recipient_key, contract_type, delivery
            )
            outcomes.append({"type": "reversal", "symbol": symbol, "sent": False, "delivery": delivery})
            continue
        if delivery.get("sent") or delivery.get("delivery_unknown"):
            delivery_status = "SENT" if delivery.get("sent") else "UNKNOWN"
            store.record_notification(
                reversal_key, delivery.get("message_id"), recipient_key=recipient_key,
                event_group=f"REVERSAL|{symbol}|{old_direction}|{new_direction}",
                fingerprint=new_fingerprint, event_state="SENT", direction=new_direction,
                invalidation_level=new_level, pair=pair,
                scenario_code=_scenario_code(new_result.scenario),
                related_event_keys=[str(old["event_key"]) for old in old_actives],
                contract_type=contract_type, delivery_status=delivery_status,
                error=str(delivery.get("error") or "") or None,
                **_signal_entry_metadata(new_result, settings),
            )
            reversal_close_open_ms = (new_result.technical or {}).get("last_closed_1h", {}).get("open_time_ms")
            for old in old_actives:
                old_group = str(old.get("event_group") or "")
                store.record_notification(
                    f"{reversal_key}|OLD_STATE|{old_group}", delivery.get("message_id"), recipient_key=recipient_key,
                    event_group=old_group, fingerprint=str(old["fingerprint"]),
                    event_state="REVERSED", direction=str(old_direction),
                    invalidation_level=float(old["invalidation_level"]),
                    pair=str(old.get("pair") or pair), scenario_code=_event_group_parts(old_group)[1].split(" — ", 1)[0],
                    contract_type=contract_type, delivery_status=delivery_status,
                    related_event_keys=[str(old.get("event_key") or "")],
                    outcome_as_of_1h_open_time_ms=(
                        int(reversal_close_open_ms) if isinstance(reversal_close_open_ms, int) else None
                    ),
                )
                reversed_groups.add((old_group, recipient_key))
            store.record_notification(
                new_event_key, delivery.get("message_id"), recipient_key=recipient_key,
                event_group=new_group, fingerprint=new_fingerprint, event_state="ACTIVE",
                direction=new_direction, invalidation_level=new_level, pair=pair,
                scenario_code=_scenario_code(new_result.scenario), contract_type=contract_type,
                delivery_status=delivery_status,
                **_signal_entry_metadata(new_result, settings),
            )
        elif not delivery.get("permanent_failure"):
            # Do not send a separate cancellation for the same old setup in this
            # run after its combined reversal message hit a retryable failure.
            pending_reversal_groups.update(
                (str(old.get("event_group") or ""), recipient_key) for old in old_actives
            )
        outcomes.append({"type": "reversal", "symbol": symbol, "sent": bool(delivery.get("sent")), "delivery": delivery})

    # Reconcile durable reversal markers independently of the current candidate
    # list. A process may die after Telegram accepted the reversal and after all
    # old states were marked REVERSED, but before the new ACTIVE record was saved.
    # The marker stores enough identity/level data to finish that transaction on
    # a later run even if the 1H trigger is no longer fresh.
    for marker in store.reversal_markers(recipient_keys=recipients):
        parts = str(marker.get("event_key") or "").split("|", 4)
        if len(parts) != 5 or parts[0] != "REVERSAL":
            continue
        _, symbol, old_direction, new_direction, new_event_key = parts
        if old_direction not in {"LONG", "SHORT"} or new_direction not in {"LONG", "SHORT"}:
            continue
        if old_direction == new_direction or not new_event_key:
            continue
        recipient_key = str(marker.get("recipient_key") or "")
        marker_contract = str(marker.get("contract_type") or "").upper()
        new_group = new_event_key.rsplit("|", 1)[0]
        marker_status = str(marker.get("delivery_status") or "UNKNOWN").upper()
        if marker_status not in {"SENT", "UNKNOWN"}:
            marker_status = "UNKNOWN"
        marker_level = marker.get("invalidation_level")
        try:
            marker_level = float(marker_level)
        except (TypeError, ValueError):
            marker_level = None
        if marker_level is not None and (not math.isfinite(marker_level) or marker_level <= 0):
            marker_level = None
        # Markers written by pre-1.0.22 builds did not persist the new level.
        # If the same trigger is still present, its current identity is a safe
        # fallback for completing that legacy partial transition.
        if marker_level is None:
            candidate = next((
                item for item in confirmed_opposite.get(symbol, [])
                if item[2][1] == new_event_key
            ), None)
            if candidate is not None:
                marker_level = candidate[2][3]

        latest_new = store.latest_notification_state(
            new_group, recipient_key=recipient_key, contract_type=marker_contract
        )
        marker_sent_at = str(marker.get("sent_at") or "")
        related_event_keys = set(marker.get("related_event_keys") or [])
        newer_opposite_active = any(
            item.get("recipient_key") == recipient_key
            and str(item.get("contract_type") or "").upper() == marker_contract
            and str(item.get("direction") or "").upper() == old_direction
            and str(item.get("event_group") or "").startswith(f"{symbol}|")
            and str(item.get("event_key") or "") not in related_event_keys
            and str(item.get("sent_at") or "") > marker_sent_at
            for item in normalized_active
        )
        if newer_opposite_active:
            # A later setup on the old side supersedes this marker. Do not
            # resurrect an older opposite event alongside that newer ACTIVE.
            continue
        can_finish_new_state = latest_new is not None or marker_level is not None
        if can_finish_new_state and related_event_keys:
            for old_active in normalized_active:
                old_group = str(old_active.get("event_group") or "")
                if (
                    old_active.get("recipient_key") != recipient_key
                    or str(old_active.get("contract_type") or "").upper() != marker_contract
                    or str(old_active.get("direction") or "").upper() != old_direction
                    or (
                        str(old_active.get("event_key") or "") not in related_event_keys
                        if related_event_keys
                        else not marker_sent_at or str(old_active.get("sent_at") or "") > marker_sent_at
                    )
                    or not old_group.startswith(f"{symbol}|")
                ):
                    continue
                latest_old = store.latest_notification_state(
                    old_group, recipient_key=recipient_key, contract_type=marker_contract
                )
                reversed_groups.add((old_group, recipient_key))
                if latest_old is None or str(latest_old.get("event_state") or "").upper() != "ACTIVE":
                    continue
                old_level = latest_old.get("invalidation_level")
                if not isinstance(old_level, (int, float)) or isinstance(old_level, bool):
                    continue
                store.record_notification(
                    f"OLD_STATE|{latest_old['event_key']}|REVERSED|{new_event_key}",
                    marker.get("telegram_message_id"), recipient_key=recipient_key,
                    event_group=old_group, fingerprint=str(latest_old["fingerprint"]),
                    event_state="REVERSED", direction=old_direction,
                    invalidation_level=float(old_level),
                    pair=str(latest_old.get("pair") or symbol),
                    scenario_code=str(latest_old.get("scenario_code") or "UNKNOWN"),
                    contract_type=marker_contract, delivery_status=marker_status,
                    related_event_keys=[str(latest_old.get("event_key") or "")],
                    outcome_as_of_1h_open_time_ms=(
                        marker.get("entry_reference_1h_open_time_ms")
                        if isinstance(marker.get("entry_reference_1h_open_time_ms"), int) else None
                    ),
                )

        if (
            _reversal_new_state_needs_recovery(latest_new, new_event_key, marker_sent_at)
            and marker_level is not None
        ):
            store.record_notification(
                new_event_key, marker.get("telegram_message_id"),
                recipient_key=recipient_key, event_group=new_group,
                fingerprint=str(marker.get("fingerprint") or ""),
                event_state="ACTIVE", direction=new_direction,
                invalidation_level=marker_level,
                pair=str(marker.get("pair") or symbol),
                scenario_code=str(marker.get("scenario_code") or "UNKNOWN"),
                contract_type=marker_contract, delivery_status=marker_status,
                entry_reference=marker.get("entry_reference"),
                entry_reference_time_ms=marker.get("entry_reference_time_ms"),
                entry_reference_1h_open_time_ms=marker.get("entry_reference_1h_open_time_ms"),
                entry_reference_type=marker.get("entry_reference_type"),
                strategy_version=marker.get("strategy_version"),
            )

    # Invalidate remaining ACTIVE scenarios by the exact 1H level saved when
    # their original message was sent. Each cancellation is its own Telegram post.
    for active in normalized_active:
        group = str(active.get("event_group") or "")
        symbol, scenario, direction = _event_group_parts(group)
        recipient_key = str(active["recipient_key"])
        if (group, recipient_key) in reversed_groups or (group, recipient_key) in pending_reversal_groups:
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
        active_contract_type = str(active["contract_type"]).upper()
        cancel_failure_key = f"FAILED|CANCEL|{active.get('event_key')}"
        if store.notification_exists(
            cancel_failure_key, recipient_key=recipient_key, contract_type=active_contract_type
        ) or store.notification_exists(
            cancel_key, recipient_key=recipient_key, contract_type=active_contract_type
        ):
            continue
        pair = str(active.get("pair") or technical.get("binance_symbol") or symbol)
        message = format_cancellation_message(pair, direction, _scenario_code(scenario), level)
        delivery = _send_one(telegram, message, recipient_to_chat[recipient_key])
        if delivery.get("permanent_failure"):
            _record_permanent_delivery_failure(
                store, cancel_failure_key, recipient_key, active_contract_type, delivery
            )
        elif delivery.get("sent") or delivery.get("delivery_unknown"):
            store.record_notification(
                cancel_key, delivery.get("message_id"), recipient_key=recipient_key,
                event_group=group, fingerprint=str(active["fingerprint"]),
                event_state="CANCELLED", direction=direction, invalidation_level=level,
                pair=pair, scenario_code=_scenario_code(scenario), contract_type=active_contract_type,
                delivery_status="SENT" if delivery.get("sent") else "UNKNOWN",
                error=str(delivery.get("error") or "") or None,
                related_event_keys=[str(active.get("event_key") or "")],
                outcome_as_of_1h_open_time_ms=(
                    last.get("open_time_ms") if isinstance(last.get("open_time_ms"), int) else None
                ),
            )
        if not delivery.get("permanent_failure"):
            outcomes.append({"type": "cancellation", "symbol": symbol, "sent": bool(delivery.get("sent")), "delivery": delivery})

    # New confirmations are sent as separate Telegram posts, one signal per
    # message and recipient. An already ACTIVE group is never repeated because
    # its trigger timestamp/metrics changed.
    active_groups_by_recipient = {
        (str(item["event_group"]), str(item["recipient_key"]))
        for item in contract_active_states
    }
    normal_by_recipient: dict[str, list[tuple[ScenarioResult, EnrichedRow, tuple[str, str, str, float]]]] = {
        recipient: [] for recipient in recipients
    }
    for symbol, items in confirmed.items():
        for result, row, identity in items:
            group, event_key, fingerprint, level = identity
            for recipient_key in recipients:
                if store.notification_exists(
                    event_key, recipient_key=recipient_key, contract_type=contract_type
                ):
                    continue
                last_cancelled_close = store.latest_cancelled_close_time_ms(
                    symbol, result.direction, recipient_key=recipient_key, contract_type=contract_type
                )
                confirmation_anchor = _confirmation_anchor(result)
                if (
                    last_cancelled_close is not None
                    and confirmation_anchor is not None
                    and confirmation_anchor <= last_cancelled_close
                ):
                    log.info(
                        "Suppressing %s %s re-entry: trigger confirmation %s is not newer than cancelled 1H close %s",
                        symbol, result.direction, confirmation_anchor, last_cancelled_close,
                    )
                    continue
                if (group, recipient_key) in active_groups_by_recipient or (group, recipient_key) in reversed_groups:
                    continue
                has_opposite_active = any(
                    _event_group_parts(active_group)[0] == symbol
                    and _event_group_parts(active_group)[2] != result.direction.upper()
                    and active_recipient == recipient_key
                    for active_group, active_recipient in active_groups_by_recipient
                )
                if has_opposite_active:
                    continue
                # A reversal message already carried this new scenario for this pair.
                reversal_group = f"REVERSAL|{symbol}|"
                if any(g.startswith(reversal_group) and c == recipient_key for g, c in active_groups_by_recipient):
                    continue
                normal_by_recipient[recipient_key].append((result, row, identity))

    for recipient_key, items in normal_by_recipient.items():
        for result, row, identity in items:
            group, event_key, fingerprint, level = identity
            try:
                message = format_alert(result, row, result.technical, fetched_at=fetched_at)
            except (TypeError, ValueError) as exc:
                log.error("Cannot format confirmed signal %s %s: %s", result.symbol, result.scenario, exc)
                outcomes.append({
                    "type": "confirmed_signal", "sent": False, "symbol": result.symbol,
                    "scenario": result.scenario, "processing_error": True, "error": str(exc),
                })
                continue

            # One call to sendMessage per signal. Do not combine messages,
            # prepend a batch heading, or place separator lines between signals.
            delivery = _send_one(telegram, message, recipient_to_chat[recipient_key])
            if delivery.get("permanent_failure"):
                _record_permanent_delivery_failure(
                    store, event_key, recipient_key, contract_type, delivery
                )
            elif delivery.get("sent") or delivery.get("delivery_unknown"):
                store.record_notification(
                    event_key, delivery.get("message_id"), recipient_key=recipient_key,
                    event_group=group, fingerprint=fingerprint, event_state="ACTIVE",
                    direction=result.direction, invalidation_level=level,
                    pair=str((result.technical or {}).get("binance_symbol") or row.symbol),
                    scenario_code=_scenario_code(result.scenario), contract_type=contract_type,
                    delivery_status="SENT" if delivery.get("sent") else "UNKNOWN",
                    error=str(delivery.get("error") or "") or None,
                    **_signal_entry_metadata(result, settings),
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
    use_dry_run = settings.dry_run if dry_run is None else dry_run
    store = Store(settings.data_dir, create_files=not use_dry_run)
    run_id = None
    if not use_dry_run:
        pruned = store.prune(snapshot_retention_days=30, run_retention_days=90, notification_retention_days=180)
        if any(pruned.values()):
            log.info("Persistent state pruned: %s", pruned)
        run_id = store.start_run()
    try:
        cooldown_remaining = store.binance_cooldown_remaining()
        if cooldown_remaining > 0:
            raise BinanceCooldownActiveError(
                f"Persisted Binance Retry-After cooldown is active for another {cooldown_remaining:.1f}s; "
                "skipping the scan to avoid extending an IP ban.",
                retry_after_seconds=cooldown_remaining,
            )
        scrape = fetch_table(
            url=settings.coinalyze_url,
            p_sid=settings.coinalyze_p_sid,
            chat_sid=settings.coinalyze_chat_sid,
            max_pages=settings.coinalyze_max_pages,
            headless=settings.coinalyze_headless,
            timeout_ms=settings.coinalyze_timeout_ms,
            page_attempts=getattr(settings, "coinalyze_page_attempts", 3),
            retry_backoff_sec=getattr(settings, "coinalyze_retry_backoff_sec", 3.0),
        )
        # Dry runs retain the legacy 4H duplicate-snapshot check but never
        # mutate the new last-successful-live-poll marker.
        unchanged = (
            store.snapshot_unchanged(scrape.contract_type, scrape.rows)
            if use_dry_run
            else store.poll_unchanged(scrape.contract_type, scrape.rows)
        )
        snapshot_cfg = settings.p1.get("scenario", {})
        snapshot_interval_hours = float(snapshot_cfg.get("snapshot_interval_hours", 4.0))
        snapshot_tolerance_minutes = int(snapshot_cfg.get("snapshot_interval_tolerance_minutes", 10))
        snapshot_saved = False
        snapshot_pending = (
            not use_dry_run
            and not unchanged
            and store.snapshot_due(
                scrape.fetched_at,
                interval_hours=snapshot_interval_hours,
                tolerance_minutes=snapshot_tolerance_minutes,
                contract_type=scrape.contract_type,
            )
        )
        # Keep the current due observation separate until all scenario decisions
        # have consumed the preceding committed history. This is required by
        # SHORT E: the live F/S turn-down must not replace the last non-bearish
        # historical row before its historical predicates are checked.
        current_observations: dict[str, dict[str, Any]] = {}
        if snapshot_pending:
            current_observations = {
                row.symbol: {
                    **row.to_dict(),
                    "snapshot_at": scrape.fetched_at.isoformat(),
                    "contract_type": scrape.contract_type,
                }
                for row in scrape.rows
            }
        enriched, stats = analyze_universe(scrape.rows, settings.p1)
        if unchanged:
            stats.warnings.append("No new Coinalyze table data: parsed values fully match the previous saved snapshot")
            log.info("Coinalyze values unchanged vs the previous saved snapshot")
        elif not snapshot_pending:
            stats.warnings.append(
                f"Market values changed, but durable history is sampled every ~{snapshot_interval_hours:g}h; "
                "this poll is not counted as an independent P1 snapshot"
            )
        if scrape.missing_columns:
            stats.warnings.append("Missing recognized optional/required columns: " + ", ".join(scrape.missing_columns))
        if scrape.missing_required_columns:
            stats.warnings.append(
                "Some legacy P1 diagnostic fields are missing; affected scenario checks remain UNKNOWN: "
                + ", ".join(scrape.missing_required_columns)
            )
        if scrape.contract_type == "UNKNOWN":
            stats.warnings.append("Perpetual vs All Contracts could not be verified")
        candidate_symbols = _preliminary_candidate_symbols(
            enriched, stats, settings, store, contract_type=scrape.contract_type,
            as_of=scrape.fetched_at, current_observations=current_observations,
        )
        binance = BinanceClient(
            settings.binance_base_url,
            settings.binance_request_interval_sec,
            http_proxy=settings.binance_http_proxy,
            vpn_enabled=settings.binance_vpn_enabled,
        )
        recipient_keys = [
            key for chat_id in settings.telegram_chat_ids
            if (key := _recipient_storage_key(str(chat_id), settings)) is not None
        ]
        active_states_before_scan = store.active_notification_states(
            recipient_keys=recipient_keys, contract_type=scrape.contract_type
        )
        # Active notification event groups use Coinalyze symbols, while
        # their ``pair`` field stores the resolved Binance USD-M symbol.
        active_coinalyze_symbols = {
            str(item.get("event_group") or "").split("|", 1)[0].upper()
            for item in active_states_before_scan
            if str(item.get("event_group") or "").split("|", 1)[0]
        }
        technical_by_symbol: dict[str, dict[str, Any]] = {}
        candles_1h_by_symbol: dict[str, list[Candle]] = {}
        for index, symbol in enumerate(sorted(candidate_symbols), start=1):
            log.info("Fetching candles for chart candidate %d/%d: %s", index, len(candidate_symbols), symbol)
            if symbol.upper() in active_coinalyze_symbols:
                technical = _technical_for_symbol(
                    binance, symbol, settings, include_outcome_history=True
                )
            else:
                # Keep the established call shape for the common path and
                # existing integrations that wrap this function.
                technical = _technical_for_symbol(binance, symbol, settings)
            outcome_candles = technical.pop("_candles_1h", [])
            candles_1h_by_symbol[symbol.upper()] = outcome_candles
            resolved_pair = str(technical.get("binance_symbol") or "").upper()
            if resolved_pair and symbol.upper() in active_coinalyze_symbols:
                candles_1h_by_symbol[resolved_pair] = outcome_candles
            technical_by_symbol[symbol] = technical
        chart_ready = sum(
            isinstance(item.get("last_closed_1h"), dict)
            and isinstance((context := item.get("context_4h")), dict)
            and isinstance(context.get("ema20"), (int, float))
            and isinstance(context.get("ema50"), (int, float))
            for item in technical_by_symbol.values()
        )
        if chart_ready < len(candidate_symbols):
            stats.warnings.append(
                f"Incomplete Binance chart coverage: {chart_ready}/{len(candidate_symbols)} candidates "
                "have both 1H and 4H context; candidates without a fresh closed 1H candle cannot trigger entry alerts"
            )

        results: list[ScenarioResult] = []
        for row in enriched:
            # Use only durable observations for multi-snapshot patterns. An
            # unchanged Coinalyze poll is deliberately not a new independent sample.
            history = store.recent_symbol_history(row.symbol, limit=100, contract_type=scrape.contract_type)
            if not store.symbol_history_is_current(row.symbol, contract_type=scrape.contract_type):
                history = []
            spot_sequence = store.spot_sequence_status(
                row.symbol,
                required=int(settings.p1.get("scenario", {}).get("spot_accumulation_required_snapshots", 3)),
                interval_hours=float(settings.p1.get("scenario", {}).get("snapshot_interval_hours", 4.0)),
                tolerance_minutes=int(settings.p1.get("scenario", {}).get("spot_accumulation_snapshot_tolerance_minutes", 30)),
                as_of=scrape.fetched_at,
                contract_type=scrape.contract_type,
                current_observation=current_observations.get(row.symbol),
            )
            technical = {"spot_sequence_ok": spot_sequence, **technical_by_symbol.get(row.symbol, {})}
            scenario_results = evaluate_scenarios(
                row, stats, settings.p1, technical=technical, history=history, as_of=scrape.fetched_at
            )
            for scenario_result in scenario_results:
                # Preserve both global diagnostics (order book/context) and scenario-specific predicates.
                scenario_result.technical = {**technical, **scenario_result.technical}
            results.extend(scenario_results)

        if snapshot_pending:
            quality_assessments = {
                item.market.symbol: {
                    "quality_clean": not item.quality_flags,
                    "quality_flags": list(item.quality_flags),
                    "oi_vol_ratio": item.market.oi_vol_ratio,
                    "vol_mktcap_ratio": item.market.vol_mktcap_ratio,
                    "oi_mktcap_ratio": item.market.oi_mktcap_ratio,
                    "diagnostic_notes": list(item.diagnostic_notes),
                }
                for item in enriched
            }
            store.add_snapshot(
                scrape.fetched_at, scrape.contract_type, scrape.rows,
                quality_assessments=quality_assessments,
            )
            snapshot_saved = True

        # Persist only pre-screened/near-confirmation candidates. The evaluation
        # key includes strategy/status/last closed 1H candle, so an unchanged
        # NOT_MET universe never creates 97 x 9 records every five minutes.
        rows_by_symbol = {item.market.symbol: item for item in enriched}
        evaluation_results = [
            item for item in results
            if item.near_candidate or item.status in {"CONFIRMED_SIGNAL", "CONFIRMED_SIGNAL_WITH_WARNING"}
        ]
        evaluation_symbols: list[str] = []
        evaluation_symbol_priority: dict[str, tuple[float, float]] = {}
        if not use_dry_run:
            for item in evaluation_results:
                symbol = item.symbol.upper()
                row = rows_by_symbol.get(symbol)
                if row is None:
                    continue
                last = (technical_by_symbol.get(symbol, {}).get("last_closed_1h") or {})
                last_open = last.get("open_time_ms") if isinstance(last.get("open_time_ms"), int) else None
                if last_open is None:
                    # No closed 1H candle means no stable evaluation identity.
                    continue
                if store.has_signal_evaluation(
                    contract_type=scrape.contract_type,
                    symbol=symbol,
                    scenario=_scenario_code(item.scenario),
                    strategy_version=__version__,
                    status=item.status,
                    last_closed_1h_open_time_ms=last_open,
                ):
                    continue
                evaluation_symbol_priority[symbol] = max(
                    evaluation_symbol_priority.get(symbol, (0.0, 0.0)),
                    (float(row.market.volume24_usd or 0.0), float(row.market.oi_usd or 0.0)),
                )
            max_context = max(0, int(settings.p1.get("scanner", {}).get("market_context_max_candidates", 20)))
            evaluation_symbols = sorted(
                evaluation_symbol_priority,
                key=lambda symbol: evaluation_symbol_priority[symbol],
                reverse=True,
            )[:max_context]

        market_context_by_symbol: dict[str, dict[str, Any]] = {}
        if evaluation_symbols:
            spot_client = SpotMarketClient(min_interval_sec=settings.binance_request_interval_sec)
            for symbol in evaluation_symbols:
                futures_symbol = str(technical_by_symbol.get(symbol, {}).get("binance_symbol") or "")
                if not futures_symbol:
                    market_context_by_symbol[symbol] = {"status": "FUTURES_SYMBOL_UNAVAILABLE"}
                    continue
                try:
                    market_context_by_symbol[symbol] = _market_context_for_symbol(
                        binance, spot_client, futures_symbol,
                        candles_1h_by_symbol.get(symbol, []), settings,
                    )
                except BinanceScanAbortError as exc:
                    market_context_by_symbol[symbol] = {
                        "status": "BINANCE_CONTEXT_ABORT",
                        "futures_symbol": futures_symbol,
                        "error": str(exc)[:300],
                    }
                    if exc.retry_after_seconds is not None and exc.retry_after_seconds > 0:
                        try:
                            store.record_binance_cooldown(exc.retry_after_seconds)
                        except Exception:
                            log.exception("Could not persist Binance cooldown after optional context request")
                    log.error("Optional market-context collection stopped after Binance guard: %s", exc)
                    break
                except BinanceDataError as exc:
                    market_context_by_symbol[symbol] = {
                        "status": "PARTIAL", "futures_symbol": futures_symbol,
                        "error": str(exc)[:300],
                    }
                except Exception as exc:
                    market_context_by_symbol[symbol] = {
                        "status": "PARTIAL", "futures_symbol": futures_symbol,
                        "error": str(exc)[:300],
                    }

        if not use_dry_run:
            for item in evaluation_results:
                symbol = item.symbol.upper()
                row = rows_by_symbol.get(symbol)
                if row is None:
                    continue
                last = technical_by_symbol.get(symbol, {}).get("last_closed_1h") or {}
                if not isinstance(last.get("open_time_ms"), int):
                    continue
                market_record = row.market.to_dict()
                market_record.pop("raw", None)
                store.record_signal_evaluation({
                    "contract_type": scrape.contract_type,
                    "symbol": symbol,
                    "scenario": _scenario_code(item.scenario),
                    "strategy_version": __version__,
                    "status": item.status,
                    "last_closed_1h_open_time_ms": last.get("open_time_ms"),
                    "evaluated_at": scrape.fetched_at.isoformat(),
                    "direction": item.direction,
                    "reason": item.reason,
                    "reason_code": item.reason_code,
                    "diagnostic_status": item.diagnostic_status,
                    "failed_checks": list(item.failed_checks),
                    "unknown_checks": list(item.unknown_checks),
                    "summary": item.summary,
                    "checks": [
                        {"name": check.name, "passed": check.passed, "actual": check.actual, "rule": check.rule}
                        for check in item.checks
                    ],
                    "features": market_record,
                    "technical": item.technical,
                    "market_context": market_context_by_symbol.get(symbol),
                    "data_quality": {
                        # Scenario rule UNKNOWNs are deliberately not conflated
                        # with missing optional market-context measurements.
                        "unknown_checks": item.unknown_count,
                        "quality_flags": list(item.quality_flags),
                        **_market_context_quality(market_context_by_symbol.get(symbol)),
                    },
                })

        # Active-signal progress is sampled once per newly closed 1H candle.
        # OHLC paths are hourly and do not claim intrabar event ordering.
        if not use_dry_run and active_states_before_scan:
            _record_signal_progress(
                store, active_states_before_scan, candles_1h_by_symbol,
                maker_fee_bps=float(getattr(settings, "maker_fee_bps", 2.0)),
                taker_fee_bps=float(getattr(settings, "taker_fee_bps", 5.0)),
                entry_execution_type=str(getattr(settings, "shadow_entry_execution_type", "TAKER")),
                exit_execution_type=str(getattr(settings, "shadow_exit_execution_type", "TAKER")),
                notional_usd=float(getattr(settings, "shadow_notional_usd", 1000.0)),
            )

        allowed, guard_reason = _can_send_alerts(scrape, settings)
        # A fresh closed Binance 1H candle can confirm a new trigger even when
        # Coinalyze ratios are unchanged. Event identity and active lifecycle
        # state provide idempotency; unchanged polls still do not create 4H
        # history snapshots or advance the last-successful-poll marker.
        entry_allowed = allowed
        if unchanged:
            stats.warnings.append("Coinalyze values unchanged; fresh 1H Binance confirmations remain eligible, event deduplication is active")
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
        if run_id is not None:
            store.finish_run(run_id, len(scrape.rows), len(visible_results), "OK")
        retryable_delivery_failures = _has_retryable_notification_failures(notifications)
        if retryable_delivery_failures:
            log.warning(
                "Last-poll marker not advanced: at least one notification has a retryable delivery failure"
            )
        poll_state_updated = False
        if (
            not use_dry_run
            and allowed
            and not unchanged
            and not retryable_delivery_failures
            and settings.telegram_bot_token
            and settings.telegram_chat_ids
        ):
            store.record_poll(scrape.contract_type, scrape.rows, polled_at=scrape.fetched_at)
            poll_state_updated = True
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
            "retryable_delivery_failures": retryable_delivery_failures,
            "poll_state_updated": poll_state_updated,
            "alert_guard": guard_reason,
        }
    except IncompleteUniverseError as exc:
        if run_id is not None:
            store.finish_run(run_id, 0, 0, "INCOMPLETE_UNIVERSE", str(exc))
        log.exception("Coinalyze universe incomplete; no analysis/notifications sent")
        raise
    except BinanceScanAbortError as exc:
        if run_id is not None:
            store.finish_run(run_id, 0, 0, "BINANCE_ABORT", str(exc))
        if (
            not isinstance(exc, BinanceCooldownActiveError)
            and exc.retry_after_seconds is not None
            and exc.retry_after_seconds > 0
        ):
            try:
                until = store.record_binance_cooldown(exc.retry_after_seconds)
                if until is not None:
                    log.error("Persisted Binance cooldown until %s", until.isoformat())
            except Exception:
                log.exception("Could not persist Binance cooldown; manual review is required")
        log.error("Binance scan aborted: %s", exc)
        raise
    except Exception as exc:
        if run_id is not None:
            store.finish_run(run_id, 0, 0, "FAILED", str(exc))
        log.exception("P1 scan failed")
        raise
