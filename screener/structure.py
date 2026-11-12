from __future__ import annotations

import math
from typing import Any

from .models import Candle


def atr(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        current = candles[i]
        previous_close = candles[i - 1].close
        true_ranges.append(max(current.high - current.low, abs(current.high - previous_close), abs(current.low - previous_close)))
    sample = true_ranges[-period:]
    return sum(sample) / len(sample) if sample else None


def confirmed_pivots(
    candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return confirmed (index, price) swing highs/lows; right-side bars confirm a pivot."""
    if len(candles) < left + right + 3:
        return [], []
    start = max(left, len(candles) - lookback)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    last_confirmable = len(candles) - right
    for i in range(start, last_confirmable):
        left_slice = candles[i - left:i]
        right_slice = candles[i + 1:i + right + 1]
        if len(left_slice) != left or len(right_slice) != right:
            continue
        current = candles[i]
        if current.high > max(c.high for c in left_slice) and current.high >= max(c.high for c in right_slice):
            highs.append((i, current.high))
        if current.low < min(c.low for c in left_slice) and current.low <= min(c.low for c in right_slice):
            lows.append((i, current.low))
    return highs, lows


def higher_low(candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60) -> dict[str, Any]:
    _, lows = confirmed_pivots(candles, left, right, lookback)
    if len(lows) < 2:
        return {"confirmed": False, "reason": "Need at least two confirmed 1H swing lows", "pivot_lows": [x[1] for x in lows]}
    previous, latest = lows[-2], lows[-1]
    confirmed = latest[1] > previous[1]
    return {
        "confirmed": confirmed,
        "reason": "Latest confirmed swing low is higher" if confirmed else "Latest confirmed swing low is not higher",
        "previous_low": previous[1],
        "latest_low": latest[1],
        "previous_open_time_ms": candles[previous[0]].open_time_ms,
        "latest_open_time_ms": candles[latest[0]].open_time_ms,
    }


def breakout_and_retest(
    candles: list[Candle],
    side: str,
    *,
    left: int = 2,
    right: int = 2,
    lookback: int = 60,
    retest_lookback: int = 8,
    atr_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Confirm a recent breakout, then a retest that has not been invalidated.

    The state machine is important: after a valid retest, any later 1H close
    back through the level invalidates that setup. A subsequent re-break must
    produce a *new* retest; the old retest cannot become valid again merely
    because price later returns to the breakout side.
    """
    side = side.upper()
    if side not in {"LONG", "SHORT"}:
        return {"confirmed": False, "reason": f"Unsupported breakout side: {side}"}
    highs, lows = confirmed_pivots(candles, left, right, lookback)
    pivots = highs if side == "LONG" else lows
    if not pivots:
        return {"confirmed": False, "reason": f"No confirmed 1H swing {'high' if side == 'LONG' else 'low'}"}
    volatility = atr(candles)
    if volatility is None or not math.isfinite(volatility) or volatility <= 0:
        return {
            "confirmed": False,
            "reason": "Insufficient/invalid candles to calculate ATR",
            "level": pivots[-1][1],
            "level_open_time_ms": candles[pivots[-1][0]].open_time_ms,
            "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
        }

    tolerance = volatility * max(0.0, float(atr_tolerance))
    last_index = len(candles) - 1
    safe_retest_lookback = max(1, int(retest_lookback))
    earliest_retest = max(0, last_index - safe_retest_lookback + 1)

    def is_breakout_close(close: float, level: float) -> bool:
        return close > level if side == "LONG" else close < level

    def is_invalidation_close(close: float, level: float) -> bool:
        return close < level if side == "LONG" else close > level

    def retest_holds(candle: Candle, level: float) -> bool:
        if side == "LONG":
            touched = candle.low <= level + tolerance and candle.low >= level - 2 * tolerance
            return touched and candle.close > level
        touched = candle.high >= level - tolerance and candle.high <= level + 2 * tolerance
        return touched and candle.close < level

    for pivot_idx, level in reversed(pivots):
        state = "WAIT_BREAKOUT"
        breakout_idx: int | None = None
        retest_idx: int | None = None

        for index in range(pivot_idx + 1, len(candles)):
            candle = candles[index]
            close = candle.close
            if state == "WAIT_BREAKOUT":
                if is_breakout_close(close, level):
                    state = "WAIT_RETEST"
                    breakout_idx = index
                    retest_idx = None
                continue

            if state == "WAIT_RETEST":
                if is_invalidation_close(close, level):
                    # The breakout failed before a retest. A later close back
                    # through the level starts a new breakout attempt.
                    state = "WAIT_BREAKOUT"
                    breakout_idx = None
                    retest_idx = None
                    continue
                if index > (breakout_idx if breakout_idx is not None else index) and retest_holds(candle, level):
                    state = "VALID_RETEST"
                    retest_idx = index
                continue

            # A valid retest remains valid only while every later closed 1H
            # candle stays on the breakout side. A fresh touch can renew the
            # retest anchor, but a close beyond the level resets the sequence.
            if is_invalidation_close(close, level):
                state = "WAIT_BREAKOUT"
                breakout_idx = None
                retest_idx = None
            elif retest_holds(candle, level):
                retest_idx = index

        if state == "VALID_RETEST" and retest_idx is not None and retest_idx >= earliest_retest:
            assert breakout_idx is not None
            return {
                "confirmed": True,
                "reason": "Closed 1H breakout followed by a retest with no subsequent invalidating close",
                "side": side,
                "level": level,
                "pivot_open_time_ms": candles[pivot_idx].open_time_ms,
                "atr14": volatility,
                "tolerance": tolerance,
                "breakout_open_time_ms": candles[breakout_idx].open_time_ms,
                "retest_open_time_ms": candles[retest_idx].open_time_ms,
                "last_close": candles[-1].close,
            }

    return {
        "confirmed": False,
        "reason": "No recent breakout with a subsequent retest that remained valid through the latest closed 1H candle",
        "level": pivots[-1][1],
        "level_open_time_ms": candles[pivots[-1][0]].open_time_ms,
        "atr14": volatility,
        "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
    }


def bearish_structure_break(
    candles: list[Candle],
    left: int = 2,
    right: int = 2,
    lookback: int = 60,
    *,
    reference_level: float | None = None,
    reference_pivot_open_time_ms: int | None = None,
    reference_break_open_time_ms: int | None = None,
) -> dict[str, Any]:
    """Confirm a close below a selected confirmed swing low.

    When ``reference_level`` is supplied by a confirmed short retest, use that
    exact pivot rather than independently selecting the newest swing low. This
    keeps breakout, retest, invalidation and structure confirmation aligned.
    """
    _, lows = confirmed_pivots(candles, left, right, lookback)
    if not lows:
        return {"confirmed": False, "reason": "No confirmed 1H swing low"}

    if reference_pivot_open_time_ms is not None:
        matching_by_time = [
            (idx, price) for idx, price in lows
            if candles[idx].open_time_ms == reference_pivot_open_time_ms
        ]
        if not matching_by_time:
            return {
                "confirmed": False,
                "reason": "Short-retest reference pivot is not a confirmed swing low",
                "level": reference_level,
                "pivot_open_time_ms": reference_pivot_open_time_ms,
            }
        pivot_idx, level = matching_by_time[-1]
        if reference_level is not None and not math.isclose(
            float(reference_level), level, rel_tol=1e-9, abs_tol=1e-12
        ):
            return {
                "confirmed": False,
                "reason": "Short-retest pivot time and level do not match",
                "level": level,
                "pivot_open_time_ms": reference_pivot_open_time_ms,
            }
    elif reference_level is None:
        pivot_idx, level = lows[-1]
    else:
        try:
            level = float(reference_level)
        except (TypeError, ValueError):
            return {"confirmed": False, "reason": "Invalid short-retest reference level"}
        if not math.isfinite(level):
            return {"confirmed": False, "reason": "Invalid short-retest reference level"}
        matching = [(idx, price) for idx, price in lows if math.isclose(price, level, rel_tol=1e-9, abs_tol=1e-12)]
        if not matching:
            return {
                "confirmed": False,
                "reason": "Short-retest reference level is not a confirmed swing low",
                "level": level,
            }
        pivot_idx, level = matching[-1]

    if len(candles) <= pivot_idx + 1:
        return {
            "confirmed": False,
            "reason": "No closed candle after the reference swing low",
            "level": level,
            "last_close": candles[-1].close if candles else None,
        }

    break_indices = [idx for idx in range(pivot_idx + 1, len(candles)) if candles[idx].close < level]
    if reference_break_open_time_ms is not None:
        matching_breaks = [
            idx for idx in break_indices
            if candles[idx].open_time_ms == reference_break_open_time_ms
        ]
        selected_break_idx = matching_breaks[0] if matching_breaks else None
    else:
        selected_break_idx = break_indices[-1] if break_indices else None
    latest_holds_break = candles[-1].close < level
    confirmed = selected_break_idx is not None and latest_holds_break
    return {
        "confirmed": confirmed,
        "reason": (
            "A closed 1H candle broke the selected swing low and the latest close remains below it"
            if confirmed
            else "Selected swing low has not been broken and held by the latest 1H close"
        ),
        "level": level,
        "pivot_open_time_ms": candles[pivot_idx].open_time_ms,
        "last_close": candles[-1].close,
        "break_open_time_ms": candles[selected_break_idx].open_time_ms if confirmed and selected_break_idx is not None else None,
    }


def weak_bounce_rejection(
    candles: list[Candle],
    *,
    left: int = 2,
    right: int = 2,
    lookback: int = 60,
    wick_body_ratio: float = 0.75,
    atr_tolerance: float = 0.5,
) -> dict[str, Any]:
    highs, _ = confirmed_pivots(candles, left, right, lookback)
    if len(highs) < 2:
        return {"confirmed": False, "reason": "Need two confirmed swing highs to identify a lower high", "candidate_resistance": highs[-1][1] if highs else None}
    previous, latest = highs[-2], highs[-1]
    if latest[1] >= previous[1]:
        return {"confirmed": False, "reason": "Bounce high is not lower than the previous swing high", "candidate_resistance": latest[1]}
    volatility = atr(candles)
    if volatility is None:
        return {"confirmed": False, "reason": "Insufficient candles to calculate ATR", "candidate_resistance": latest[1]}
    tolerance = volatility * atr_tolerance
    resistance = latest[1]
    for candle in candles[-3:]:
        body = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        wick_ok = upper_wick >= max(body * wick_body_ratio, volatility * 0.05)
        near_resistance = candle.high >= resistance - tolerance and candle.high <= resistance + 2 * tolerance
        bearish_close = candle.close < candle.open and candle.close <= candle.low + (candle.high - candle.low) * 0.45
        if wick_ok and near_resistance and bearish_close:
            return {
                "confirmed": True,
                "reason": "Lower-high bounce rejected near resistance on a closed 1H candle",
                "resistance": resistance,
                "previous_swing_high": previous[1],
                "latest_swing_high": latest[1],
                "rejection_open_time_ms": candle.open_time_ms,
                "atr14": volatility,
            }
    return {"confirmed": False, "reason": "No closed 1H rejection candle at the lower-high resistance"}


def rising_1h(candles: list[Candle]) -> bool | None:
    if len(candles) < 2:
        return None
    return candles[-1].close > candles[-2].close


def failed_breakout_rejection(candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60) -> dict[str, Any]:
    """Detect a recent push above a confirmed 1H swing high that closed back below it."""
    highs, _ = confirmed_pivots(candles, left, right, lookback)
    if not highs or len(candles) < 2:
        return {"confirmed": False, "reason": "No confirmed 1H swing high or insufficient candles"}
    last = candles[-1]
    for pivot_idx, level in reversed(highs):
        recent = candles[max(pivot_idx + 1, len(candles) - 8):]
        ran_above = any(c.high > level for c in recent)
        if ran_above and last.close < level:
            return {"confirmed": True, "reason": "Price traded above a prior 1H swing high but the latest close is back below it", "failed_level": level, "last_close": last.close, "failure_open_time_ms": last.open_time_ms}
    return {"confirmed": False, "reason": "No recent failed upside breakout detected"}


def scenario_invalidation(direction: str, scenario: str, technical: dict[str, Any]) -> dict[str, Any]:
    """Select a scenario-specific 1H invalidation level from confirmed closed candles."""
    direction = direction.upper()
    scenario = scenario.upper()
    if direction == "LONG":
        if scenario.startswith(("LONG B", "LONG D")):
            source = technical.get("higher_low") or {}
            level = source.get("latest_low")
            source_name = "latest confirmed 1H higher-low pivot"
        else:
            source = technical.get("trend_retest_long") or technical.get("1h_retest") or {}
            level = source.get("level")
            source_name = "1H breakout/retest level"
        rule = "1H close below level invalidates the LONG structure"
    else:
        if scenario.startswith("SHORT D"):
            source = technical.get("weak_bounce_rejection") or {}
            level = source.get("resistance") or source.get("candidate_resistance")
            source_name = "1H weak-bounce resistance"
        elif scenario.startswith("SHORT E"):
            source = technical.get("failed_breakout_rejection") or {}
            level = source.get("failed_level")
            source_name = "failed 1H breakout level"
        elif scenario.startswith("SHORT C"):
            source = technical.get("bearish_structure_break") or {}
            level = source.get("level")
            source_name = "1H bearish structure level"
        elif scenario.startswith("SHORT B"):
            source = technical.get("trend_retest_short") or technical.get("bearish_structure_break") or {}
            level = source.get("level")
            source_name = "1H breakdown/retest level"
        else:
            source = technical.get("trend_retest_short") or technical.get("1h_retest") or {}
            level = source.get("level")
            source_name = "1H breakdown/retest level"
        rule = "1H close above level invalidates the SHORT structure"
    return {
        "level": float(level) if isinstance(level, (int, float)) else None,
        "rule": rule if isinstance(level, (int, float)) else "not determined from available closed candles",
        "source": source_name if isinstance(level, (int, float)) else None,
    }
