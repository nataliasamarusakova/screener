from __future__ import annotations

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
    side = side.upper()
    highs, lows = confirmed_pivots(candles, left, right, lookback)
    pivots = highs if side == "LONG" else lows
    if not pivots:
        return {"confirmed": False, "reason": f"No confirmed 1H swing {'high' if side == 'LONG' else 'low'}"}
    reference_idx, reference_level = pivots[-1]
    volatility = atr(candles)
    if volatility is None:
        return {
            "confirmed": False,
            "reason": "Insufficient candles to calculate ATR",
            "level": reference_level,
            "level_open_time_ms": candles[reference_idx].open_time_ms,
            "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
        }
    tolerance = volatility * atr_tolerance
    last_index = len(candles) - 1
    earliest_retest = max(0, last_index - retest_lookback + 1)

    for pivot_idx, level in reversed(pivots):
        breakout_indices: list[int] = []
        for i in range(pivot_idx + 1, last_index + 1):
            if side == "LONG" and candles[i].close > level:
                breakout_indices.append(i)
            elif side == "SHORT" and candles[i].close < level:
                breakout_indices.append(i)
        if not breakout_indices:
            continue
        breakout_idx = breakout_indices[0]
        # Require a later closed candle to test the level and close back on the breakout side.
        for i in range(max(breakout_idx + 1, earliest_retest), last_index + 1):
            candle = candles[i]
            if side == "LONG":
                touched = candle.low <= level + tolerance and candle.low >= level - 2 * tolerance
                held = candle.close > level
            else:
                touched = candle.high >= level - tolerance and candle.high <= level + 2 * tolerance
                held = candle.close < level
            if touched and held:
                return {
                    "confirmed": True,
                    "reason": "Closed 1H breakout followed by a successful closed-candle retest",
                    "side": side,
                    "level": level,
                    "atr14": volatility,
                    "tolerance": tolerance,
                    "breakout_open_time_ms": candles[breakout_idx].open_time_ms,
                    "retest_open_time_ms": candle.open_time_ms,
                    "last_close": candles[-1].close,
                }
    return {
        "confirmed": False,
        "reason": "No 1H close + subsequent retest found in the recent window",
        "level": reference_level,
        "level_open_time_ms": candles[reference_idx].open_time_ms,
        "atr14": volatility,
        "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
    }


def bearish_structure_break(candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60) -> dict[str, Any]:
    _, lows = confirmed_pivots(candles, left, right, lookback)
    if not lows:
        return {"confirmed": False, "reason": "No confirmed 1H swing low"}
    pivot_idx, level = lows[-1]
    last = candles[-1]
    confirmed = last.close < level and len(candles) - 1 > pivot_idx
    return {
        "confirmed": confirmed,
        "reason": "Last closed 1H candle closed below the latest confirmed swing low" if confirmed else "Latest confirmed swing low has not been broken by a 1H close",
        "level": level,
        "last_close": last.close,
        "break_open_time_ms": last.open_time_ms if confirmed else None,
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
            return {"confirmed": True, "reason": "Price traded above a prior 1H swing high but the latest close is back below it", "failed_level": level, "last_close": last.close}
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
