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
        return {
            "confirmed": False,
            "invalidated": False,
            "reason": "Need at least two confirmed 1H swing lows",
            "pivot_lows": [x[1] for x in lows],
        }
    previous, latest = lows[-2], lows[-1]
    is_higher = latest[1] > previous[1]
    # A higher low is no longer valid after a later *closed* 1H candle closes
    # below that pivot. The wick alone does not invalidate the structure.
    invalidating = next(
        (
            (idx, candle)
            for idx, candle in enumerate(candles[latest[0] + 1 :], start=latest[0] + 1)
            if candle.close < latest[1]
        ),
        None,
    )
    invalidated = bool(is_higher and invalidating is not None)
    confirmed = bool(is_higher and not invalidated)
    result: dict[str, Any] = {
        "confirmed": confirmed,
        "invalidated": invalidated,
        "reason": (
            "Higher low was invalidated by a later 1H close below its pivot"
            if invalidated
            else "Latest confirmed swing low is higher"
            if confirmed
            else "Latest confirmed swing low is not higher"
        ),
        "previous_low": previous[1],
        "latest_low": latest[1],
        "previous_open_time_ms": candles[previous[0]].open_time_ms,
        "latest_open_time_ms": candles[latest[0]].open_time_ms,
    }
    if invalidating is not None:
        invalid_idx, invalid_candle = invalidating
        result["invalidation_open_time_ms"] = invalid_candle.open_time_ms
        result["invalidation_close"] = invalid_candle.close
    return result


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
    """Confirm the latest pivot's breakout/retest, unless a later close invalidates it."""
    side = side.upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    highs, lows = confirmed_pivots(candles, left, right, lookback)
    pivots = highs if side == "LONG" else lows
    if not pivots:
        return {
            "confirmed": False,
            "invalidated": False,
            "reason": f"No confirmed 1H swing {'high' if side == 'LONG' else 'low'}",
        }

    # Choose the newest confirmed swing level that was actually broken after
    # it formed. A local high/low made during the breakout leg can itself become
    # a newer pivot; treating that as the reference too early would discard the
    # valid breakout/retest of the prior structural level.
    selected: tuple[int, float] | None = None
    for candidate_idx, candidate_level in reversed(pivots):
        candidate_breakout = next(
            (
                idx for idx in range(candidate_idx + 1, len(candles))
                if (candles[idx].close > candidate_level if side == "LONG" else candles[idx].close < candidate_level)
            ),
            None,
        )
        if candidate_breakout is not None:
            selected = (candidate_idx, candidate_level)
            break
    if selected is None:
        pivot_idx, level = pivots[-1]
        return {
            "side": side,
            "level": level,
            "level_open_time_ms": candles[pivot_idx].open_time_ms,
            "confirmed": False,
            "invalidated": False,
            "reason": "No closed 1H breakout beyond a confirmed swing level",
            "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
        }
    pivot_idx, level = selected
    volatility = atr(candles)
    base = {
        "side": side,
        "level": level,
        "level_open_time_ms": candles[pivot_idx].open_time_ms,
    }
    if volatility is None:
        return {
            **base,
            "confirmed": False,
            "invalidated": False,
            "reason": "Insufficient candles to calculate ATR",
            "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
        }

    tolerance = volatility * atr_tolerance
    last_index = len(candles) - 1
    earliest_retest = max(0, last_index - retest_lookback + 1)
    breakout_idx: int | None = None
    retest_idx: int | None = None
    invalidating: Candle | None = None
    most_recent_breakout_idx: int | None = None

    # Walk the closed candles as a small state machine. A close back through the
    # level before retest cancels that breakout attempt; after a valid retest it
    # invalidates the setup. A later fresh breakout may start a new attempt.
    for idx in range(pivot_idx + 1, last_index + 1):
        candle = candles[idx]
        breakout_close = candle.close > level if side == "LONG" else candle.close < level
        failed_close = candle.close < level if side == "LONG" else candle.close > level

        if retest_idx is not None:
            if failed_close:
                invalidating = candle
                retest_idx = None
                breakout_idx = None
            continue

        if breakout_idx is None:
            if breakout_close:
                # A fresh breakout supersedes any previously invalidated attempt.
                invalidating = None
                breakout_idx = idx
                most_recent_breakout_idx = idx
            continue

        if failed_close:
            # The attempted breakout failed before a qualifying retest.
            breakout_idx = None
            continue

        if side == "LONG":
            touched = candle.low <= level + tolerance and candle.low >= level - 2 * tolerance
            held = candle.close > level
        else:
            touched = candle.high >= level - tolerance and candle.high <= level + 2 * tolerance
            held = candle.close < level
        if idx >= earliest_retest and touched and held:
            retest_idx = idx
            invalidating = None

    if retest_idx is None:
        return {
            **base,
            "confirmed": False,
            "invalidated": invalidating is not None,
            "reason": (
                "A later closed 1H candle crossed back through the retest level"
                if invalidating is not None
                else "No successful closed-candle retest after the latest breakout attempt"
            ),
            "atr14": volatility,
            "tolerance": tolerance,
            "breakout_open_time_ms": (
                candles[most_recent_breakout_idx].open_time_ms if most_recent_breakout_idx is not None else None
            ),
            "invalidation_open_time_ms": invalidating.open_time_ms if invalidating else None,
            "invalidation_close": invalidating.close if invalidating else None,
            "invalidation_rule": "LONG: 1H close below level" if side == "LONG" else "SHORT: 1H close above level",
        }

    return {
        **base,
        "confirmed": True,
        "invalidated": False,
        "reason": "Closed 1H breakout followed by a successful retest; no later close invalidated the latest attempt",
        "atr14": volatility,
        "tolerance": tolerance,
        "breakout_open_time_ms": candles[breakout_idx].open_time_ms,
        "retest_open_time_ms": candles[retest_idx].open_time_ms,
        "last_close": candles[-1].close,
        "invalidation_open_time_ms": invalidating.open_time_ms if invalidating else None,
        "invalidation_close": invalidating.close if invalidating else None,
    }

def bearish_structure_break(candles: list[Candle], left: int = 2, right: int = 2, lookback: int = 60) -> dict[str, Any]:
    _, lows = confirmed_pivots(candles, left, right, lookback)
    if not lows:
        return {"confirmed": False, "reason": "No confirmed 1H swing low"}
    pivot_idx, level = lows[-1]
    last = candles[-1]
    confirmed = last.close < level and len(candles) - 1 > pivot_idx
    return {"confirmed": confirmed, "reason": "Last closed 1H candle closed below the latest confirmed swing low" if confirmed else "Latest confirmed swing low has not been broken by a 1H close", "level": level, "last_close": last.close}


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
        return {
            "confirmed": False,
            "invalidated": False,
            "reason": "Need two confirmed swing highs to identify a lower high",
            "candidate_resistance": highs[-1][1] if highs else None,
        }
    previous, latest = highs[-2], highs[-1]
    if latest[1] >= previous[1]:
        return {
            "confirmed": False,
            "invalidated": False,
            "reason": "Bounce high is not lower than the previous swing high",
            "candidate_resistance": latest[1],
        }
    volatility = atr(candles)
    if volatility is None:
        return {
            "confirmed": False,
            "invalidated": False,
            "reason": "Insufficient candles to calculate ATR",
            "candidate_resistance": latest[1],
        }

    tolerance = volatility * atr_tolerance
    resistance = latest[1]
    rejection_candidates: list[tuple[int, Candle]] = []
    start = max(0, len(candles) - 3)
    for idx in range(start, len(candles)):
        candle = candles[idx]
        body = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        wick_ok = upper_wick >= max(body * wick_body_ratio, volatility * 0.05)
        near_resistance = resistance - tolerance <= candle.high <= resistance + 2 * tolerance
        candle_range = candle.high - candle.low
        bearish_close = (
            candle.close < candle.open
            and candle_range > 0
            and candle.close <= candle.low + candle_range * 0.45
        )
        if wick_ok and near_resistance and bearish_close:
            rejection_candidates.append((idx, candle))

    for idx, candle in reversed(rejection_candidates):
        invalidating = next(
            (later for later in candles[idx + 1 :] if later.close > resistance),
            None,
        )
        if invalidating is None:
            return {
                "confirmed": True,
                "invalidated": False,
                "reason": "Lower-high bounce rejected near resistance; no later 1H close reclaimed resistance",
                "resistance": resistance,
                "previous_swing_high": previous[1],
                "latest_swing_high": latest[1],
                "rejection_open_time_ms": candle.open_time_ms,
                "atr14": volatility,
            }
        return {
            "confirmed": False,
            "invalidated": True,
            "reason": "A later closed 1H candle reclaimed resistance after the rejection",
            "resistance": resistance,
            "previous_swing_high": previous[1],
            "latest_swing_high": latest[1],
            "rejection_open_time_ms": candle.open_time_ms,
            "invalidation_open_time_ms": invalidating.open_time_ms,
            "invalidation_close": invalidating.close,
            "atr14": volatility,
        }

    return {
        "confirmed": False,
        "invalidated": False,
        "reason": "No closed 1H rejection candle at the lower-high resistance",
        "candidate_resistance": resistance,
    }

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
    """Select an active 1H invalidation level only from a confirmed structure."""
    direction = direction.upper()
    scenario = scenario.upper()
    if direction == "LONG":
        if scenario.startswith(("LONG B", "LONG D")):
            source = technical.get("higher_low") or {}
            source_name = "confirmed 1H higher-low pivot"
        else:
            source = technical.get("trend_retest_long") or technical.get("1h_retest") or {}
            source_name = "confirmed 1H breakout/retest level"
        confirmed = source.get("confirmed") is True
        level = source.get("latest_low") if scenario.startswith(("LONG B", "LONG D")) else source.get("level")
        rule = "1H close below level invalidates the LONG structure"
    else:
        if scenario.startswith("SHORT D"):
            source = technical.get("weak_bounce_rejection") or {}
            source_name = "confirmed 1H weak-bounce resistance"
            level = source.get("resistance")
        elif scenario.startswith("SHORT E"):
            source = technical.get("failed_breakout_rejection") or {}
            source_name = "confirmed failed 1H breakout level"
            level = source.get("failed_level")
        elif scenario.startswith("SHORT B"):
            source = technical.get("trend_retest_short") or technical.get("1h_retest") or {}
            source_name = "confirmed 1H breakdown/retest level"
            level = source.get("level")
        elif scenario.startswith("SHORT C"):
            source = technical.get("bearish_structure_break") or {}
            source_name = "confirmed 1H bearish structure level"
            level = source.get("level")
        else:
            source = technical.get("trend_retest_short") or technical.get("1h_retest") or {}
            source_name = "confirmed 1H breakdown/retest level"
            level = source.get("level")
        confirmed = source.get("confirmed") is True
        rule = "1H close above level invalidates the SHORT structure"

    valid_level = isinstance(level, (int, float)) and not isinstance(level, bool)
    if not confirmed or not valid_level:
        return {
            "level": None,
            "rule": "not determined from confirmed closed-candle structure",
            "source": None,
        }
    return {"level": float(level), "rule": rule, "source": source_name}

