"""12h resistance-to-support flip detector (former swing highs retested from above).

This module is deliberately pure: it only evaluates completed OHLCV bars plus the
current 12h bar/price, and has no dependency on the existing 5m signal engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional, Sequence
import math


@dataclass(frozen=True)
class OHLCVCandle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def is_valid(self) -> bool:
        values = (self.open, self.high, self.low, self.close, self.volume)
        return (
            self.open_time_ms >= 0
            and all(math.isfinite(float(v)) for v in values)
            and min(self.open, self.high, self.low, self.close) > 0.0
            and self.volume >= 0.0
            and self.high >= max(self.open, self.close, self.low)
            and self.low <= min(self.open, self.close, self.high)
        )


@dataclass(frozen=True)
class LevelRetestConfig:
    """Conservative defaults for a first research version; tune by walk-forward."""
    pivot_left: int = 2
    pivot_right: int = 2
    min_touches: int = 2
    min_touch_spacing_bars: int = 3
    level_cluster_tolerance_pct: float = 0.012
    base_touch_tolerance_pct: float = 0.005
    atr_zone_multiplier: float = 0.30
    max_touch_tolerance_pct: float = 0.015
    min_breakout_pct: float = 0.004
    atr_breakout_multiplier: float = 0.15
    max_breakout_buffer_pct: float = 0.010
    min_bounce_pct: float = 0.003
    max_support_penetration_pct: float = 0.012
    max_price_extension_pct: float = 0.025
    recent_retest_bars: int = 3
    atr_lookback: int = 14
    min_score: float = 55.0

    def __post_init__(self) -> None:
        if self.pivot_left < 1 or self.pivot_right < 1:
            raise ValueError("pivot_left and pivot_right must be >= 1")
        if self.min_touches < 2 or self.min_touch_spacing_bars < 1:
            raise ValueError("min_touches must be >= 2 and touch spacing >= 1")
        pct_fields = (
            self.level_cluster_tolerance_pct,
            self.base_touch_tolerance_pct,
            self.atr_zone_multiplier,
            self.max_touch_tolerance_pct,
            self.min_breakout_pct,
            self.atr_breakout_multiplier,
            self.max_breakout_buffer_pct,
            self.min_bounce_pct,
            self.max_support_penetration_pct,
            self.max_price_extension_pct,
        )
        if not all(math.isfinite(x) and x > 0.0 for x in pct_fields):
            raise ValueError("percentage and ATR multiplier parameters must be finite and positive")
        if self.recent_retest_bars < 1 or self.atr_lookback < 2:
            raise ValueError("recent_retest_bars must be >= 1 and atr_lookback >= 2")
        if not (0.0 <= self.min_score <= 100.0):
            raise ValueError("min_score must be within [0, 100]")


@dataclass(frozen=True)
class LevelRetestEvent:
    symbol: str
    pattern_type: str
    timeframe: str
    support_level: float
    zone_low: float
    zone_high: float
    price: float
    distance_to_level_pct: float
    touch_count: int
    prior_peak_times_ms: tuple[int, ...]
    breakout_time_ms: int
    retest_time_ms: int
    bars_since_breakout: int
    atr_pct: float
    breakout_pct: float
    bounce_pct: float
    score: float
    invalidation_price: float

    @property
    def event_key(self) -> str:
        # Deduplicate one support flip while allowing a genuinely new breakout later.
        return f"{self.symbol}:{self.breakout_time_ms}:{self.support_level:.8g}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["prior_peak_times_ms"] = list(self.prior_peak_times_ms)
        result["event_key"] = self.event_key
        return result


def _atr_pct(candles: Sequence[OHLCVCandle], lookback: int) -> float:
    recent = candles[-(lookback + 1):]
    true_ranges: list[float] = []
    for index, candle in enumerate(recent):
        if index == 0:
            true_range = candle.high - candle.low
        else:
            previous_close = recent[index - 1].close
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        if candle.close > 0.0:
            true_ranges.append(true_range / candle.close)
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def _confirmed_pivot_highs(
    candles: Sequence[OHLCVCandle], left: int, right: int,
) -> list[tuple[int, float]]:
    """Only return pivots with all right-hand confirmation bars already closed."""
    pivots: list[tuple[int, float]] = []
    for index in range(left, len(candles) - right):
        high = candles[index].high
        left_highs = [candles[j].high for j in range(index - left, index)]
        right_highs = [candles[j].high for j in range(index + 1, index + right + 1)]
        if high >= max(left_highs) and high > max(right_highs):
            pivots.append((index, high))
    return pivots


def _spaced_touches(
    points: Sequence[tuple[int, float]], spacing: int,
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    for point in points:
        if not selected or point[0] - selected[-1][0] >= spacing:
            selected.append(point)
        elif point[1] > selected[-1][1]:
            # Keep the more pronounced high if two detected pivots are too close.
            selected[-1] = point
    return selected


def detect_bullish_level_retest(
    *,
    symbol: str,
    completed_candles: Sequence[OHLCVCandle],
    live_candle: Optional[OHLCVCandle],
    current_price: float,
    config: LevelRetestConfig = LevelRetestConfig(),
) -> Optional[LevelRetestEvent]:
    """Find former-high resistance that was broken and is now holding as support.

    Required sequence:
      1. >=2 separated, confirmed 12h swing highs cluster around a price level;
      2. a later *closed* 12h candle closes decisively above that level;
      3. a recent later candle revisits the level from above and rebounds;
      4. the current price remains close enough to the level to be actionable as a setup.

    No future candles are used for pivot confirmation or breakout validation.
    """
    if not symbol or not math.isfinite(current_price) or current_price <= 0.0:
        return None
    bars = list(completed_candles)
    if len(bars) < max(20, config.pivot_left + config.pivot_right + 5):
        return None
    if any(not candle.is_valid() for candle in bars):
        return None
    if any(bars[i].open_time_ms >= bars[i + 1].open_time_ms for i in range(len(bars) - 1)):
        return None
    if live_candle is not None and not live_candle.is_valid():
        return None

    atr_pct = _atr_pct(bars, config.atr_lookback)
    touch_tolerance = min(
        config.max_touch_tolerance_pct,
        max(config.base_touch_tolerance_pct, atr_pct * config.atr_zone_multiplier),
    )
    breakout_buffer = min(
        config.max_breakout_buffer_pct,
        max(config.min_breakout_pct, atr_pct * config.atr_breakout_multiplier),
    )
    max_penetration = max(config.max_support_penetration_pct, touch_tolerance * 1.25)
    pivots = _confirmed_pivot_highs(bars, config.pivot_left, config.pivot_right)
    if len(pivots) < config.min_touches:
        return None

    # Examine only unique pivot-centred candidates; the eventual level is a robust median.
    candidate_levels: list[LevelRetestEvent] = []
    seen_levels: set[tuple[str, int]] = set()
    for seed_idx, seed_high in pivots:
        for breakout_idx in range(seed_idx + 1, len(bars)):
            # Pivots must precede the breakout; post-breakout highs cannot define the level.
            prior = [
                p for p in pivots
                # The pivot's right-hand confirmation candles must themselves
                # close before the breakout candle; otherwise the detector would
                # use post-breakout data to label an earlier high as resistance.
                if p[0] + config.pivot_right < breakout_idx
                and abs(p[1] - seed_high) / seed_high <= config.level_cluster_tolerance_pct
            ]
            prior = _spaced_touches(prior, config.min_touch_spacing_bars)
            if len(prior) < config.min_touches:
                continue
            level = float(median(price for _, price in prior))
            if breakout_idx <= prior[-1][0]:
                continue
            breakout_close = bars[breakout_idx].close
            threshold = level * (1.0 + breakout_buffer)
            if breakout_close < threshold:
                continue
            # Only the first qualifying close after the final old peak is the breakout.
            first_breakout = next(
                (
                    j for j in range(prior[-1][0] + 1, len(bars))
                    if bars[j].close >= threshold
                    and bars[j - 1].close < threshold
                ),
                None,
            )
            if first_breakout != breakout_idx:
                continue

            key = (format(level, ".8g"), bars[breakout_idx].open_time_ms)
            if key in seen_levels:
                continue

            post_breakout: list[tuple[OHLCVCandle, float]] = [
                (bar, bar.close) for bar in bars[breakout_idx + 1:]
            ]
            if live_candle is not None:
                post_breakout.append((live_candle, current_price))
            if not post_breakout:
                continue

            latest = post_breakout[-config.recent_retest_bars:]
            found_retest: Optional[tuple[OHLCVCandle, float, float]] = None
            for bar, effective_close in reversed(latest):
                if effective_close <= 0.0:
                    continue
                low_vs_level = bar.low / level - 1.0
                bounce_pct = effective_close / bar.low - 1.0
                closes_above = effective_close >= level * 0.998
                stays_near = effective_close <= level * (1.0 + config.max_price_extension_pct)
                touches_zone = low_vs_level <= touch_tolerance
                does_not_fail = low_vs_level >= -max_penetration
                bounced = bounce_pct >= config.min_bounce_pct
                # For completed bars the close is final; for the live bar it is the latest price.
                if closes_above and stays_near and touches_zone and does_not_fail and bounced:
                    found_retest = (bar, effective_close, bounce_pct)
                    break
            if found_retest is None:
                continue

            retest_bar, _retest_close, bounce_pct = found_retest
            current_distance = (current_price / level - 1.0) * 100.0
            if (
                current_price < level * 0.998
                or current_price > level * (1.0 + config.max_price_extension_pct)
            ):
                continue

            breakout_pct = (breakout_close / level - 1.0) * 100.0
            # Transparent heuristic ranking, not a probability estimate.
            touch_points = min(20.0, 10.0 * (len(prior) - config.min_touches + 1))
            breakout_points = min(25.0, max(0.0, (breakout_pct - breakout_buffer * 100.0) * 2.0))
            bounce_points = min(25.0, bounce_pct * 100.0 * 8.0)
            distance_points = max(0.0, 20.0 - abs(current_distance) * 4.0)
            bars_after_retest = sum(
                1 for bar, _close in post_breakout
                if bar.open_time_ms > retest_bar.open_time_ms
            )
            recency_points = max(0.0, 10.0 - 4.0 * bars_after_retest)
            score = min(100.0, touch_points + breakout_points + bounce_points + distance_points + recency_points)
            if score < config.min_score:
                continue

            seen_levels.add(key)
            candidate_levels.append(LevelRetestEvent(
                symbol=symbol,
                pattern_type="BULLISH_RESISTANCE_FLIP_RETEST",
                timeframe="12h",
                support_level=round(level, 8),
                zone_low=round(level * (1.0 - touch_tolerance), 8),
                zone_high=round(level * (1.0 + touch_tolerance), 8),
                price=round(current_price, 8),
                distance_to_level_pct=round(current_distance, 4),
                touch_count=len(prior),
                prior_peak_times_ms=tuple(bars[index].open_time_ms for index, _ in prior),
                breakout_time_ms=bars[breakout_idx].open_time_ms,
                retest_time_ms=retest_bar.open_time_ms,
                bars_since_breakout=len(bars) - 1 - breakout_idx,
                atr_pct=round(atr_pct * 100.0, 4),
                breakout_pct=round(breakout_pct, 4),
                bounce_pct=round(bounce_pct * 100.0, 4),
                score=round(score, 2),
                invalidation_price=round(level * (1.0 - max_penetration), 8),
            ))

    if not candidate_levels:
        return None
    return max(candidate_levels, key=lambda event: (event.score, event.touch_count, -abs(event.distance_to_level_pct)))
