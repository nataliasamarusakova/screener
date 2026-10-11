from __future__ import annotations

from engine.level_retest import (
    LevelRetestConfig,
    OHLCVCandle,
    detect_bullish_level_retest,
)

INTERVAL_MS = 12 * 60 * 60 * 1000


def _bar(i: int, *, open_: float = 100.0, high: float = 103.0,
         low: float = 98.0, close: float = 101.0) -> OHLCVCandle:
    return OHLCVCandle(
        open_time_ms=i * INTERVAL_MS,
        open=open_, high=high, low=low, close=close, volume=1000.0,
    )


def _valid_pattern() -> tuple[list[OHLCVCandle], OHLCVCandle, float]:
    bars = [_bar(i) for i in range(55)]
    # Three separated former highs near 105, each confirmed by surrounding bars.
    for i in (5, 13, 21):
        bars[i] = _bar(i, open_=102.0, high=105.0, low=100.0, close=102.0)
    # A closed candle breaks above the former-high cluster.
    bars[24] = _bar(24, open_=103.0, high=113.0, low=102.0, close=112.0)
    # Price holds above the old resistance before returning to test it.
    for i in range(25, 54):
        bars[i] = _bar(i, open_=115.0, high=120.0, low=110.0, close=117.0)
    bars[54] = _bar(54, open_=110.0, high=111.0, low=104.8, close=106.3)
    live = _bar(55, open_=106.0, high=107.0, low=105.4, close=106.4)
    return bars, live, 106.4


def test_detects_old_peaks_breakout_and_recent_bounce_from_above():
    bars, live, price = _valid_pattern()
    event = detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=price,
    )
    assert event is not None
    assert event.pattern_type == "BULLISH_RESISTANCE_FLIP_RETEST"
    assert event.timeframe == "12h"
    assert event.support_level == 105.0
    assert event.touch_count >= 2
    assert len(event.prior_peak_times_ms) == event.touch_count
    assert event.breakout_time_ms == bars[24].open_time_ms
    assert event.retest_time_ms >= bars[54].open_time_ms
    assert event.price > event.support_level
    assert event.score >= 55.0
    assert event.invalidation_price < event.support_level
    assert event.to_dict()["event_key"] == event.event_key


def test_does_not_signal_when_current_price_has_lost_support():
    bars, live, _ = _valid_pattern()
    event = detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=103.0,
    )
    assert event is None


def test_does_not_signal_without_confirmed_breakout_close():
    bars, live, price = _valid_pattern()
    bars[24] = _bar(24, open_=103.0, high=105.8, low=102.0, close=105.2)
    # Hold every later close below the breakout threshold too.
    for i in range(25, 54):
        bars[i] = _bar(i, open_=104.0, high=105.7, low=102.0, close=104.5)
    bars[54] = _bar(54, open_=104.0, high=106.0, low=104.0, close=105.5)
    event = detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=price,
    )
    assert event is None


def test_does_not_signal_with_only_one_old_peak():
    bars, live, price = _valid_pattern()
    bars[13] = _bar(13)
    bars[21] = _bar(21)
    event = detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=price,
        config=LevelRetestConfig(min_score=0.0),
    )
    assert event is None


def test_rejects_invalid_or_unsorted_candle_history():
    bars, live, price = _valid_pattern()
    bars[3] = OHLCVCandle(
        open_time_ms=bars[2].open_time_ms,
        open=100, high=103, low=98, close=101, volume=1,
    )
    assert detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=price,
    ) is None


def test_does_not_use_pivot_unconfirmed_before_breakout(monkeypatch):
    bars, live, price = _valid_pattern()
    # The peak at index 23 needs two bars to its right, so its confirmation
    # would consume the breakout candle at index 24 and a post-breakout candle.
    # Pretend the low-level pivot finder surfaced it; the detector must still
    # exclude it from the level that supposedly existed before that breakout.
    monkeypatch.setattr(
        "engine.level_retest._confirmed_pivot_highs",
        lambda *_args: [(5, 105.0), (23, 105.5)],
    )
    assert detect_bullish_level_retest(
        symbol="TESTUSDT", completed_candles=bars, live_candle=live,
        current_price=price,
    ) is None
