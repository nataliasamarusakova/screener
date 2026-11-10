from screener.models import Candle
from screener.structure import confirmed_pivots, higher_low


def candle(i: int, high: float, low: float, close: float | None = None) -> Candle:
    close = close if close is not None else (high + low) / 2
    return Candle(i * 60_000, i * 60_000 + 59_999, close, high, low, close, 1, 1, 10)


def test_higher_low_uses_only_confirmed_swings():
    values = [
        candle(0, 11, 9), candle(1, 12, 8), candle(2, 11, 9), candle(3, 10, 7),
        candle(4, 11, 8), candle(5, 13, 10), candle(6, 12, 9), candle(7, 11, 8),
        candle(8, 12, 9), candle(9, 14, 11), candle(10, 13, 10), candle(11, 12, 9),
    ]
    highs, lows = confirmed_pivots(values, left=1, right=1, lookback=30)
    assert len(lows) >= 2
    result = higher_low(values, left=1, right=1, lookback=30)
    assert result["confirmed"] is True


def test_scenario_specific_invalidation_uses_the_correct_structure_level():
    from screener.structure import scenario_invalidation

    long_d = scenario_invalidation("LONG", "LONG D — Post-Long-Flush Reversal", {
        "higher_low": {"confirmed": True, "latest_low": 98.5},
        "trend_retest_long": {"level": 105.0},
    })
    assert long_d["level"] == 98.5
    assert "below" in long_d["rule"]

    short_d = scenario_invalidation("SHORT", "SHORT D — Long Liquidation Cascade", {
        "weak_bounce_rejection": {"confirmed": True, "resistance": 112.0},
    })
    assert short_d["level"] == 112.0
    assert "above" in short_d["rule"]

    absent = scenario_invalidation("LONG", "LONG A — Trend Continuation", {})
    assert absent["level"] is None


def test_scenario_invalidation_never_uses_an_unconfirmed_candidate_level():
    from screener.structure import scenario_invalidation

    long_d = scenario_invalidation("LONG", "LONG D — Post-Long-Flush Reversal", {
        "higher_low": {"confirmed": False, "latest_low": 98.5},
    })
    assert long_d["level"] is None

    short_d = scenario_invalidation("SHORT", "SHORT D — Long Liquidation Cascade", {
        "weak_bounce_rejection": {"confirmed": False, "candidate_resistance": 112.0},
    })
    assert short_d["level"] is None


def test_higher_low_is_invalidated_by_a_later_closed_hourly_close():
    values = [
        candle(0, 11, 9), candle(1, 12, 8), candle(2, 11, 9), candle(3, 10, 7),
        candle(4, 11, 8), candle(5, 13, 10), candle(6, 12, 9), candle(7, 11, 8),
        candle(8, 12, 9), candle(9, 14, 11), candle(10, 13, 10), candle(11, 12, 9),
    ]
    # A new close below the most recent confirmed swing-low invalidates it.
    values.append(candle(12, 9, 7.5, close=7.9))
    result = higher_low(values, left=1, right=1, lookback=30)
    assert result["confirmed"] is False
    assert result["invalidated"] is True


def _hour_candle(index: int, open_: float, high: float, low: float, close: float) -> Candle:
    duration = 60 * 60_000
    return Candle(index * duration, (index + 1) * duration - 1, open_, high, low, close, 1.0, 1.0, 10)


def test_breakout_retest_is_invalidated_by_a_later_hourly_close_beyond_level():
    from screener.structure import breakout_and_retest

    values = [_hour_candle(i, 9.5, 10, 9, 9.5) for i in range(14)]
    values += [
        _hour_candle(14, 11, 12, 10, 11),
        _hour_candle(15, 10, 11, 9, 10),
        _hour_candle(16, 9.5, 10.8, 8.5, 9.5),
        _hour_candle(17, 10.8, 11.5, 9.5, 10.8),
        _hour_candle(18, 12, 12.5, 11.5, 12.2),
        _hour_candle(19, 12.2, 12.3, 11.8, 12.1),
        _hour_candle(20, 12.8, 13, 12.1, 12.8),
        _hour_candle(21, 13, 13.1, 12.4, 13),
        _hour_candle(22, 13.5, 13.8, 12.9, 13.5),
    ]
    active = breakout_and_retest(values, "LONG", left=1, right=1, lookback=60, retest_lookback=8)
    assert active["confirmed"] is True
    assert active["invalidated"] is False

    values.append(_hour_candle(23, 12.5, 12.7, 12.0, 12.4))
    invalidated = breakout_and_retest(values, "LONG", left=1, right=1, lookback=60, retest_lookback=8)
    assert invalidated["confirmed"] is False
    assert invalidated["invalidated"] is True
    assert invalidated["invalidation_close"] < invalidated["level"]


def test_weak_bounce_rejection_is_invalidated_when_resistance_is_reclaimed():
    from screener.structure import weak_bounce_rejection

    values = [_hour_candle(i, 9.4, 9.8, 9.0, 9.5) for i in range(14)]
    values += [
        _hour_candle(14, 11.2, 12, 10, 11.5),
        _hour_candle(15, 10.2, 10.8, 9.3, 10),
        _hour_candle(16, 10, 10.5, 9.2, 9.8),
        _hour_candle(17, 10.2, 11, 9.5, 10.5),
        _hour_candle(18, 10.3, 10.8, 9.4, 10),
        _hour_candle(19, 10.2, 10.7, 9.5, 10.1),
        _hour_candle(20, 10.4, 11, 9.9, 10.0),
    ]
    confirmed = weak_bounce_rejection(values, left=1, right=1, lookback=60)
    assert confirmed["confirmed"] is True

    values.append(_hour_candle(21, 11, 11.5, 10.8, 11.2))
    invalidated = weak_bounce_rejection(values, left=1, right=1, lookback=60)
    assert invalidated["confirmed"] is False
    assert invalidated["invalidated"] is True
    assert invalidated["invalidation_close"] > invalidated["resistance"]


def test_new_breakout_attempt_clears_invalidation_from_an_older_failed_retest():
    from screener.structure import breakout_and_retest

    values = [_hour_candle(i, 9.5, 10, 9, 9.5) for i in range(14)]
    values += [
        _hour_candle(14, 11, 12, 10, 11),
        _hour_candle(15, 10, 11, 9, 10),
        _hour_candle(16, 9.5, 10.8, 8.5, 9.5),
        _hour_candle(17, 10.8, 11.5, 9.5, 10.8),
        _hour_candle(18, 12, 12.5, 11.5, 12.2),
        _hour_candle(19, 12.2, 12.3, 11.8, 12.1),
        _hour_candle(20, 12.8, 13, 12.1, 12.8),
        _hour_candle(21, 13, 13.1, 12.4, 13),  # Successful retest of 12.5.
        _hour_candle(22, 13.5, 13.8, 12.9, 13.5),
        _hour_candle(23, 12.5, 12.7, 12.0, 12.4),  # That retest is invalidated.
        _hour_candle(24, 12.5, 13.0, 12.6, 12.8),  # A fresh breakout attempt starts.
        _hour_candle(25, 13.2, 13.5, 13.0, 13.4),  # No retest yet.
    ]
    result = breakout_and_retest(values, "LONG", left=1, right=1, lookback=60, retest_lookback=8)
    assert result["confirmed"] is False  # Fresh attempt still needs its own retest.
    assert result["invalidated"] is False  # Prior attempt's invalidation is no longer current.
    assert result["breakout_open_time_ms"] == values[24].open_time_ms
