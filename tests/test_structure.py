from screener.models import Candle
from screener.structure import confirmed_pivots, failed_breakout_rejection, higher_low


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
        "higher_low": {"latest_low": 98.5},
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



def test_failed_breakout_rejection_returns_the_confirmation_candle_anchor():
    values = [
        candle(0, 9, 7, 8),
        candle(1, 10, 8, 9),
        candle(2, 12, 9, 11),
        candle(3, 11, 8, 9),
        candle(4, 11.5, 8.5, 10),
        candle(5, 12.5, 9, 11.5),
    ]
    result = failed_breakout_rejection(values, left=1, right=1, lookback=30)

    assert result["confirmed"] is True
    assert result["failure_open_time_ms"] == values[-1].open_time_ms
