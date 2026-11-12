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


def test_short_breakdown_retest_uses_same_old_level_as_structure_break(monkeypatch):
    import screener.structure as structure

    values = [
        candle(0, 105, 103, 104),
        candle(1, 103, 101, 102),
        candle(2, 102, 100, 101),  # old confirmed support at 100
        candle(3, 99, 94, 95),     # close breaks 100
        candle(4, 97, 92, 94),
        candle(5, 95, 90, 91),     # later lower swing low
        candle(6, 96, 92, 94),
        candle(7, 99, 95, 97),
        candle(8, 98, 96, 97),     # newer pivot low at 96
        candle(9, 101, 97, 99),    # retest of old 100 level
        candle(10, 99, 96, 98),    # latest close remains below old level
    ]
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([], [(2, 100.0), (5, 90.0), (8, 96.0)]))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    retest = structure.breakout_and_retest(values, "SHORT", left=1, right=1, lookback=30, retest_lookback=8)
    paired_break = structure.bearish_structure_break(
        values, left=1, right=1, lookback=30, reference_level=retest.get("level")
    )
    invalidation = structure.scenario_invalidation("SHORT", "SHORT C — Long Squeeze", {
        "trend_retest_short": retest,
        "bearish_structure_break": paired_break,
    })

    assert retest["confirmed"] is True
    assert retest["level"] == 100.0
    assert retest["pivot_open_time_ms"] == values[2].open_time_ms
    assert paired_break["confirmed"] is True
    assert paired_break["level"] == 100.0
    assert paired_break["pivot_open_time_ms"] == retest["pivot_open_time_ms"]
    assert invalidation["level"] == 100.0


def test_breakout_retest_is_not_confirmed_if_latest_close_loses_the_level(monkeypatch):
    import screener.structure as structure

    values = [
        candle(0, 105, 103, 104),
        candle(1, 103, 101, 102),
        candle(2, 102, 100, 101),
        candle(3, 99, 94, 95),
        candle(4, 97, 92, 94),
        candle(5, 101, 97, 99),  # retest initially holds below 100
        candle(6, 102, 98, 101), # a later close reclaims the broken level
    ]
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([], [(2, 100.0)]))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = structure.breakout_and_retest(values, "SHORT", left=1, right=1, lookback=30, retest_lookback=8)

    assert result["confirmed"] is False
