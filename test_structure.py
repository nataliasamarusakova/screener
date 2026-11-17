from screener.models import Candle
from screener.structure import confirmed_pivots, failed_breakout_rejection, higher_low, weak_bounce_rejection


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
    assert result["confirmation_open_time_ms"] > result["latest_open_time_ms"]
    assert result["confirmation_close_time_ms"] == result["confirmation_open_time_ms"] + 59_999


def test_higher_low_is_invalidated_by_any_later_close_below_latest_pivot(monkeypatch):
    import screener.structure as structure

    values = [
        candle(0, 91, 89, 90), candle(1, 94, 91, 92), candle(2, 96, 95, 95.5),
        candle(3, 86, 84, 85), candle(4, 98, 95, 96),
    ]
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([], [(0, 90.0), (2, 95.0)]))

    result = higher_low(values, left=1, right=1)

    assert result["confirmed"] is False
    assert "invalidated" in result["reason"].lower()


def test_weak_bounce_rejection_requires_the_latest_closed_candle(monkeypatch):
    import screener.structure as structure

    values = [
        candle(0, 115, 110, 112),
        candle(1, 108, 104, 104.5),  # old pin/rejection at resistance
        candle(2, 96, 92, 93),       # price moves away from resistance
        candle(3, 85, 80, 81),       # later close reaches a fresh low
    ]
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(0, 115.0), (1, 108.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = weak_bounce_rejection(values, left=1, right=1)

    assert result["confirmed"] is False


def test_weak_bounce_rejection_accepts_fresh_bearish_sweep_below_resistance(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 115, 100, 107) for i in range(24)]
    values[-1] = Candle(
        open_time_ms=23 * 60_000, close_time_ms=23 * 60_000 + 59_999,
        open=108.5, high=110.0, low=105.0, close=107.0,
        volume=1, quote_volume=1, trades=10,
    )
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(0, 115.0), (10, 108.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = weak_bounce_rejection(values, left=1, right=1)

    assert result["confirmed"] is True
    assert result["rejection_open_time_ms"] == values[-1].open_time_ms


def test_weak_bounce_rejection_requires_close_below_resistance(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 115, 110, 112) for i in range(20)]
    # Bearish candle with an upper wick, but its close remains above resistance.
    values[-1] = Candle(
        open_time_ms=19 * 60_000, close_time_ms=19 * 60_000 + 59_999,
        open=109.0, high=110.0, low=107.0, close=108.1,
        volume=1, quote_volume=1, trades=10,
    )
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(0, 115.0), (10, 108.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = weak_bounce_rejection(values, left=1, right=1)

    assert result["confirmed"] is False


def test_confirmed_pivots_rejects_zero_sided_windows():
    values = [candle(i, 11 + i, 9 + i, 10 + i) for i in range(6)]
    for left, right in ((0, 1), (1, 0), (0, 0)):
        try:
            confirmed_pivots(values, left=left, right=right)
        except ValueError as exc:
            assert "must both be >= 1" in str(exc)
        else:
            raise AssertionError("Zero-width pivot windows must be rejected explicitly")


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


def test_failed_breakout_rejection_does_not_survive_after_price_moves_away(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 98, 96, 97) for i in range(12)]
    values[7] = candle(7, 105, 99, 99.5)  # old sweep above resistance 100
    values[-1] = candle(11, 99, 95, 96)   # latest candle no longer tests resistance
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(2, 100.0)], []))

    result = failed_breakout_rejection(values, left=1, right=1, lookback=30)

    assert result["confirmed"] is False


def test_breakout_retest_expires_when_first_retest_is_too_late(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 106, 104, 105) for i in range(14)]
    values[2] = candle(2, 100, 98, 99)
    values[3] = candle(3, 107, 101, 106)  # breakout above 100
    values[12] = candle(12, 101, 99.5, 100.5)  # first retest, nine bars later
    values[13] = candle(13, 102, 100, 101)
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(2, 100.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = structure.breakout_and_retest(values, "LONG", left=1, right=1, lookback=30, retest_lookback=8)

    assert result["confirmed"] is False


def test_old_breakout_level_cannot_confirm_long_after_late_deep_pullback(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 106, 104, 105) for i in range(59)]
    values[10] = candle(10, 100, 98, 99)
    values[11] = candle(11, 107, 101, 106)  # old H1=100 breakout
    values[35] = candle(35, 130, 124, 125)  # newer H2=130, never broken
    values[-1] = candle(58, 101, 99.8, 100.5)  # deep pullback to H1
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(10, 100.0), (35, 130.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = structure.breakout_and_retest(values, "LONG", left=1, right=1, lookback=60, retest_lookback=8)

    assert result["confirmed"] is False


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


def test_higher_low_rejects_an_unbroken_but_stale_pivot(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 110, 100, 105) for i in range(60)]
    monkeypatch.setattr(
        structure, "confirmed_pivots",
        lambda *_args, **_kwargs: ([], [(5, 90.0), (30, 95.0)]),
    )

    result = higher_low(values, left=2, right=2, lookback=60, max_age_bars=24)

    assert result["confirmed"] is False
    assert result["age_bars"] == 29
    assert "stale" in result["reason"].lower()


def test_higher_low_accepts_a_recent_unbroken_pivot(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 110, 100, 105) for i in range(60)]
    monkeypatch.setattr(
        structure, "confirmed_pivots",
        lambda *_args, **_kwargs: ([], [(30, 90.0), (50, 95.0)]),
    )

    result = higher_low(values, left=2, right=2, lookback=60, max_age_bars=24)

    assert result["confirmed"] is True
    assert result["age_bars"] == 9


def test_higher_low_rejects_invalid_max_age_configuration():
    values = [candle(i, 110, 100, 105) for i in range(10)]
    for invalid in (0, -1, True, 1.5):
        try:
            higher_low(values, max_age_bars=invalid)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"Invalid max_age_bars={invalid!r} must be rejected")


def test_weak_bounce_rejection_is_blocked_on_a_new_24h_low(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 115, 100, 112) for i in range(24)]
    values[-1] = Candle(
        open_time_ms=23 * 60_000, close_time_ms=23 * 60_000 + 59_999,
        open=108.5, high=110.0, low=99.0, close=100.0,
        volume=1, quote_volume=1, trades=10,
    )
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(0, 115.0), (10, 108.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = weak_bounce_rejection(values, left=1, right=1)

    assert result["confirmed"] is False
    assert "new 24H low" in result["reason"]



def test_weak_bounce_rejection_rejects_resistance_broken_by_intervening_close(monkeypatch):
    import screener.structure as structure

    values = [candle(i, 115, 100, 107) for i in range(24)]
    values[15] = candle(15, 116, 107, 109)  # Closed above the selected 108 resistance.
    values[-1] = Candle(
        open_time_ms=23 * 60_000, close_time_ms=23 * 60_000 + 59_999,
        open=108.5, high=110.0, low=105.0, close=107.0,
        volume=1, quote_volume=1, trades=10,
    )
    monkeypatch.setattr(structure, "confirmed_pivots", lambda *_args, **_kwargs: ([(0, 115.0), (10, 108.0)], []))
    monkeypatch.setattr(structure, "atr", lambda *_args, **_kwargs: 2.0)

    result = weak_bounce_rejection(values, left=1, right=1)

    assert result["confirmed"] is False
    assert "invalidated" in result["reason"].lower()
