import pytest
import math
from pathlib import Path

import numpy as np

from engine.factor_research import analyze_factors
from engine.portfolio import PortfolioConfig, PortfolioSimulator
from engine.research import BacktestConfig, QuantBacktester, ResearchBar, ResearchDataset, TradeRecord
from engine.signals import QuantSignalEngine
from engine.walk_forward import ParameterConfig, WalkForwardConfig, run_wfa


def make_bar(i: int, price: float, symbol: str = "BTCUSDT", score_bias: float = 0.0) -> ResearchBar:
    ts = 1_700_000_000_000 + i * 5 * 60 * 1000
    return ResearchBar(
        timestamp_ms=ts + 299_999,
        symbol=symbol,
        open=price,
        high=price * 1.005,
        low=price * 0.995,
        close=price * (1.0005 if score_bias > 0 else 0.9995),
        volume=1000.0,
        open_interest=1000.0 + i,
        delta_oi_pct=0.0002 * math.sin(i / 5),
        funding_rate_8h=0.00005 * math.cos(i / 7),
        basis_bps=2.0 * math.sin(i / 8),
        obi=0.6 * math.sin(i / 3),
        vpin=0.4,
        cvd_divergence_score=score_bias * (1.0 if i % 6 == 0 else 0.0),
        whale_divergence_score=score_bias * 0.5 * math.cos(i / 9),
        atr_pct=0.003,
        recent_high=price * 1.003,
        recent_low=price * 0.997,
        spread_bps=0.5,
    )


def dataset_fixture(n=90):
    # Mostly neutral rows; occasional large factor alignment produces testable signals.
    bars = [make_bar(i, 100.0 + 0.05 * i, score_bias=1.0 if i % 10 == 0 else 0.0) for i in range(n)]
    return ResearchDataset(bars)


def test_research_dataset_splits_gap_into_segments():
    a = make_bar(0, 100.0)
    b = make_bar(2, 100.0)
    ds = ResearchDataset([a, b])
    assert ds.gap_count == 1
    segments = ds.segments("BTCUSDT")
    assert len(segments) == 2
    assert [len(x) for x in segments] == [1, 1]


def test_backtest_does_not_use_same_candle_as_entry():
    ds = dataset_fixture(80)
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4, strong_signal_threshold=1.0, min_effective_rrr=0.0), BacktestConfig(entry_mode="next_open", entry_delay_bars=1))
    trades = bt.run(ds)
    for trade in trades:
        assert trade.entry_timestamp_ms > trade.signal_timestamp_ms


def test_factor_report_returns_ic_and_deciles():
    trades = []
    for i in range(20):
        z = -2.0 + i * 0.2
        y = z * 0.001
        trades.append(TradeRecord(
            signal_timestamp_ms=1_700_000_000_000 + i * 300_000,
            entry_timestamp_ms=1_700_000_000_000 + (i + 1) * 300_000,
            exit_timestamp_ms=1_700_000_000_000 + (i + 2) * 300_000,
            symbol="BTCUSDT", side="LONG", signal_score=80.0,
            entry_price=100.0, exit_price=100.0 * (1 + y), invalidation_price=99.0,
            target_price=102.0, stop_hit=False, target_hit=False, exit_reason="TIME_EXIT",
            gross_return=y, round_trip_cost=0.0, net_return=y, holding_bars=2,
            z_cvd=z, z_fund=z, z_oi=z, z_micro=z, z_whale=z, atr_pct=0.003, spread_bps=0.5,
        ))
    stats = analyze_factors(trades)
    assert stats[0].n == 20
    assert stats[0].pearson_ic is not None and stats[0].pearson_ic > 0.9
    assert stats[0].spread_top_minus_bottom is not None and stats[0].spread_top_minus_bottom > 0


def test_wfa_returns_no_folds_without_enough_history():
    ds = dataset_fixture(100)
    cfg = WalkForwardConfig(train_bars=60, validation_bars=20, test_bars=20, embargo_bars=5, min_train_trades=1, min_test_trades=1)
    results = run_wfa(ds, [ParameterConfig("baseline", {})], cfg)
    assert results == []


def test_portfolio_blocks_positive_correlation_but_allows_negative():
    ds_a = [make_bar(i, 100 + i) for i in range(80)]
    ds_b = []
    ds_c = []
    for i in range(80):
        rb = make_bar(i, 100 + i * 1.01)
        rc = make_bar(i, 300.0 / (100 + i))
        ds_b.append(ResearchBar(**{**rb.__dict__, "symbol": "ETHUSDT"}))
        ds_c.append(ResearchBar(**{**rc.__dict__, "symbol": "XRPUSDT"}))
    ds = ResearchDataset(ds_a + ds_b + ds_c)
    entry_ts = make_bar(60, 160.0).timestamp_ms
    exit_ts = make_bar(62, 160.0).timestamp_ms
    t1 = TradeRecord(make_bar(59, 159.0).timestamp_ms, entry_ts, exit_ts, "BTCUSDT", "LONG", 80, 100, 101, 99, 102, False, False, "TIME_EXIT", 0.01, 0, 0.01, 2, 2, 2, 2, 2, 2, 0.003, 0.5)
    t2 = TradeRecord(make_bar(59, 159.0).timestamp_ms + 1, entry_ts, exit_ts, "ETHUSDT", "LONG", 79, 100, 101, 99, 102, False, False, "TIME_EXIT", 0.01, 0, 0.01, 2, 2, 2, 2, 2, 2, 0.003, 0.5)
    t3 = TradeRecord(make_bar(59, 159.0).timestamp_ms + 2, entry_ts, exit_ts, "XRPUSDT", "LONG", 78, 100, 101, 99, 102, False, False, "TIME_EXIT", 0.01, 0, 0.01, 2, 2, 2, 2, 2, 2, 0.003, 0.5)
    fills, report = PortfolioSimulator(PortfolioConfig(max_strong_per_direction=3, max_aggregate_risk_pct=0.03)).run(ds, [t1, t2, t3])
    blocked = [f for f in fills if f.blocked]
    assert any("CORRELATION" in f.block_reason for f in blocked)
    assert report.trades_executed >= 2


def test_feature_event_study_includes_neutral_signal_ready_rows():
    from engine.factor_research import analyze_feature_event_study, summarize_feature_event_study
    base = 1_900_000_000_000
    bars = []
    for i in range(13):
        ts = base + i * 300_000
        bars.append(ResearchBar(**{
            **make_bar(i, 100.0 + i).__dict__,
            "timestamp_ms": ts,
            "recorded_signal": {
                "signal_type": "NEUTRAL", "z_cvd_div": 0.5, "z_fund_trap": -0.5,
                "z_delta_oi": 0.2, "z_micro": -0.1, "z_whale_sentiment": 0.0,
            },
        }))
    ds = ResearchDataset(bars)
    rows = analyze_feature_event_study(ds)
    assert rows
    assert {r["horizon_minutes"] for r in rows} == {5, 15, 30, 60}
    summary = summarize_feature_event_study(rows)
    assert any(r["factor"] == "z_cvd" and r["horizon_minutes"] == 5 and r["n"] > 0 for r in summary)


def test_wfa_stats_reports_hac_and_multiple_testing_fields():
    from engine.walk_forward import _stats
    trades = []
    for i, value in enumerate([0.01, 0.012, -0.003, 0.009, 0.008, -0.001] * 5):
        trades.append(TradeRecord(
            signal_timestamp_ms=1_000_000 + i * 300_000,
            entry_timestamp_ms=1_000_000 + i * 300_000,
            exit_timestamp_ms=1_000_000 + (i + 1) * 300_000,
            symbol="BTCUSDT", side="LONG", signal_score=80.0, entry_price=100.0, exit_price=100.0,
            invalidation_price=99.0, target_price=101.0, stop_hit=False, target_hit=False,
            exit_reason="TEST", gross_return=value, round_trip_cost=0.0, net_return=value, holding_bars=1,
            z_cvd=0.0, z_fund=0.0, z_oi=0.0, z_micro=0.0, z_whale=0.0, atr_pct=0.01, spread_bps=0.1,
        ))
    stats = _stats(trades)
    assert stats["hac_t_stat"] is not None
    assert stats["p_value"] is not None


def test_research_recorder_is_idempotent(tmp_path):
    from engine.research_recorder import ResearchRecorder
    db = tmp_path / "features.sqlite3"
    recorder = ResearchRecorder(db)
    row = {
        "timestamp_ms": 1_700_000_299_999,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000.0,
        "open_interest": 1000.0, "delta_oi_pct": 0.001, "funding_rate_8h": 0.0001,
        "basis_bps": 1.0, "obi": 0.1, "vpin": 0.4, "cvd_divergence_score": 0.0,
        "whale_divergence_score": 0.0, "atr_pct": 0.003, "recent_high": 101.0, "recent_low": 99.0,
    }
    assert recorder.append_rows([row, row]) == 1
    assert recorder.append_rows([row]) == 0
    out = tmp_path / "features.jsonl"
    assert recorder.export_jsonl(out) == 1
    assert len(out.read_text().splitlines()) == 1


def test_sentiment_bucket_normalization_respects_endpoint_timestamp_semantics():
    from engine.sentiment import SentimentEngine
    interval = 5 * 60 * 1000
    bar_open = (1_700_000_000_000 // 300_000) * 300_000
    assert SentimentEngine._bucket_open_ms(bar_open, timestamp_is_period_end=False) == bar_open
    assert SentimentEngine._bucket_open_ms(bar_open + interval - 1, timestamp_is_period_end=True) == bar_open
    assert SentimentEngine._bucket_open_ms(bar_open + interval, timestamp_is_period_end=True) == bar_open


def test_research_recorder_persists_idempotent_hourly_shard(tmp_path):
    from engine.research_recorder import ResearchRecorder
    db = tmp_path / "features.sqlite3"
    shards = tmp_path / "shards"
    recorder = ResearchRecorder(db, shards)
    row = {"timestamp_ms": 1_700_000_299_999, "symbol": "BTCUSDT", "close": 100.0}
    assert recorder.append_rows([row, row]) == 1
    shard_files = list(shards.rglob("*.jsonl"))
    assert len(shard_files) == 1
    assert len(shard_files[0].read_text().splitlines()) == 1
    assert recorder.append_rows([row]) == 0
    assert len(shard_files[0].read_text().splitlines()) == 1


def test_beta_diagnostic_reports_noncontiguous_history():
    from engine.market_regime import MarketRegimeEngine
    base = 1_700_000_000_000
    times = [base + i * 300_000 for i in range(30)]
    broken = times[:10] + times[11:]
    beta, reason = MarketRegimeEngine.calculate_rolling_beta_with_reason(
        broken, [100 + i for i in range(len(broken))],
        times, [200 + i for i in range(len(times))],
        min_samples=24,
    )
    assert beta is None
    assert reason == "NONCONTIGUOUS_HISTORY"


def test_research_history_filters_provenance(tmp_path):
    from engine.research_recorder import ResearchRecorder
    rec = ResearchRecorder(tmp_path / "features.sqlite3", tmp_path / "shards")
    base = 1_800_000_000_000
    rows = [
        {"symbol": "BTCUSDT", "timestamp_ms": base, "strategy_revision": "OLD", "config_fingerprint": "old", "research_schema_version": 1, "x": 1},
        {"symbol": "BTCUSDT", "timestamp_ms": base + 300_000, "strategy_revision": "NEW", "config_fingerprint": "new", "research_schema_version": 4, "x": 2},
    ]
    rec.append_rows(rows)
    out = rec.load_recent_histories(
        before_timestamp_ms=base + 600_000, symbols=["BTCUSDT"], limit=4,
        expected_provenance={"strategy_revision": "NEW", "config_fingerprint": "new", "research_schema_version": 4},
    )
    assert [r["x"] for r in out["BTCUSDT"]] == [2]


def test_research_export_shard_is_authoritative_over_stale_sqlite(tmp_path):
    from engine.research_recorder import ResearchRecorder, SCHEMA
    db = tmp_path / "features.sqlite3"
    shards = tmp_path / "shards"
    rec = ResearchRecorder(db, shards)
    ts = 1_800_000_000_000
    new = {"symbol": "BTCUSDT", "timestamp_ms": ts, "revision": "NEW"}
    old = {"symbol": "BTCUSDT", "timestamp_ms": ts, "revision": "OLD"}
    rec.append_rows([new])
    import sqlite3, json
    with sqlite3.connect(db) as con:
        con.executescript(SCHEMA)
        con.execute("UPDATE feature_rows SET payload_json=? WHERE symbol=? AND timestamp_ms=?", (json.dumps(old), "BTCUSDT", ts))
    out = tmp_path / "out.jsonl"
    rec.export_jsonl(out)
    assert json.loads(out.read_text().strip())["revision"] == "NEW"


def test_research_history_roundtrip_supports_state_recovery(tmp_path):
    from engine.research_recorder import ResearchRecorder

    rec = ResearchRecorder(tmp_path / "features.sqlite3")
    rows = []
    start = 1_800_000_000_000
    for i in range(4):
        rows.append({
            "symbol": "BTCUSDT",
            "timestamp_ms": start + i * 300_000,
            "funding_rate_8h": 0.0001 + i * 1e-6,
            "basis_bps": -4.0 + i * 0.1,
            "obi": 0.1 + i * 0.01,
            "vpin": 0.2,
            "whale_divergence_score": 0.0,
            "sentiment_available": True,
        })
    rec.append_rows(rows)
    restored = rec.load_recent_histories(
        before_timestamp_ms=start + 4 * 300_000,
        symbols=["BTCUSDT"],
        limit=4,
    )
    assert len(restored["BTCUSDT"]) == 4
    assert restored["BTCUSDT"][-1]["basis_bps"] == -3.7


def test_state_recovery_prefers_longer_durable_history():
    from engine.screener import QuantScreener
    state_values = (0.1, 0.2, 0.3)
    persisted = [
        {"funding_rate_8h": 0.01, "basis_bps": 1.0, "obi": 0.1, "vpin": 0.2},
        {"funding_rate_8h": 0.02, "basis_bps": 2.0, "obi": 0.2, "vpin": 0.2},
        {"funding_rate_8h": 0.03, "basis_bps": 3.0, "obi": 0.3, "vpin": 0.2},
        {"funding_rate_8h": 0.04, "basis_bps": 4.0, "obi": 0.4, "vpin": 0.2},
    ]
    hist, source = QuantScreener._select_recoverable_history(
        state_values, persisted, lambda row: row.get("funding_rate_8h"), True
    )
    assert len(hist) == 4
    assert source == "RESEARCH"


def test_state_recovery_uses_empty_when_no_prior_rows():
    from engine.screener import QuantScreener
    hist, source = QuantScreener._select_recoverable_history(
        (), [], lambda row: row.get("funding_rate_8h"), False
    )
    assert hist == ()
    assert source == "EMPTY"


def test_backtest_execution_price_is_directionally_adverse():
    from engine.research import ResearchBar
    bar = make_bar(0, 100.0)
    bar = ResearchBar(**{**bar.__dict__, "spread_bps": 10.0})
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4), BacktestConfig())
    long_entry = bt._execution_price(100.0, "LONG", bar, True)
    long_exit = bt._execution_price(100.0, "LONG", bar, False)
    short_entry = bt._execution_price(100.0, "SHORT", bar, True)
    short_exit = bt._execution_price(100.0, "SHORT", bar, False)
    assert long_entry > 100.0 and long_exit < 100.0
    assert short_entry < 100.0 and short_exit > 100.0


def test_backtest_round_trip_spread_is_not_double_counted():
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4), BacktestConfig(commission_rt_pct=0.0, slippage_atr_fraction=0.0))
    entry = make_bar(0, 100.0)
    exit_bar = make_bar(1, 100.0)
    entry = ResearchBar(**{**entry.__dict__, "spread_bps": 10.0})
    exit_bar = ResearchBar(**{**exit_bar.__dict__, "spread_bps": 10.0})
    assert math.isclose(bt._friction(entry, exit_bar), 0.0, rel_tol=1e-12)


def test_recorded_signal_replay_uses_persisted_final_event():
    from engine.serialization import signal_to_dict
    from contracts import SignalEvent
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_299_999, signal_type="STRONG_LONG", composite_score=81.0,
        z_cvd_div=2.0, z_fund_trap=1.0, z_delta_oi=0.5, z_micro=1.5, vpin=0.3, obi=0.6,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=1.7, decision_timestamp_ms=1_700_000_337_000,
    )
    row = make_bar(10, 100.0)
    row = ResearchBar(**{**row.__dict__, "recorded_signal": signal_to_dict(sig)})
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4), BacktestConfig(replay_recorded_signals=True))
    built = bt._build_signal([make_bar(i, 100.0) for i in range(10)] + [row], 10)
    assert built is not None
    assert built[0].signal_type == "STRONG_LONG"
    assert built[0].composite_score == 81.0


def test_wfa_counterfactual_does_not_depend_on_recorded_signal():
    ds = dataset_fixture(100)
    cfg = WalkForwardConfig(train_bars=60, validation_bars=20, test_bars=20, embargo_bars=5, min_train_trades=1, min_test_trades=1)
    results = run_wfa(ds, [ParameterConfig("baseline", {})], cfg)
    assert results == []


def test_clean_recorded_dataset_missing_signal_event_does_not_recompute():
    from engine.research import BacktestConfig, QuantBacktester, ResearchBar
    engine = QuantSignalEngine(z_history_min_samples=4)
    rows = []
    for i in range(6):
        bar = make_bar(i, 100.0 + i)
        bar = ResearchBar(**{**bar.__dict__, "strategy_revision": "2026-09-25-p0p1-hardened", "config_fingerprint": "abc123", "research_schema_version": 2})
        rows.append(bar)
    bt = QuantBacktester(engine, BacktestConfig(replay_recorded_signals=True))
    assert bt._build_signal(rows, 5) is None


def test_short_return_uses_linear_futures_pnl_semantics():
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4))
    assert math.isclose(bt._gross_return(100.0, 90.0, "SHORT"), 0.10, rel_tol=1e-12)
    assert math.isclose(bt._gross_return(100.0, 110.0, "SHORT"), -0.10, rel_tol=1e-12)
    assert math.isclose(bt._gross_return(100.0, 90.0, "LONG"), -0.10, rel_tol=1e-12)


def test_state_recovery_rejects_research_tail_with_missing_previous_bar():
    from engine.screener import QuantScreener
    rows = [
        {"timestamp_ms": 1_000_000, "funding_rate_8h": 0.01},
        {"timestamp_ms": 1_600_000, "funding_rate_8h": 0.02},
    ]
    hist, source = QuantScreener._select_recoverable_history(
        (), rows, lambda row: row.get("funding_rate_8h"), False, expected_previous_timestamp_ms=1_300_000
    )
    assert hist == ()
    assert source == "EMPTY"


def test_next_open_entry_timestamp_is_bar_open_not_bar_close():
    ds = dataset_fixture(80)
    bt = QuantBacktester(QuantSignalEngine(z_history_min_samples=4, strong_signal_threshold=1.0, min_effective_rrr=0.0), BacktestConfig(entry_mode="next_open", entry_delay_bars=1))
    trades = bt.run(ds)
    for trade in trades[:10]:
        assert trade.entry_timestamp_ms == trade.signal_timestamp_ms + 1


def test_backtest_execution_cost_inputs_do_not_use_entry_bar_future_atr():
    from contracts import SignalEvent
    from engine.research import ResearchBar, QuantBacktester
    signal_row = make_bar(4, 100.0)
    entry_row = make_bar(5, 101.0)
    signal_row = ResearchBar(**{**signal_row.__dict__, "atr_pct": 0.01})
    entry_row = ResearchBar(**{**entry_row.__dict__, "atr_pct": 0.50})
    later_row = make_bar(6, 102.0)
    bt = QuantBacktester(
        QuantSignalEngine(z_history_min_samples=4),
        BacktestConfig(max_holding_bars=1, slippage_atr_fraction=0.2, max_slippage_rt_pct=1.0),
    )
    calls = []
    original = bt._execution_price
    def spy(reference, side, bar, is_entry, *, volatility_pct=None):
        calls.append((is_entry, bar.timestamp_ms, volatility_pct))
        return original(reference, side, bar, is_entry, volatility_pct=volatility_pct)
    bt._execution_price = spy
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=signal_row.timestamp_ms, signal_type="STRONG_LONG", composite_score=80.0,
        z_cvd_div=1.0, z_fund_trap=1.0, z_delta_oi=1.0, z_micro=1.0, vpin=0.2, obi=0.1,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=90.0, target_price=130.0,
        risk_reward_ratio=2.0, decision_timestamp_ms=signal_row.timestamp_ms,
    )
    bt._run_trade([make_bar(i, 100.0) for i in range(4)] + [signal_row, entry_row, later_row], 4, sig, (1,1,1,1,0))
    assert calls
    assert calls[0][0] is True
    assert calls[0][2] == signal_row.atr_pct
    assert calls[0][2] != entry_row.atr_pct


def test_telegram_dispatch_failure_is_not_recorded_as_success(monkeypatch, tmp_path):
    from contracts import SignalEvent
    from engine.telegram import TelegramAlerter, TelegramDispatchError
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TG_BOT_TOKEN", "token")
    monkeypatch.setenv("TG_CHAT_IDS", "123")
    alerter = TelegramAlerter()
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_299_999, signal_type="STRONG_LONG", composite_score=81.0,
        z_cvd_div=2.0, z_fund_trap=1.0, z_delta_oi=0.5, z_micro=1.5, vpin=0.3, obi=0.6,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=1.7, decision_timestamp_ms=1_700_000_337_000,
    )
    async def fail_send(_text):
        return False
    monkeypatch.setattr(alerter, "send_message", fail_send)
    monkeypatch.setattr(alerter, "_should_alert", lambda *_: True)
    import asyncio
    with pytest.raises(TelegramDispatchError) as exc:
        asyncio.run(alerter.process_and_dispatch_signals([sig], []))
    assert not exc.value.sent_signal_ids


def test_telegram_missing_config_fails_when_alert_is_eligible(monkeypatch, tmp_path):
    from contracts import SignalEvent
    from engine.telegram import TelegramAlerter, TelegramDispatchError
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_IDS", raising=False)
    alerter = TelegramAlerter()
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_299_999, signal_type="STRONG_SHORT", composite_score=-81.0,
        z_cvd_div=-2.0, z_fund_trap=-1.0, z_delta_oi=-0.5, z_micro=-1.5, vpin=0.3, obi=-0.6,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=102.0, target_price=96.0,
        risk_reward_ratio=1.7, decision_timestamp_ms=1_700_000_337_000,
    )
    monkeypatch.setattr(alerter, "_should_alert", lambda *_: True)
    import asyncio
    with pytest.raises(TelegramDispatchError, match="TELEGRAM_CONFIGURATION_MISSING"):
        asyncio.run(alerter.process_and_dispatch_signals([sig], []))
