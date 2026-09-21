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


def test_research_dataset_rejects_gap():
    a = make_bar(0, 100.0)
    b = make_bar(2, 100.0)
    try:
        ResearchDataset([a, b])
    except ValueError as exc:
        assert "Gap" in str(exc)
    else:
        raise AssertionError("gap was not rejected")


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
