"""Regression tests for the production quant/risk paths.

The tests intentionally assert concrete values and failure modes rather than
implementation ranges. They cover the Phase-1 root causes fixed in Phase 2.
"""
import asyncio
import json
import math

import numpy as np
import polars as pl
import pytest

from binance_ingestion import BinanceFuturesIngestion, BinanceOrderBookFSM, OrderBookFSMState
from contracts import MarketStateSnapshot
from engine.backtester import QuantBacktester, compute_deflated_sharpe_ratio
from engine.divergence import detect_cvd_divergence_jit
from engine.execution_cost import CEXCostModel
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.mcp_server import MCPServer
from engine.microstructure_jit import compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.risk import DynamicRiskEngine
from engine.sentiment import SentimentEngine
from engine.signals import QuantSignalEngine, empirical_zscore


def test_orderbook_fsm_sync_and_gap_detection():
    fsm = BinanceOrderBookFSM("BTCUSDT")
    fsm.on_ws_connected()
    assert fsm.state == OrderBookFSMState.BUFFERING

    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 100, "s": "BTCUSDT", "U": 90, "u": 99, "pu": 89,
        "b": [["60000", "1.0"]], "a": [["60001", "1.0"]]
    })
    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 101, "s": "BTCUSDT", "U": 100, "u": 105, "pu": 99,
        "b": [["60000", "2.0"]], "a": [["60001", "1.5"]]
    })
    applied = fsm.apply_snapshot({
        "lastUpdateId": 100,
        "bids": [["60000", "1.0"], ["59990", "10.0"]],
        "asks": [["60001", "1.0"], ["60010", "10.0"]],
    })
    assert applied is True
    assert fsm.state == OrderBookFSMState.IN_SYNC
    snap = fsm.get_snapshot(depth=5)
    assert snap is not None
    assert snap.best_bid == 60000.0
    assert snap.best_ask == 60001.0
    assert snap.mid_price == 60000.5

    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 102, "s": "BTCUSDT", "U": 121, "u": 125, "pu": 120,
        "b": [], "a": []
    })
    assert fsm.state == OrderBookFSMState.DISCONNECTED


def test_orderbook_fsm_requires_last_update_plus_one_in_first_event():
    fsm = BinanceOrderBookFSM("BTCUSDT")
    fsm.on_ws_connected()
    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 100, "s": "BTCUSDT", "U": 90, "u": 100, "pu": 89,
        "b": [], "a": [],
    })
    assert fsm.apply_snapshot({
        "lastUpdateId": 100,
        "bids": [["60000", "1.0"]],
        "asks": [["60001", "1.0"]],
    }) is False
    assert fsm.state == OrderBookFSMState.BUFFERING


def test_weighted_obi_exact_value_and_symmetry():
    bids = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    asks = np.array([2.0, 1.0, 1.0], dtype=np.float64)
    obi = compute_weighted_obi_jit(bids, asks, decay=0.85)
    expected = (15.695 - 3.5725) / (15.695 + 3.5725)
    assert obi == pytest.approx(expected, abs=1e-12)
    assert compute_weighted_obi_jit(bids, bids, decay=0.85) == pytest.approx(0.0, abs=1e-12)


def test_vpin_batch_output_is_deterministic():
    qtys = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 1.0, 5.0], dtype=np.float64)
    is_buy = np.array([True, False, True, True, False, False, True, False], dtype=np.bool_)
    first = compute_vpin_numba(qtys, is_buy, bucket_vol=5.0, window_baskets=5)
    second = compute_vpin_numba(qtys, is_buy, bucket_vol=5.0, window_baskets=5)
    assert first == pytest.approx(second, abs=1e-12)
    assert first == pytest.approx(0.45, abs=1e-12)


def test_synthetic_liquidation_detection_concrete_ratio():
    detector = SyntheticLiquidationDetector(anomaly_ratio_threshold=1.5, volatility_shock_pct=0.3)
    liq = detector.detect(
        symbol="BTCUSDT",
        current_price=60000.0,
        price_change_pct=-0.5,
        delta_oi=-1000.0,
        taker_buy_vol=50.0,
        taker_sell_vol=200.0,
    )
    assert liq is not None
    assert liq.side == "LONG_LIQUIDATION"
    assert liq.anomaly_ratio == pytest.approx(4.0, abs=1e-12)
    assert liq.estimated_liquidation_volume == pytest.approx(800.0, abs=1e-12)


def test_cvd_divergence_direction_is_preserved():
    prices = np.array([100.0, 98.0, 97.0, 98.5, 96.0], dtype=np.float64)
    cvd = np.array([-50.0, -100.0, -150.0, -80.0, -40.0], dtype=np.float64)
    score, div_type = detect_cvd_divergence_jit(prices, cvd, lookback=4)
    assert div_type == 1
    assert score == pytest.approx(1.0, abs=1e-12)


def test_empirical_zscore_is_true_point_in_time_sample_zscore():
    history = [1.0, 2.0, 3.0, 4.0]
    z = empirical_zscore(5.0, history, min_samples=4)
    expected = (5.0 - 2.5) / math.sqrt(5.0 / 3.0)
    assert z == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match="Insufficient history"):
        empirical_zscore(5.0, [1.0, 2.0, 3.0], min_samples=4)
    with pytest.raises(ValueError, match="zero variance"):
        empirical_zscore(2.0, [1.0, 1.0, 1.0, 1.0], min_samples=4)


def test_explicit_5m_agg_trade_window_never_falls_back_to_last_n_trades():
    class StubIngestion(BinanceFuturesIngestion):
        async def _request_json(self, method, url, *, params=None, weight, symbol=""):
            assert params["startTime"] == "1700000000000"
            assert params["endTime"] == "1700000299999"
            assert params["limit"] == "1000"
            assert weight == 20
            return [{"T": 1700000100000, "q": "2.0", "m": True}]

    trades = asyncio.run(StubIngestion([]).fetch_5m_agg_trades(
        "BTCUSDT", 1700000000000, 1700000299999
    ))
    assert trades is not None
    assert len(trades) == 1
    assert trades[0]["T"] == 1700000100000


def test_deprecated_last_n_trade_cvd_helper_fails_loudly():
    ingestion = BinanceFuturesIngestion([])
    with pytest.raises(RuntimeError, match="deprecated"):
        asyncio.run(ingestion.fetch_recent_agg_trades_cvd("BTCUSDT"))


def test_5m_sweep_direction_is_preserved_into_signal_booster():
    event = LiquiditySweepDetector().detect(
        symbol="BTCUSDT", current_price=101.0, current_high=102.0, current_low=99.0,
        recent_swing_high=105.0, recent_swing_low=100.0, cvd_delta=500.0,
    )
    assert event is not None
    assert event.pattern_type == "BULLISH_SWEEP_RECLAIM"


def test_signal_requires_empirical_zscores_and_preserves_sweep_direction():
    engine = QuantSignalEngine(z_history_min_samples=4)
    common = dict(
        symbol="BTCUSDT",
        current_price=100.0,
        funding_rate_8h=0.0,
        basis_spread_bps=0.0,
        delta_oi=100.0,
        oi_total=1000.0,
        obi=0.8,
        vpin=0.2,
        cvd_divergence_score=1.0,
        recent_high=101.0,
        recent_low=99.0,
        z_cvd_override=3.0,
        z_fund_override=3.0,
        z_delta_oi_override=3.0,
        z_micro_override=3.0,
        z_whale_override=3.0,
        atr_pct=0.01,
        sweep_reclaim=True,
        sweep_pattern="BULLISH_SWEEP_RECLAIM",
        timestamp_ms=1700000000000,
    )
    sig = engine.compute_signal(**common)
    expected_score = 100.0 * math.tanh((25 * 3 + 25 * 3 + 15 * 3 + 15 * 3 + 20 * 3 + 25) / (100 * 1.4))
    assert sig.signal_type == "STRONG_LONG"
    assert sig.composite_score == pytest.approx(round(expected_score, 2), abs=1e-12)
    assert sig.sweep_pattern == "BULLISH_SWEEP_RECLAIM"
    assert sig.invalidation_price < 100.0 < sig.target_price

    with pytest.raises(ValueError, match="All empirical factor Z-score overrides are required"):
        engine.compute_signal(**{k: v for k, v in common.items() if k != "z_whale_override"})

    with pytest.raises(ValueError, match="explicit sweep_pattern"):
        engine.compute_signal(**{**common, "sweep_pattern": "NONE"})


def test_dynamic_risk_rejects_inconsistent_levels_and_zero_capital():
    engine = DynamicRiskEngine(default_account_capital=10000.0, risk_per_trade_pct=1.0, max_leverage=10.0)
    rec = engine.calculate_sizing(
        symbol="BTCUSDT",
        signal_type="STRONG_LONG",
        entry_price=100.0,
        invalidation_price=95.0,
        target_price=110.0,
    )
    assert rec is not None
    assert rec.recommended_quantity == pytest.approx(20.0, abs=1e-12)
    assert rec.recommended_notional_usdt == pytest.approx(2000.0, abs=1e-12)
    assert rec.invalidation_price == 95.0
    assert rec.target_price == 110.0
    assert rec.risk_reward_ratio == pytest.approx(2.0, abs=1e-12)

    assert engine.calculate_sizing(
        symbol="BTCUSDT", signal_type="STRONG_LONG", entry_price=100.0,
        invalidation_price=101.0, target_price=110.0,
    ) is None
    assert engine.calculate_sizing(
        symbol="BTCUSDT", signal_type="STRONG_LONG", entry_price=100.0,
        invalidation_price=95.0, target_price=90.0,
    ) is None
    assert engine.calculate_sizing(
        symbol="BTCUSDT", signal_type="STRONG_LONG", entry_price=100.0,
        invalidation_price=95.0, target_price=110.0, account_capital=0.0,
    ) is None


def test_liquidity_constraint_is_real_and_numeric():
    engine = DynamicRiskEngine(default_account_capital=10000.0, risk_per_trade_pct=1.0, max_leverage=10.0, max_liquidity_impact_pct=1.5)
    rec = engine.calculate_sizing(
        symbol="BTCUSDT",
        signal_type="STRONG_LONG",
        entry_price=100.0,
        invalidation_price=90.0,
        target_price=120.0,
        available_liquidity_usdt=5000.0,
    )
    assert rec is not None
    assert rec.liquidity_constraint_applied is True
    assert rec.recommended_notional_usdt == pytest.approx(75.0, abs=1e-12)
    assert rec.recommended_quantity == pytest.approx(0.75, abs=1e-12)


def test_empirical_zscore_rejects_nonfinite_history_instead_of_shrinking_sample():
    history = [1.0, 2.0, float("nan"), 4.0]
    with pytest.raises(ValueError, match="history contains non-finite"):
        empirical_zscore(3.0, history, min_samples=3)


def test_funding_gate_rejects_nonfinite_funding_rate():
    result = FundingFilterEngine().evaluate_funding_gate(
        symbol="BTCUSDT",
        signal_type="STRONG_LONG",
        funding_rate_8h=float("nan"),
        next_funding_time_ms=1_700_000_300_000,
        current_time_ms=1_700_000_000_000,
    )
    assert result.allow_long is False
    assert result.allow_short is False
    assert result.gate_reason == "INVALID_FUNDING_RATE"


def test_funding_unknown_time_is_fail_closed():
    result = FundingFilterEngine(proximity_threshold_minutes=20.0).evaluate_funding_gate(
        symbol="BTCUSDT",
        signal_type="STRONG_LONG",
        funding_rate_8h=0.001,
        next_funding_time_ms=0,
        current_time_ms=1700000000000,
    )
    assert result.allow_long is False
    assert result.allow_short is False
    assert result.gate_reason == "UNKNOWN_FUNDING_TIME"


def test_quality_gate_rejects_nan_before_threshold_logic():
    result = QualityFilter().evaluate(
        symbol="BTCUSDT",
        quote_volume_24h=float("nan"),
        spread_bps=1.0,
        funding_rate_8h=0.0,
    )
    assert result.is_valid is False
    assert result.rejection_reason == "NON_FINITE_INPUT"


def test_rolling_beta_is_timestamp_aligned_and_requires_contiguous_5m_data():
    step = 5 * 60 * 1000
    times = [1700000000000 + i * step for i in range(30)]
    btc = [60000.0]
    alt = [3000.0]
    returns = [0.0005, -0.0002, 0.0008, 0.0001, -0.0004] * 6
    for r in returns:
        btc.append(btc[-1] * (1.0 + r))
        alt.append(alt[-1] * (1.0 + 2.0 * r))
    beta = MarketRegimeEngine.calculate_rolling_beta(times, alt, times, btc, min_samples=24)
    assert beta == pytest.approx(2.0, rel=1e-3)

    gapped_times = times.copy()
    gapped_times[15] += step
    gapped_beta = MarketRegimeEngine.calculate_rolling_beta(gapped_times, alt, times, btc, min_samples=24)
    assert gapped_beta == pytest.approx(2.0, rel=1e-3)


def test_backtest_uses_next_open_and_includes_both_entry_and_exit_fees():
    df = pl.DataFrame({
        "open": [100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "timestamp_ms": [i * 300000 for i in range(8)],
        "composite_score": [80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    backtester = QuantBacktester(market_spread_bps=1.5)
    metrics = backtester.run_backtest(df, holding_bars=2)
    half_spread = 1.5 / 2.0
    adverse = half_spread * 0.4 * 1.2
    slippage = half_spread + adverse
    entry_exec = 102.0 * (1.0 + slippage / 10000.0)
    exit_exec = 103.0 * (1.0 - slippage / 10000.0)
    expected = (exit_exec * (1.0 - 0.0005) - entry_exec * (1.0 + 0.0005)) / entry_exec
    assert metrics.total_trades == 1
    assert metrics.total_net_pnl_pct == pytest.approx(round(expected * 100.0, 2), abs=1e-12)


def test_backtest_bar_returns_match_exact_execution_return():
    df = pl.DataFrame({
        "open": [100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "timestamp_ms": [i * 300000 for i in range(8)],
        "composite_score": [80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    backtester = QuantBacktester(market_spread_bps=1.5)
    metrics = backtester.run_backtest(df, holding_bars=2)
    half_spread = 1.5 / 2.0
    slippage = half_spread + half_spread * 0.4 * 1.2
    entry_exec = 102.0 * (1.0 + slippage / 10000.0)
    exit_exec = 103.0 * (1.0 - slippage / 10000.0)
    expected = (exit_exec * (1.0 - 0.0005) - entry_exec * (1.0 + 0.0005)) / entry_exec
    assert metrics.total_net_pnl_pct == pytest.approx(round(expected * 100.0, 2), abs=1e-12)


def test_liquidation_detector_rejects_zero_visible_taker_volume():
    result = SyntheticLiquidationDetector().detect(
        symbol="BTCUSDT",
        current_price=60000.0,
        price_change_pct=-1.0,
        delta_oi=-1000.0,
        taker_buy_vol=0.0,
        taker_sell_vol=0.0,
    )
    assert result is None


def test_execution_cost_rejects_invalid_reference_and_zero_spread_is_exact():
    model = CEXCostModel()
    with pytest.raises(ValueError, match="reference_price"):
        model.simulate_execution("BTCUSDT", "BUY", float("nan"), 1.0)
    result = model.simulate_execution("BTCUSDT", "BUY", 100.0, 0.0)
    assert result.half_spread_bps == pytest.approx(0.0, abs=1e-12)
    assert result.executed_price == pytest.approx(100.0, abs=1e-12)


def test_signal_rejects_nonfinite_z_override():
    engine = QuantSignalEngine()
    with pytest.raises(ValueError, match="Non-finite signal input"):
        engine.compute_signal(
            symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
            delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=0.0,
            recent_high=101.0, recent_low=99.0, z_cvd_override=float("nan"), z_fund_override=0.0,
            z_delta_oi_override=0.0, z_micro_override=0.0, z_whale_override=0.0, atr_pct=1.0,
        )


def test_mcp_notification_has_no_jsonrpc_response():
    server = MCPServer()
    response = server.handle_request(json.dumps({"jsonrpc": "2.0", "method": "tools/list", "params": {}}))
    assert response == ""


def test_backtest_refuses_close_only_input_instead_of_substituting_it_for_open():
    df = pl.DataFrame({
        "close": [100.0, 102.0, 104.0, 106.0, 108.0],
        "timestamp_ms": [i * 300000 for i in range(5)],
        "composite_score": [80.0, 0.0, 0.0, 0.0, 0.0],
    })
    metrics = QuantBacktester().run_backtest(df, holding_bars=1)
    assert metrics.total_trades == 0
    assert metrics.total_net_pnl_pct == 0.0


def test_dsr_requires_actual_trial_metadata_and_uses_raw_kurtosis():
    returns = np.array([-0.02, -0.01, -0.005, 0.0, 0.004, 0.01, 0.018, 0.024], dtype=np.float64)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    observed_sr = mean_r / std_r
    diffs = returns - mean_r
    expected_raw_kurtosis = float(np.mean(diffs ** 4) / (std_r ** 4))

    dsr0, _, kurt0 = compute_deflated_sharpe_ratio(observed_sr, returns)
    assert dsr0 == 0.0
    assert kurt0 == pytest.approx(expected_raw_kurtosis, abs=1e-12)

    dsr1, _, kurt1 = compute_deflated_sharpe_ratio(observed_sr, returns, n_trials=1)
    assert kurt1 == pytest.approx(expected_raw_kurtosis, abs=1e-12)
    assert 0.0 <= dsr1 <= 1.0

    with pytest.raises(ValueError, match="var_trials_sr is required"):
        compute_deflated_sharpe_ratio(observed_sr, returns, n_trials=20)


def test_sentiment_fails_closed_without_api_key():
    result = asyncio.run(SentimentEngine(api_key=None).fetch_sentiment_divergence("BTCUSDT"))
    assert result is None


def test_sentiment_observation_selector_is_bounded_by_pit_window():
    start = 1_700_000_000_000
    end = start + 300_000
    data = [
        {"timestamp": start - 1, "longShortRatio": "1.0"},
        {"timestamp": start + 60_000, "longShortRatio": "1.1"},
        {"timestamp": end + 1, "longShortRatio": "9.9"},
    ]
    selected = SentimentEngine._select_observation(data, start, end)
    assert selected is not None
    assert selected["longShortRatio"] == "1.1"


def test_mcp_rejects_unknown_kwargs_and_requires_real_backtest_metadata():
    server = MCPServer()
    listed = json.loads(server.handle_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})))
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert "available_liquidity_usdt" in tools["calculate_position_size"]["inputSchema"]["required"]
    assert "data_path" in tools["run_strategy_backtest"]["inputSchema"]["required"]
    assert "n_trials" in tools["run_strategy_backtest"]["inputSchema"]["required"]

    bad_kwargs = json.loads(server.handle_request(json.dumps({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "calculate_position_size", "arguments": {"foo": 1}},
    })))
    assert bad_kwargs["error"]["code"] == -32602

    missing_path = json.loads(server.handle_request(json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "run_strategy_backtest", "arguments": {}},
    })))
    assert missing_path["error"]["code"] == -32602

    missing_liquidity = json.loads(server.handle_request(json.dumps({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {
            "name": "calculate_position_size",
            "arguments": {
                "symbol": "BTCUSDT", "signal_type": "STRONG_LONG",
                "entry_price": 100.0, "invalidation_price": 95.0, "target_price": 110.0,
                "account_capital_usdt": 10000.0, "risk_pct": 1.0,
            },
        },
    })))
    assert missing_liquidity["error"]["code"] == -32602


def test_state_contract_backward_compatible_with_old_payload():
    old_payload = {
        "symbol": "BTCUSDT",
        "timestamp_ms": 1700000000000,
        "last_price": 60000.0,
        "open_interest": 1000.0,
        "delta_oi_5m": 0.0,
        "cumulative_cvd_5m": 0.0,
        "funding_rate_8h": 0.0,
        "basis_bps": 0.0,
        "vpin_estimate": 0.1,
        "obi_score": 0.2,
        "composite_score": 0.0,
        "low_24h": 59000.0,
        "high_24h": 61000.0,
        "whale_sentiment_z": 0.0,
        "cvd_history_5m": (),
        "price_history_5m": (),
    }
    raw = json.dumps(old_payload).encode()
    decoded = __import__("msgspec").json.decode(raw, type=MarketStateSnapshot)
    assert decoded.signal_ready is False
    assert decoded.candle_open_time_ms == 0


async def _fake_screener_scan_dependencies():
    return None


def test_backtest_rejects_non_5m_timestamps_instead_of_misannualizing():
    df = pl.DataFrame({
        "open": [100.0, 100.5, 101.0, 101.5],
        "close": [100.0, 101.0, 101.5, 102.0],
        "timestamp_ms": [0, 300000, 600000, 1200000],
        "composite_score": [80.0, 0.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="contiguous 5m"):
        QuantBacktester().run_backtest(df, holding_bars=1)


def test_backtest_requires_timestamp_column_for_annualized_metrics():
    df = pl.DataFrame({
        "open": [100.0, 100.5, 101.0, 101.5],
        "close": [100.0, 101.0, 101.5, 102.0],
        "composite_score": [80.0, 0.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="requires 'timestamp_ms'"):
        QuantBacktester().run_backtest(df, holding_bars=1)
