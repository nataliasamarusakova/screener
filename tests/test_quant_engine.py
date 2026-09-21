"""Regression tests for the production quant/risk paths."""
import asyncio
import json
import math

import numpy as np
import pytest

from binance_ingestion import (
    BinanceFuturesIngestion,
    BinanceOrderBookFSM,
    BinanceRestrictedLocationError,
    OrderBookFSMState,
)
from contracts import MarketStateSnapshot
from engine.divergence import detect_cvd_divergence_jit
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.microstructure_jit import compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.sentiment import SentimentEngine
from engine.signals import QuantSignalEngine, empirical_zscore, calculate_position_size
from engine.screener import QuantScreener


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


def test_position_sizing_calculation():
    # Depo = $10,000, 1% risk = $100. Stop is 2% away. Size must be $5,000, leverage 1x
    pos_usd, lev = calculate_position_size(10000.0, 100.0, 98.0, risk_per_trade_pct=0.01)
    assert pos_usd == 5000.0
    assert lev == 1

    # Stop is 0.5% away. Position would be $20,000, leverage 2x
    pos_usd, lev = calculate_position_size(10000.0, 100.0, 99.5, risk_per_trade_pct=0.01)
    assert pos_usd == 20000.0
    assert lev == 2


def test_negative_beta_calculation_is_supported():
    step = 5 * 60 * 1000
    times = [1700000000000 + i * step for i in range(30)]
    btc = [60000.0]
    alt = [3000.0]
    returns = [0.0005, -0.0002, 0.0008, 0.0001, -0.0004] * 6
    for r in returns:
        btc.append(btc[-1] * (1.0 + r))
        alt.append(alt[-1] * (1.0 - 1.5 * r)) # Negative correlation
    beta = MarketRegimeEngine.calculate_rolling_beta(times, alt, times, btc, min_samples=24)
    assert beta is not None
    assert beta == pytest.approx(-1.5, rel=1e-2)


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
    assert empirical_zscore(2.0, [1.0, 1.0, 1.0, 1.0], min_samples=4) == 0.0
    assert empirical_zscore(1.0, [1.0, 1.0, 1.0, 1.0], min_samples=4) == 0.0


def test_signal_generation_with_position_sizing_and_friction():
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
        account_equity=10000.0,
    )
    sig = engine.compute_signal(**common)
    assert sig.signal_type == "STRONG_LONG"
    assert sig.suggested_position_usd > 0.0
    assert sig.suggested_leverage >= 1
    assert sig.invalidation_price < 100.0 < sig.target_price


def test_friction_gate_blocks_signals_with_insufficient_net_rrr():
    # If target is too close and fees eat the edge, signal must be downgraded to NEUTRAL
    engine = QuantSignalEngine(min_effective_rrr=3.0) # Unattainable hurdle
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=100.0, oi_total=1000.0, obi=0.8, vpin=0.2, cvd_divergence_score=1.0,
        recent_high=101.0, recent_low=99.0, z_cvd_override=3.0, z_fund_override=3.0,
        z_delta_oi_override=3.0, z_micro_override=3.0, z_whale_override=3.0, atr_pct=0.01,
        sweep_reclaim=True, sweep_pattern="BULLISH_SWEEP_RECLAIM", timestamp_ms=1700000000000,
    )
    assert sig.signal_type == "NEUTRAL"
    assert "BLOCKED_UNPROFITABLE_AFTER_FEES" in sig.gate_status


def test_sentiment_fails_closed_without_api_key():
    result = asyncio.run(SentimentEngine(api_key=None).fetch_sentiment_divergence("BTCUSDT"))
    assert result is None
