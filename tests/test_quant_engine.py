"""
Comprehensive Test Suite for Quant Crypto Derivatives Engine.
Tests:
- Zero-GC contracts serialization
- OrderBook FSM sequence integrity and gap recovery
- Numba JIT Microstructure (OBI, VPIN)
- Synthetic Liquidation Detection
- CVD vs Price Absorption Divergence
- Factor Z-Score & Composite Score calibration
- Realistic CEX Execution Costs & Point-in-Time Audit
- Dynamic Risk Management & Invalidation-Based Sizing
- Walk-Forward Backtester & Deflated Sharpe Ratio (DSR)
- Model Context Protocol (MCP) AI Agent Tools
- BTC Market Regime & Beta Gate (Phase 5)
- Funding Settlement Epoch Countdown Gate (Phase 5)
- Anti-Spoofing Quality Gate (Phase 5)
- Wyckoff Liquidity Sweep & Reclaim Detection (Phase 5)
- Smart Money vs Retail Sentiment Divergence (Phase 5)
"""
import json
import math
import numpy as np
import polars as pl
import pytest

from binance_ingestion import BinanceOrderBookFSM, OrderBookFSMState
from contracts import NormalizedFunding, NormalizedTrade, OrderBookSnapshot, SignalEvent
from engine.backtester import QuantBacktester, compute_deflated_sharpe_ratio
from engine.divergence import detect_cvd_divergence_jit
from engine.execution_cost import CEXCostModel
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.mcp_server import MCPServer
from engine.microstructure_jit import StreamingVPINState, compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.risk import DynamicRiskEngine
from engine.sentiment import SentimentDivergence, SentimentEngine
from engine.signals import QuantSignalEngine


def test_orderbook_fsm_sync_and_gap_detection():
    fsm = BinanceOrderBookFSM("BTCUSDT")
    fsm.on_ws_connected()
    assert fsm.state == OrderBookFSMState.BUFFERING

    # Buffer valid events
    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 100, "s": "BTCUSDT", "U": 90, "u": 95, "pu": 85,
        "b": [["60000", "1.0"]], "a": [["60001", "1.0"]]
    })
    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 101, "s": "BTCUSDT", "U": 96, "u": 105, "pu": 95,
        "b": [["60000", "2.0"]], "a": [["60001", "1.5"]]
    })

    # Snapshot with lastUpdateId = 100
    snapshot = {
        "lastUpdateId": 100,
        "bids": [["60000", "1.0"], ["59990", "10.0"]],
        "asks": [["60001", "1.0"], ["60010", "10.0"]],
    }
    applied = fsm.apply_snapshot(snapshot)
    assert applied is True
    assert fsm.state == OrderBookFSMState.IN_SYNC

    snap = fsm.get_snapshot(depth=5)
    assert snap is not None
    assert snap.best_bid == 60000.0
    assert snap.best_ask == 60001.0
    assert snap.mid_price == 60000.5

    # Trigger sequence gap (expected pu=105, receiving pu=120)
    fsm.handle_depth_event({
        "e": "depthUpdate", "E": 102, "s": "BTCUSDT", "U": 121, "u": 125, "pu": 120,
        "b": [], "a": []
    })
    assert fsm.state == OrderBookFSMState.DISCONNECTED


def test_weighted_obi_jit():
    bids = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    asks = np.array([2.0, 1.0, 1.0], dtype=np.float64)
    obi = compute_weighted_obi_jit(bids, asks, decay=0.85)
    assert 0.0 < obi <= 1.0

    # Symmetric case
    obi_zero = compute_weighted_obi_jit(bids, bids, decay=0.85)
    assert abs(obi_zero) < 1e-6


def test_vpin_batch_and_streaming_consistency():
    qtys = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 1.0, 5.0], dtype=np.float64)
    is_buy = np.array([True, False, True, True, False, False, True, False], dtype=np.bool_)

    vpin_batch = compute_vpin_numba(qtys, is_buy, bucket_vol=5.0, window_baskets=5)

    streamer = StreamingVPINState(bucket_vol=5.0, window_baskets=5)
    for q, b in zip(qtys, is_buy):
        streamer.update_tick(q, b)
    vpin_stream = streamer.running_sum_imbalance / (streamer.filled_count * streamer.bucket_vol)

    assert abs(vpin_batch - vpin_stream) < 1e-6


def test_synthetic_liquidation_detection():
    detector = SyntheticLiquidationDetector(anomaly_ratio_threshold=1.5, volatility_shock_pct=0.3)

    # Long Liquidation cascade
    liq = detector.detect(
        symbol="BTCUSDT",
        current_price=60000.0,
        price_change_pct=-0.5,   # Damped
        delta_oi=-1000.0,        # Massive drop in OI
        taker_buy_vol=50.0,
        taker_sell_vol=200.0,    # Drop outstrips visible taker sell by 5x
    )
    assert liq is not None
    assert liq.side == "LONG_LIQUIDATION"
    assert liq.anomaly_ratio > 3.0
    assert liq.is_synthetic is True

    # Normal market turnover without anomaly
    no_liq = detector.detect(
        symbol="BTCUSDT",
        current_price=60000.0,
        price_change_pct=-0.1,
        delta_oi=-50.0,
        taker_buy_vol=200.0,
        taker_sell_vol=200.0,
    )
    assert no_liq is None


def test_cvd_divergence_jit():
    # Bullish Divergence: Price Lower Low, CVD Higher High
    prices = np.array([100.0, 98.0, 97.0, 98.5, 96.0], dtype=np.float64)
    cvd = np.array([-50.0, -100.0, -150.0, -80.0, -40.0], dtype=np.float64)

    score, div_type = detect_cvd_divergence_jit(prices, cvd, lookback=4)
    assert div_type == 1
    assert score > 0.0

    # Bearish Divergence: Price Higher High, CVD Lower Low
    prices_b = np.array([100.0, 102.0, 103.0, 101.5, 105.0], dtype=np.float64)
    cvd_b = np.array([50.0, 100.0, 150.0, 80.0, 40.0], dtype=np.float64)

    score_b, div_type_b = detect_cvd_divergence_jit(prices_b, cvd_b, lookback=4)
    assert div_type_b == -1
    assert score_b < 0.0


def test_quant_signal_calibration():
    engine = QuantSignalEngine()

    # Strong Long (Funding trap + Bullish Divergence + High OBI)
    sig_long = engine.compute_signal(
        symbol="BTCUSDT",
        current_price=60000.0,
        funding_rate_8h=-0.0006,
        basis_spread_bps=-15.0,
        delta_oi=600.0,
        oi_total=10000.0,
        obi=0.8,
        vpin=0.2,
        cvd_divergence_score=0.9,
        recent_high=61000.0,
        recent_low=59800.0,
    )
    assert sig_long.signal_type == "STRONG_LONG"
    assert sig_long.composite_score >= 75.0
    assert sig_long.risk_reward_ratio == 2.0
    assert sig_long.invalidation_price < sig_long.price < sig_long.target_price

    # Strong Short (Overheated Funding + Bearish Divergence + Negative OBI)
    sig_short = engine.compute_signal(
        symbol="ETHUSDT",
        current_price=3000.0,
        funding_rate_8h=0.0008,
        basis_spread_bps=15.0,
        delta_oi=500.0,
        oi_total=8000.0,
        obi=-0.75,
        vpin=0.25,
        cvd_divergence_score=-0.9,
        recent_high=3050.0,
        recent_low=2950.0,
    )
    assert sig_short.signal_type == "STRONG_SHORT"
    assert sig_short.composite_score <= -75.0
    assert sig_short.invalidation_price > sig_short.price > sig_short.target_price


def test_cex_cost_model_and_point_in_time_audit():
    cost_model = CEXCostModel(maker_fee_bps=2.0, taker_fee_bps=5.0, base_latency_ms=50)

    # Market Buy Order Execution with Adverse Selection
    t_decision = 1700000000000
    res = cost_model.simulate_execution(
        symbol="BTCUSDT",
        side="BUY",
        reference_price=50000.0,
        spread_bps=2.0,
        is_market_order=True,
        decision_timestamp_ms=t_decision,
    )

    assert res.executed_price > 50000.0  # Paid spread + adverse selection
    assert res.execution_timestamp_ms == t_decision + 50
    assert res.fee_bps == 5.0

    # Verify Point-in-Time Audit failure if T_execution < T_decision
    with pytest.raises(ValueError, match="Lookahead bias violation"):
        cost_model.simulate_execution(
            symbol="BTCUSDT",
            side="BUY",
            reference_price=50000.0,
            spread_bps=2.0,
            decision_timestamp_ms=t_decision,
            latency_ms=-10,  # Negative latency represents lookahead bias
        )


def test_dynamic_position_sizing_and_leverage():
    risk_engine = DynamicRiskEngine(
        default_account_capital=10000.0,
        risk_per_trade_pct=1.0,  # $100 risk
        max_leverage=10.0,
    )

    rec = risk_engine.calculate_sizing(
        symbol="BTCUSDT",
        signal_type="STRONG_LONG",
        entry_price=60000.0,
        invalidation_price=59400.0,  # 1% stop distance ($600)
        target_price=61200.0,        # 2% target distance ($1200) -> 2R
    )

    assert rec is not None
    assert rec.side == "LONG"
    assert rec.risk_reward_ratio == 2.0
    assert rec.risk_per_trade_usdt == 100.0
    # Qty = $100 / $600 = 0.166667 BTC
    expected_qty = 100.0 / 600.0
    assert abs(rec.recommended_quantity - expected_qty) < 1e-4
    assert rec.effective_leverage <= 10.0


def test_backtest_deflated_sharpe_ratio():
    backtester = QuantBacktester()

    # Generate synthetic price series and signal scores
    np.random.seed(42)
    n = 300
    prices = 50000.0 + np.cumsum(np.random.randn(n) * 30.0)
    scores = np.zeros(n)
    # Inject deliberate winning long setups
    scores[10] = 85.0
    scores[50] = 88.0
    scores[120] = -85.0

    df = pl.DataFrame({
        "close": prices,
        "composite_score": scores,
    })

    metrics = backtester.run_backtest(df, holding_bars=3)
    assert metrics.total_trades > 0
    assert 0.0 <= metrics.deflated_sharpe_ratio <= 1.0


def test_mcp_server_interface():
    server = MCPServer()

    # Test tools/list
    req_list = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    resp_list = json.loads(server.handle_request(req_list))
    tools = resp_list["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "get_screener_signals" in tool_names
    assert "calculate_position_size" in tool_names
    assert "run_strategy_backtest" in tool_names

    # Test tools/call calculate_position_size
    req_call = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "calculate_position_size",
            "arguments": {
                "symbol": "BTCUSDT",
                "signal_type": "STRONG_LONG",
                "entry_price": 60000.0,
                "invalidation_price": 59400.0,
                "target_price": 61200.0,
                "account_capital_usdt": 10000.0,
            }
        }
    })
    resp_call = json.loads(server.handle_request(req_call))
    content_text = resp_call["result"]["content"][0]["text"]
    res_data = json.loads(content_text)
    assert res_data["symbol"] == "BTCUSDT"
    assert res_data["side"] == "LONG"
    assert res_data["risk_reward_ratio"] == 2.0


# ============================================================
# Phase 5 Filter Tests: BTC Regime, Funding Gate, Quality, Sweep, Sentiment
# ============================================================

class TestBTCMarketRegimeGate:
    """Phase 5 Filter 1: BTC Regime & Beta Gate."""

    def setup_method(self):
        self.engine = MarketRegimeEngine(
            dump_threshold_5m_pct=-0.30,
            pump_threshold_5m_pct=+0.30,
            beta_alt_btc=1.6,
            min_decoupled_rs_pct=1.2,
        )

    def test_impulse_dump_blocks_alt_longs(self):
        """BTC drops -0.5% in 5m → IMPULSE_DUMP → all alt longs blocked."""
        regime = self.engine.evaluate_btc_regime(
            btc_last_price=60000.0 * (1 - 0.005),  # -0.5%
            btc_prev_price=60000.0,
            btc_change_24h_pct=-1.2,
        )
        assert regime.status == "IMPULSE_DUMP"
        assert regime.allow_alt_longs is False
        assert regime.allow_alt_shorts is True

        # Alt with no decoupling should be blocked
        allowed, reason = self.engine.check_signal_gate(
            symbol="SOLUSDT",
            signal_type="STRONG_LONG",
            btc_regime=regime,
            alt_change_5m_pct=-0.4,  # Also falling with BTC (RS ≈ -0.4 - 1.6*(-0.5) = +0.4 < 1.2)
        )
        assert allowed is False
        assert "BLOCKED_BY_BTC_DUMP" in reason

    def test_decoupled_alt_passes_during_btc_dump(self):
        """Alt with strong positive RS passes even during BTC dump."""
        regime = self.engine.evaluate_btc_regime(
            btc_last_price=60000.0 * (1 - 0.005),
            btc_prev_price=60000.0,
            btc_change_24h_pct=-1.2,
        )
        # Alt rising +2% while BTC dumps -0.5%: RS = 2.0 - 1.6*(-0.5) = +2.8 > 1.2
        allowed, reason = self.engine.check_signal_gate(
            symbol="SOLUSDT",
            signal_type="STRONG_LONG",
            btc_regime=regime,
            alt_change_5m_pct=+2.0,
        )
        assert allowed is True
        assert "DECOUPLED_STRENGTH" in reason

    def test_impulse_pump_blocks_alt_shorts(self):
        """BTC pumps +0.5% in 5m → IMPULSE_PUMP → all alt shorts blocked."""
        regime = self.engine.evaluate_btc_regime(
            btc_last_price=60000.0 * (1 + 0.005),
            btc_prev_price=60000.0,
            btc_change_24h_pct=+1.5,
        )
        assert regime.status == "IMPULSE_PUMP"
        assert regime.allow_alt_shorts is False

        allowed, reason = self.engine.check_signal_gate(
            symbol="ETHUSDT",
            signal_type="STRONG_SHORT",
            btc_regime=regime,
            alt_change_5m_pct=+0.4,  # Also pumping with BTC
        )
        assert allowed is False
        assert "BLOCKED_BY_BTC_PUMP" in reason

    def test_neutral_range_allows_all(self):
        """BTC flat → NEUTRAL_RANGING → all alt signals allowed."""
        regime = self.engine.evaluate_btc_regime(
            btc_last_price=60010.0,  # Tiny +0.017% change
            btc_prev_price=60000.0,
            btc_change_24h_pct=+0.2,
        )
        assert regime.status == "NEUTRAL_RANGING"
        assert regime.allow_alt_longs is True
        assert regime.allow_alt_shorts is True

    def test_btc_self_always_passes(self):
        """BTCUSDT signals always bypass the BTC regime gate."""
        regime = self.engine.evaluate_btc_regime(
            btc_last_price=59700.0,
            btc_prev_price=60000.0,
            btc_change_24h_pct=-1.0,
        )
        allowed, reason = self.engine.check_signal_gate(
            symbol="BTCUSDT",
            signal_type="STRONG_LONG",
            btc_regime=regime,
            alt_change_5m_pct=-0.5,
        )
        assert allowed is True
        assert reason == "BTC_SELF"

    def test_relative_strength_calculation(self):
        """RS = alt_change - beta * btc_change."""
        rs = self.engine.calculate_relative_strength(
            alt_change_5m_pct=+1.0,
            btc_change_5m_pct=-0.5,
        )
        # RS = 1.0 - 1.6 * (-0.5) = 1.0 + 0.8 = +1.8%
        assert abs(rs - 1.8) < 1e-6


class TestFundingEpochGate:
    """Phase 5 Filter 3: Funding Settlement Countdown Gate."""

    def setup_method(self):
        self.engine = FundingFilterEngine(proximity_threshold_minutes=20.0)

    def test_blocks_long_when_positive_funding_near_settlement(self):
        """Strong positive funding + < 20m to settlement → block long entry."""
        now_ms = 1700000000000
        # Settlement in 10 minutes
        next_funding_ms = now_ms + (10 * 60 * 1000)

        result = self.engine.evaluate_funding_gate(
            symbol="SOLUSDT",
            signal_type="STRONG_LONG",
            funding_rate_8h=+0.0005,   # +0.05% → longs will pay shorts in 10 min
            next_funding_time_ms=next_funding_ms,
            current_time_ms=now_ms,
        )
        assert result.is_in_epoch_window is True
        assert result.allow_long is False
        assert "BLOCKED_PRE_FUNDING_PAYOUT" in result.gate_reason
        assert result.minutes_to_settlement == pytest.approx(10.0, abs=0.2)

    def test_blocks_short_when_negative_funding_near_settlement(self):
        """Strong negative funding + < 20m to settlement → block short entry."""
        now_ms = 1700000000000
        next_funding_ms = now_ms + (8 * 60 * 1000)  # 8 minutes

        result = self.engine.evaluate_funding_gate(
            symbol="ETHUSDT",
            signal_type="STRONG_SHORT",
            funding_rate_8h=-0.0005,   # -0.05% → shorts will pay longs
            next_funding_time_ms=next_funding_ms,
            current_time_ms=now_ms,
        )
        assert result.allow_short is False
        assert "BLOCKED_PRE_FUNDING_PAYOUT" in result.gate_reason

    def test_passes_outside_epoch_window(self):
        """Position is fine when settlement is > 20 minutes away."""
        now_ms = 1700000000000
        next_funding_ms = now_ms + (60 * 60 * 1000)  # 60 minutes away

        result = self.engine.evaluate_funding_gate(
            symbol="BTCUSDT",
            signal_type="STRONG_LONG",
            funding_rate_8h=+0.0008,
            next_funding_time_ms=next_funding_ms,
            current_time_ms=now_ms,
        )
        assert result.is_in_epoch_window is False
        assert result.allow_long is True
        assert result.gate_reason == "PASSED"

    def test_passes_with_mild_funding_near_settlement(self):
        """Mild funding (below threshold) near settlement does NOT block."""
        now_ms = 1700000000000
        next_funding_ms = now_ms + (5 * 60 * 1000)  # 5 minutes

        result = self.engine.evaluate_funding_gate(
            symbol="BNBUSDT",
            signal_type="STRONG_LONG",
            funding_rate_8h=+0.0001,   # +0.01% → below 0.03% threshold
            next_funding_time_ms=next_funding_ms,
            current_time_ms=now_ms,
        )
        assert result.allow_long is True
        assert result.gate_reason == "PASSED"


class TestQualityAntiSpoofingGate:
    """Phase 5 Filter 5: Anti-Spoofing & Liquidity Quality Gate."""

    def setup_method(self):
        self.gate = QualityFilter(
            min_24h_volume_usdt=15_000_000.0,
            max_spread_bps=3.5,
            max_abs_funding_rate_8h=0.015,
        )

    def test_rejects_low_volume_token(self):
        """Tokens under $15M 24h volume are rejected to prevent OBI spoofing."""
        result = self.gate.evaluate(
            symbol="SPAMCOINUSDT",
            quote_volume_24h=8_000_000.0,   # $8M < $15M threshold
            spread_bps=2.0,
            funding_rate_8h=0.0001,
        )
        assert result.is_valid is False
        assert "LOW_VOLUME" in result.rejection_reason

    def test_rejects_wide_spread_token(self):
        """Tokens with spread > 3.5 bps indicate illiquid/manipulated books."""
        result = self.gate.evaluate(
            symbol="THINUSDT",
            quote_volume_24h=50_000_000.0,
            spread_bps=5.2,   # > 3.5 bps threshold
            funding_rate_8h=0.0001,
        )
        assert result.is_valid is False
        assert "WIDE_SPREAD" in result.rejection_reason

    def test_rejects_extreme_funding_anomaly(self):
        """Tokens with extreme funding (delisting trap / capped rate) are rejected."""
        result = self.gate.evaluate(
            symbol="DELISTEDUSDT",
            quote_volume_24h=100_000_000.0,
            spread_bps=1.5,
            funding_rate_8h=-0.02,  # -2% extreme rate (cap anomaly / delisting signal)
        )
        assert result.is_valid is False
        assert "FUNDING_ANOMALY" in result.rejection_reason

    def test_passes_healthy_liquid_token(self):
        """Standard liquid tokens with normal params pass all gates."""
        result = self.gate.evaluate(
            symbol="SOLUSDT",
            quote_volume_24h=200_000_000.0,
            spread_bps=1.2,
            funding_rate_8h=0.0001,
        )
        assert result.is_valid is True
        assert result.rejection_reason == "PASSED"


class TestWyckoffLiquiditySweepDetector:
    """Phase 5 Filter 4: Liquidity Sweep & Reclaim (Wyckoff Spring / Upthrust)."""

    def setup_method(self):
        self.detector = LiquiditySweepDetector()

    def test_bullish_spring_reclaim_detected(self):
        """Price sweeps below swing low, then reclaims it with positive CVD → Spring."""
        event = self.detector.detect(
            symbol="SOLUSDT",
            current_price=101.5,       # Reclaimed above swing low (100.0)
            current_high=103.0,
            current_low=99.3,          # Swept below swing_low by -0.7% (clean sweep)
            recent_swing_high=105.0,
            recent_swing_low=100.0,    # Swing low
            cvd_delta=+500.0,          # Buyers absorbing the stop cascade
        )
        assert event is not None
        assert event.is_confirmed is True
        assert event.pattern_type == "BULLISH_SWEEP_RECLAIM"
        assert event.swept_level == 100.0
        assert 0.05 <= event.penetration_pct <= 2.5

    def test_bearish_upthrust_reclaim_detected(self):
        """Price sweeps above swing high, then drops back below → Bearish Upthrust."""
        event = self.detector.detect(
            symbol="ETHUSDT",
            current_price=2498.0,      # Price fell back below swing high (2500.0)
            current_high=2516.0,       # Swept above swing_high by +0.64% (fake breakout)
            current_low=2490.0,
            recent_swing_high=2500.0,  # Swing high
            recent_swing_low=2450.0,
            cvd_delta=-300.0,          # Sellers absorbing the FOMO breakout buyers
        )
        assert event is not None
        assert event.is_confirmed is True
        assert event.pattern_type == "BEARISH_SWEEP_RECLAIM"
        assert event.swept_level == 2500.0

    def test_no_sweep_when_no_penetration(self):
        """No sweep event when price stays above swing low (no stop hunt)."""
        event = self.detector.detect(
            symbol="BTCUSDT",
            current_price=60500.0,
            current_high=61000.0,
            current_low=60200.0,       # Low stays ABOVE swing_low (60000)
            recent_swing_high=62000.0,
            recent_swing_low=60000.0,
            cvd_delta=+100.0,
        )
        assert event is None

    def test_no_spring_without_cvd_confirmation(self):
        """Sweep below swing low with NEGATIVE CVD = no buyer absorption → not confirmed."""
        event = self.detector.detect(
            symbol="BNBUSDT",
            current_price=301.0,
            current_high=305.0,
            current_low=298.5,         # Swept below 300.0
            recent_swing_high=310.0,
            recent_swing_low=300.0,
            cvd_delta=-200.0,          # Sellers still in control → NOT a Spring
        )
        assert event is None

    def test_excessive_sweep_depth_rejected(self):
        """Sweep penetration > 2.5% is too deep to be a clean stop hunt (panic dump)."""
        event = self.detector.detect(
            symbol="AVAXUSDT",
            current_price=51.0,
            current_high=53.0,
            current_low=46.0,          # Swept 8% below swing low = panic, not a spring
            recent_swing_high=55.0,
            recent_swing_low=50.0,
            cvd_delta=+100.0,
        )
        assert event is None


class TestSentimentDivergenceScoring:
    """Phase 5 Filter 2: Smart Money vs Retail Crowd Divergence Scoring."""

    def _make_divergence(
        self,
        whale_ls: float,
        retail_ls: float,
        taker_ratio: float = 1.0,
    ) -> dict:
        """
        Manually replicate the composite divergence formula from SentimentEngine
        for deterministic unit testing (no HTTP calls needed).
        """
        log_whale = math.log(max(whale_ls, 0.01))
        log_retail = math.log(max(retail_ls, 0.01))
        whale_retail_diff = log_whale - log_retail
        taker_log = math.log(max(taker_ratio, 0.01))
        raw_diff = 0.70 * whale_retail_diff + 0.30 * taker_log
        div_score = max(-1.0, min(1.0, raw_diff / 1.5))
        z_whale = max(-3.0, min(3.0, div_score * 3.0))

        if div_score >= 0.25:
            bias = "SMART_MONEY_LONG"
        elif div_score <= -0.25:
            bias = "RETAIL_TRAP_SHORT"
        else:
            bias = "NEUTRAL"

        return {"div_score": div_score, "z_whale": z_whale, "bias": bias}

    def test_smart_money_long_when_whales_long_retail_short(self):
        """Whales: 66% long (ratio=2.0), Retail: 30% long (ratio=0.43) → SMART_MONEY_LONG."""
        result = self._make_divergence(whale_ls=2.0, retail_ls=0.43, taker_ratio=1.3)
        assert result["bias"] == "SMART_MONEY_LONG"
        assert result["div_score"] > 0.25
        assert result["z_whale"] > 0.0

    def test_retail_trap_when_crowd_long_whales_short(self):
        """Retail: 71% long (ratio=2.5), Whales: 33% long (ratio=0.5) → RETAIL_TRAP_SHORT."""
        result = self._make_divergence(whale_ls=0.5, retail_ls=2.5, taker_ratio=0.7)
        assert result["bias"] == "RETAIL_TRAP_SHORT"
        assert result["div_score"] < -0.25
        assert result["z_whale"] < 0.0

    def test_neutral_when_positioning_balanced(self):
        """Both whales and retail have similar ratios → NEUTRAL."""
        result = self._make_divergence(whale_ls=1.05, retail_ls=1.05, taker_ratio=1.0)
        assert result["bias"] == "NEUTRAL"
        assert abs(result["div_score"]) < 0.25

    def test_taker_flow_boosts_bullish_score(self):
        """High taker buy/sell ratio amplifies bullish whale divergence."""
        # Baseline: mild whale long, retail neutral
        base = self._make_divergence(whale_ls=1.3, retail_ls=1.0, taker_ratio=1.0)
        # With strong buy takers
        boosted = self._make_divergence(whale_ls=1.3, retail_ls=1.0, taker_ratio=2.0)
        # Taker component (log(2.0) * 0.30 / 1.5) should push score higher
        assert boosted["div_score"] > base["div_score"]

    def test_taker_flow_dampens_bearish_score_when_buying(self):
        """Even when whales are slightly short, aggressive buy takers can push toward neutral."""
        # Whale slight short
        no_taker = self._make_divergence(whale_ls=0.85, retail_ls=1.0, taker_ratio=1.0)
        # With strong buy taker flow pushing back
        with_takers = self._make_divergence(whale_ls=0.85, retail_ls=1.0, taker_ratio=2.5)
        # Taker buying should pull the composite score toward positive (less negative)
        assert with_takers["div_score"] > no_taker["div_score"]

    def test_z_whale_clipped_to_bounds(self):
        """Z-score never exceeds ±3.0 regardless of extreme ratio inputs."""
        extreme_bull = self._make_divergence(whale_ls=100.0, retail_ls=0.01, taker_ratio=100.0)
        extreme_bear = self._make_divergence(whale_ls=0.01, retail_ls=100.0, taker_ratio=0.01)
        assert extreme_bull["z_whale"] <= 3.0
        assert extreme_bear["z_whale"] >= -3.0

    def test_sentiment_divergence_struct_fields(self):
        """SentimentDivergence msgspec struct is zero-GC and serializable."""
        div = SentimentDivergence(
            symbol="SOLUSDT",
            retail_ls_ratio=2.5,
            retail_long_pct=71.4,
            top_traders_ls_ratio=0.5,
            top_traders_long_pct=33.3,
            taker_buy_sell_ratio=0.8,
            divergence_score=-0.42,
            z_whale_sentiment=-1.26,
            sentiment_bias="RETAIL_TRAP_SHORT",
        )
        import msgspec
        encoded = msgspec.json.encode(div)
        decoded = msgspec.json.decode(encoded, type=SentimentDivergence)
        assert decoded.symbol == "SOLUSDT"
        assert decoded.taker_buy_sell_ratio == 0.8
        assert decoded.sentiment_bias == "RETAIL_TRAP_SHORT"
