import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from binance_ingestion import (
    BinanceFuturesIngestion,
    BinanceOrderBookFSM,
    BinanceRestrictedLocationError,
    OrderBookFSMState,
)
from contracts import MarketStateSnapshot, SignalEvent
from engine.circuit_breaker import CircuitBreaker
from engine.divergence import detect_cvd_divergence_jit
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.microstructure_jit import compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.screener import QuantScreener
from engine.sentiment import SentimentEngine
from engine.signals import (
    QuantSignalEngine,
    calculate_position_size,
    empirical_zscore,
    percentile_score,
)

def test_percentile_score_handles_bimodal_distribution():
    """CVD divergence is bimodal (mostly 0, occasionally ±1). Z-score explodes; percentile is stable."""
    from engine.signals import percentile_score
    history = [0.0] * 20 + [1.0, -1.0, 0.0, 0.0]  # 24 samples
    z_extreme = percentile_score(1.0, history, min_samples=20)
    z_zero = percentile_score(0.0, history, min_samples=20)
    # Extreme value should be high but not clipped to +3 in a way that loses info
    assert z_extreme > z_zero
    assert -3.0 <= z_extreme <= 3.0
    assert -3.0 <= z_zero <= 3.0


def test_position_sizing_respects_max_leverage():
    from engine.signals import calculate_position_size
    # Very tight stop (0.1%) — position would exceed max_leverage, should be capped
    pos_usd, lev = calculate_position_size(10000.0, 100.0, 99.9, risk_per_trade_pct=0.01, max_leverage=3)
    assert lev <= 3
    assert pos_usd <= 30000.0


def test_neutral_signal_has_zero_rr_not_default():
    """Regression: NEUTRAL signals must report R:R = 0.0, not the default 2.0."""
    engine = QuantSignalEngine(z_history_min_samples=4)
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=0.0,
        recent_high=101.0, recent_low=99.0,
        z_cvd_override=0.0, z_fund_override=0.0, z_delta_oi_override=0.0,
        z_micro_override=0.0, z_whale_override=0.0, atr_pct=0.01,
    )
    assert sig.signal_type == "NEUTRAL"
    assert sig.risk_reward_ratio == 0.0


def test_funding_filter_does_not_block_short_at_positive_extreme():
    """SHORT at +0.15% funding receives payment — should not be blocked."""
    engine = FundingFilterEngine()
    result = engine.evaluate_funding_gate(
        symbol="BTCUSDT", signal_type="STRONG_SHORT",
        funding_rate_8h=0.0015, next_funding_time_ms=1700000000000 + 10 * 60 * 1000,
        current_time_ms=1700000000000,
    )
    assert result.allow_short is True
    assert result.is_favorable_for_short is True


def test_funding_filter_blocks_long_at_positive_extreme():
    engine = FundingFilterEngine()
    result = engine.evaluate_funding_gate(
        symbol="BTCUSDT", signal_type="STRONG_LONG",
        funding_rate_8h=0.0015, next_funding_time_ms=1700000000000 + 10 * 60 * 1000,
        current_time_ms=1700000000000,
    )
    assert result.allow_long is False


def test_funding_filter_does_not_block_at_below_old_threshold():
    """Old code blocked at 0.03%. New threshold is 0.10%. 0.05% should pass."""
    engine = FundingFilterEngine()
    result = engine.evaluate_funding_gate(
        symbol="BTCUSDT", signal_type="STRONG_LONG",
        funding_rate_8h=0.0005, next_funding_time_ms=1700000000000 + 10 * 60 * 1000,
        current_time_ms=1700000000000,
    )
    assert result.allow_long is True


def test_friction_model_btc_vs_meme_is_asymmetric():
    """BTC (spread 0.5 bps, ATR 0.15%) should have lower friction than meme (spread 5 bps, ATR 2%)."""
    btc_friction = QuantScreener._compute_friction_rt(0.5, 0.0015)
    meme_friction = QuantScreener._compute_friction_rt(5.0, 0.02)
    assert btc_friction < meme_friction
    assert btc_friction < 0.0015  # < 15 bps for BTC
    assert meme_friction >= 0.0020  # > 20 bps for meme


def test_circuit_breaker_halts_after_max_failures(tmp_path):
    from engine.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(state_file=tmp_path / "cb.json", max_consecutive_failures=3, halt_duration_minutes=10)
    assert not cb.is_halted()
    for _ in range(3):
        cb.record_failure()
    assert cb.is_halted()
    cb.record_success()
    assert not cb.is_halted()


def test_correlation_limit_downgrades_excess_strong_signals():
    """When 5 STRONG_LONG signals are returned, only max_per_direction should survive."""
    from contracts import SignalEvent
    screener = QuantScreener(state_file=Path("/tmp/dummy.bin"), top_n_symbols=1, concurrency_limit=1)
    screener.max_strong_per_direction = 3

    def mk_sym(sym, score):
        return SignalEvent(
            symbol=sym, timestamp_ms=1, signal_type="STRONG_LONG", composite_score=score,
            z_cvd_div=3.0, z_fund_trap=3.0, z_delta_oi=3.0, z_micro=3.0, vpin=0.5, obi=0.0,
            funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=99.0,
            target_price=102.0, risk_reward_ratio=2.0, decision_timestamp_ms=1,
        )

    signals = [mk_sym(f"SYM{i}", 80.0 - i) for i in range(5)]
    filtered, downgraded = screener._apply_portfolio_correlation_limit(signals)
    strong_remaining = [s for s in filtered if s.signal_type == "STRONG_LONG"]
    assert len(strong_remaining) == 3
    assert downgraded == 2

def test_atr_stop_multiplier_is_used_for_strong_signal():
    engine = QuantSignalEngine(
        z_history_min_samples=4,
        atr_stop_multiplier=2.0,
        max_atr_multiplier=3.0,
        sweep_booster_points=0.0,
        min_effective_rrr=0.0,
    )
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=1.0,
        recent_high=101.0, recent_low=99.0, atr_pct=0.01,
        z_cvd_override=3.0, z_fund_override=3.0, z_delta_oi_override=3.0,
        z_micro_override=3.0, z_whale_override=3.0,
    )
    assert sig.signal_type == "STRONG_LONG"
    assert math.isclose((sig.price - sig.invalidation_price) / sig.price, 0.02, rel_tol=1e-9)


def test_portfolio_correlation_blocks_only_highly_correlated_candidates():
    screener = QuantScreener(state_file=Path("/tmp/dummy.bin"), top_n_symbols=1, concurrency_limit=1)
    screener.max_strong_per_direction = 3
    screener.max_portfolio_correlation = 0.80
    screener.max_aggregate_risk_pct = 0.03

    def mk_sym(sym, score):
        return SignalEvent(
            symbol=sym, timestamp_ms=1, signal_type="STRONG_LONG", composite_score=score,
            z_cvd_div=3.0, z_fund_trap=3.0, z_delta_oi=3.0, z_micro=3.0, vpin=0.5, obi=0.0,
            funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0,
            target_price=104.0, risk_reward_ratio=2.0, decision_timestamp_ms=1,
            suggested_position_usd=500.0, suggested_leverage=1,
        )

    base = tuple(float(x) for x in range(1, 61))
    anti = tuple(float(61 - x) for x in range(1, 61))
    snapshots = {
        "A": MarketStateSnapshot(symbol="A", timestamp_ms=1, last_price=100.0, open_interest=1.0, delta_oi_5m=0.0,
                                  cumulative_cvd_5m=0.0, funding_rate_8h=0.0, basis_bps=0.0, vpin_estimate=0.5,
                                  obi_score=0.0, composite_score=90.0, price_history_5m=base),
        "B": MarketStateSnapshot(symbol="B", timestamp_ms=1, last_price=100.0, open_interest=1.0, delta_oi_5m=0.0,
                                  cumulative_cvd_5m=0.0, funding_rate_8h=0.0, basis_bps=0.0, vpin_estimate=0.5,
                                  obi_score=0.0, composite_score=85.0, price_history_5m=base),
        "C": MarketStateSnapshot(symbol="C", timestamp_ms=1, last_price=100.0, open_interest=1.0, delta_oi_5m=0.0,
                                  cumulative_cvd_5m=0.0, funding_rate_8h=0.0, basis_bps=0.0, vpin_estimate=0.5,
                                  obi_score=0.0, composite_score=80.0, price_history_5m=anti),
    }
    signals = [mk_sym("A", 90.0), mk_sym("B", 85.0), mk_sym("C", 80.0)]
    filtered, downgraded = screener._apply_portfolio_correlation_limit(signals, snapshots)
    strong = [s.symbol for s in filtered if s.signal_type == "STRONG_LONG"]
    assert strong == ["A", "C"]
    assert downgraded == 1
    assert next(s for s in filtered if s.symbol == "B").gate_status.startswith("BLOCKED_PORTFOLIO_CORRELATION")


def test_cvd_series_reconstruction_is_point_in_time():
    rows = []
    for i in range(12):
        # open, high, low, close, volume, taker_buy, taker_sell
        price = 100.0 + (i if i < 8 else 8 - (i - 7) * 0.5)
        rows.append((i, price, price + 1.0, price - 1.0, price, 10.0, 6.0 + i * 0.1, 4.0 - i * 0.1))
    cvd, div = QuantScreener._derive_cvd_series_and_divergence(rows, lookback=4)
    assert len(cvd) == len(rows)
    assert len(div) == len(rows)
    assert all(math.isfinite(x) for x in cvd)
    assert all(math.isfinite(x) for x in div)


def test_signal_timestamp_distinguishes_candle_close_from_decision_time():
    engine = QuantSignalEngine(z_history_min_samples=4, min_effective_rrr=0.0)
    candle_close_ms = 1_700_000_299_999
    decision_ms = candle_close_ms + 37_000
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=0.0,
        recent_high=101.0, recent_low=99.0, atr_pct=0.01,
        z_cvd_override=0.0, z_fund_override=0.0, z_delta_oi_override=0.0,
        z_micro_override=0.0, z_whale_override=0.0,
        timestamp_ms=candle_close_ms, decision_timestamp_ms=decision_ms,
    )
    assert sig.timestamp_ms == candle_close_ms
    assert sig.decision_timestamp_ms == decision_ms


def test_event_study_uses_completed_candle_without_one_bar_shift():
    from scripts.evaluate_signal_ledger import signal_candle_open_ms
    candle_open = (1_700_000_299_999 // (5 * 60 * 1000)) * (5 * 60 * 1000)
    assert signal_candle_open_ms(1_700_000_299_999) == candle_open
    assert signal_candle_open_ms(1_700_000_299_999, candle_open) == candle_open


def test_signal_ledger_is_idempotent_and_records_candle_bounds(tmp_path):
    from engine.signal_ledger import append_signal_events
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_099_999, signal_type="STRONG_LONG", composite_score=80.0,
        z_cvd_div=1.0, z_fund_trap=1.0, z_delta_oi=1.0, z_micro=1.0, vpin=0.5, obi=0.1,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=1.8, decision_timestamp_ms=1_700_000_137_000,
    )
    ledger = tmp_path / "ledger.jsonl"
    assert append_signal_events(ledger, [sig]) == 1
    assert append_signal_events(ledger, [sig]) == 0
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert row["candle_open_ms"] == (sig.timestamp_ms // (5 * 60 * 1000)) * (5 * 60 * 1000)
    assert row["candle_close_ms"] == row["candle_open_ms"] + 5 * 60 * 1000 - 1


def test_risk_guard_fails_closed_without_live_equity(tmp_path, monkeypatch):
    from engine.risk_guard import RiskGuard
    monkeypatch.delenv("CURRENT_EQUITY_USDT", raising=False)
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    guard = RiskGuard(
        state_file=tmp_path / "equity.json", account_equity=10_000.0,
        paper_trading=False, kill_switch_file=tmp_path / "KILL_SWITCH",
    )
    allowed, reason = guard.evaluate()
    assert allowed is False
    assert reason == "EQUITY_RECONCILIATION_REQUIRED"
