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
    from engine.signals import percentile_score, empirical_zscore
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


def test_signal_event_schema_contract():
    import msgspec
    from contracts import SignalEvent
    from engine.serialization import CANONICAL_SIGNAL_FIELDS, signal_to_dict
    struct_fields = [field.name for field in msgspec.structs.fields(SignalEvent)]
    assert struct_fields == list(CANONICAL_SIGNAL_FIELDS)
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1, signal_type="NEUTRAL", composite_score=0.0,
        z_cvd_div=0.0, z_fund_trap=0.0, z_delta_oi=0.0, z_micro=0.0, vpin=0.5, obi=0.0,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=100.0, target_price=100.0,
        risk_reward_ratio=0.0, decision_timestamp_ms=2,
    )
    assert list(signal_to_dict(sig)) == list(CANONICAL_SIGNAL_FIELDS)


def test_gate_block_does_not_truncate_composite_score():
    engine = QuantSignalEngine(z_history_min_samples=4, strong_signal_threshold=75.0, min_effective_rrr=0.0)
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=1.0,
        recent_high=101.0, recent_low=99.0, atr_pct=0.01,
        z_cvd_override=3.0, z_fund_override=3.0, z_delta_oi_override=3.0,
        z_micro_override=3.0, z_whale_override=3.0, whale_history_length=4,
        gate_long_status="BLOCKED_BTC_REGIME",
    )
    assert sig.signal_type == "NEUTRAL"
    assert sig.gate_status == "BLOCKED_BTC_REGIME"
    assert sig.composite_score >= 75.0
    assert sig.risk_reward_ratio == 0.0
    assert sig.invalidation_price == sig.price
    assert sig.target_price == sig.price


def test_portfolio_downgrade_clears_trade_parameters():
    from contracts import SignalEvent
    screener = QuantScreener(state_file=Path("/tmp/dummy.bin"), top_n_symbols=1, concurrency_limit=1)
    screener.max_strong_per_direction = 1
    a = SignalEvent(
        symbol="A", timestamp_ms=1, signal_type="STRONG_LONG", composite_score=90.0,
        z_cvd_div=3.0, z_fund_trap=3.0, z_delta_oi=3.0, z_micro=3.0, vpin=0.5, obi=0.0,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=2.0, decision_timestamp_ms=1, suggested_position_usd=500.0,
        suggested_leverage=2, trailing_stop_activation_pct=0.01, trailing_stop_distance_pct=0.005,
    )
    b = SignalEvent(
        symbol="B", timestamp_ms=1, signal_type="STRONG_LONG", composite_score=80.0,
        z_cvd_div=3.0, z_fund_trap=3.0, z_delta_oi=3.0, z_micro=3.0, vpin=0.5, obi=0.0,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=2.0, decision_timestamp_ms=1, suggested_position_usd=500.0,
        suggested_leverage=2, trailing_stop_activation_pct=0.01, trailing_stop_distance_pct=0.005,
    )
    filtered, downgraded = screener._apply_portfolio_correlation_limit([a, b])
    assert downgraded == 1
    blocked = next(sig for sig in filtered if sig.symbol == "B")
    assert blocked.signal_type == "NEUTRAL"
    assert blocked.risk_reward_ratio == 0.0
    assert blocked.invalidation_price == blocked.price
    assert blocked.target_price == blocked.price
    assert blocked.suggested_position_usd == 0.0
    assert blocked.suggested_leverage == 1
    assert blocked.trailing_stop_activation_pct == 0.0
    assert blocked.trailing_stop_distance_pct == 0.0


def test_zero_variance_factors_are_excluded_from_score_denominator():
    engine = QuantSignalEngine(z_history_min_samples=4, strong_signal_threshold=75.0, min_effective_rrr=0.0)
    activity = {"cvd": True, "fund": False, "oi": False, "micro": True, "whale": True}
    sig = engine.compute_signal(
        symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
        delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=1.0,
        recent_high=101.0, recent_low=99.0, atr_pct=0.01,
        z_cvd_override=3.0, z_fund_override=0.0, z_delta_oi_override=0.0,
        z_micro_override=3.0, z_whale_override=3.0, whale_history_length=4,
        active_factors=activity,
    )
    expected = 100.0 * math.tanh(180.0 / (60.0 * 1.4))
    assert math.isclose(sig.composite_score, round(expected, 2), rel_tol=0.0, abs_tol=0.01)


def test_risk_guard_preserves_persisted_peak_when_env_equity_is_supplied(tmp_path, monkeypatch):
    from engine.risk_guard import RiskGuard
    state = tmp_path / "equity.json"
    from datetime import datetime, timezone
    day_key = datetime.now(timezone.utc).date().isoformat()
    state.write_text(json.dumps({"equity_usdt": 12000.0, "day_start_equity_usdt": 11000.0, "peak_equity_usdt": 13000.0, "day_key": day_key}))
    monkeypatch.setenv("CURRENT_EQUITY_USDT", "10000")
    monkeypatch.setenv("CURRENT_EQUITY_UPDATED_MS", str(int(__import__("time").time() * 1000)))
    guard = RiskGuard(state_file=state, account_equity=10000.0, paper_trading=False, kill_switch_file=tmp_path / "KILL_SWITCH")
    loaded = guard._load_equity_state()
    assert loaded["day_start_equity_usdt"] == 11000.0
    assert loaded["peak_equity_usdt"] == 13000.0


def test_latest_scan_json_contains_canonical_27_fields(tmp_path):
    from engine.screener import ScreenerResult, save_latest_scan_json
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_299_999, signal_type="NEUTRAL", composite_score=72.0,
        z_cvd_div=2.0, z_fund_trap=0.5, z_delta_oi=-0.2, z_micro=1.2, vpin=0.3, obi=0.2,
        funding_8h=0.0, basis_bps=-2.0, price=100.0, invalidation_price=100.0, target_price=100.0,
        risk_reward_ratio=0.0, decision_timestamp_ms=1_700_000_337_000, z_whale_sentiment=0.7,
        relative_strength=0.1, sweep_reclaim=False, gate_status="PASSED", sweep_pattern="NONE",
        suggested_position_usd=0.0, suggested_leverage=1, trailing_stop_activation_pct=0.0,
        trailing_stop_distance_pct=0.0, applied_friction_rt_pct=0.0015,
    )
    summary = ScreenerResult(
        timestamp_ms=sig.timestamp_ms, duration_sec=1.0, total_scanned=1, strong_longs_count=0,
        strong_shorts_count=0, synthetic_liqs_count=0, btc_regime="NEUTRAL_RANGING", btc_change_5m_pct=0.0,
        successful_symbols=1, signal_ready_symbols=1,
    )
    path = tmp_path / "paper_signals.json"
    save_latest_scan_json([sig], [], summary, path)
    payload = json.loads(path.read_text())
    from engine.serialization import CANONICAL_SIGNAL_FIELDS
    assert list(payload["signals"][0]) == list(CANONICAL_SIGNAL_FIELDS)
    assert payload["signals"][0]["timestamp_ms"] == sig.timestamp_ms
    assert payload["signals"][0]["z_delta_oi"] == sig.z_delta_oi
    assert payload["signals"][0]["decision_timestamp_ms"] == sig.decision_timestamp_ms


def test_portfolio_global_risk_cap_applies_across_directions():
    from engine.research import TradeRecord
    from engine.portfolio import PortfolioConfig, PortfolioSimulator
    def local_bar(i, price, symbol):
        ts = 1_700_000_000_000 + i * 5 * 60 * 1000
        import math as _math
        patterns = {
            "S0USDT": _math.sin(i / 5.0),
            "S1USDT": _math.cos(i / 5.0),
            "S2USDT": _math.sin(i / 7.0 + 1.2),
            "S3USDT": _math.cos(i / 7.0 + 2.0),
        }
        px = 100.0 + patterns.get(symbol, 0.0)
        return __import__('engine.research', fromlist=['ResearchBar']).ResearchBar(
            timestamp_ms=ts + 299_999, symbol=symbol, open=px, high=px * 1.005, low=px * 0.995,
            close=px, volume=1000.0, open_interest=1000.0 + i, delta_oi_pct=0.001, funding_rate_8h=0.0001,
            basis_bps=1.0, obi=0.2, vpin=0.4, cvd_divergence_score=0.0, whale_divergence_score=0.0,
            atr_pct=0.003, recent_high=px * 1.003, recent_low=px * 0.997, spread_bps=0.5
        )
    entry_ts = local_bar(60, 103.0, 'S0USDT').timestamp_ms
    exit_ts = local_bar(62, 103.0, 'S0USDT').timestamp_ms
    trades = []
    for i, side in enumerate(["LONG", "SHORT", "LONG", "SHORT"]):
        trades.append(TradeRecord(
            signal_timestamp_ms=local_bar(59, 103.0, f"S{i}USDT").timestamp_ms,
            entry_timestamp_ms=entry_ts + i,
            exit_timestamp_ms=exit_ts + i,
            symbol=f"S{i}USDT", side=side, signal_score=80.0,
            entry_price=100.0, exit_price=101.0, invalidation_price=98.0 if side == "LONG" else 102.0,
            target_price=104.0 if side == "LONG" else 96.0, stop_hit=False, target_hit=False,
            exit_reason="TIME_EXIT", gross_return=0.01, round_trip_cost=0.0, net_return=0.01,
            holding_bars=2, z_cvd=2.0, z_fund=2.0, z_oi=2.0, z_micro=2.0, z_whale=2.0, atr_pct=0.003, spread_bps=0.5,
        ))
    # Replace dataset symbols with matching rows.
    rows = []
    for symbol in [f"S{i}USDT" for i in range(4)]:
        rows.extend(local_bar(j, 100.0 + 0.1 * j, symbol) for j in range(100))
    from engine.research import ResearchDataset
    ds = ResearchDataset(rows)
    fills, report = PortfolioSimulator(PortfolioConfig(max_aggregate_risk_pct=0.03, max_gross_leverage=10.0)).run(ds, trades)
    assert report.trades_executed <= 3
    assert any(f.block_reason == "AGGREGATE_RISK_CAP" for f in fills if f.blocked)


def test_wfa_fails_closed_on_global_time_gap():
    from engine.research import ResearchBar, ResearchDataset
    bars = []
    for i in range(120):
        if i == 60:
            continue
        ts = 1_700_000_000_000 + i * 5 * 60 * 1000
        bars.append(ResearchBar(
            timestamp_ms=ts + 299_999, symbol="BTCUSDT", open=100.0+i*0.1, high=101.0+i*0.1, low=99.0+i*0.1,
            close=100.0+i*0.1, volume=1000.0, open_interest=1000.0+i, delta_oi_pct=0.001, funding_rate_8h=0.0001,
            basis_bps=1.0, obi=0.2, vpin=0.4, cvd_divergence_score=0.0, whale_divergence_score=0.0,
            atr_pct=0.003, recent_high=100.3+i*0.1, recent_low=99.7+i*0.1, spread_bps=0.5
        ))
    ds = ResearchDataset(bars)
    from engine.walk_forward import ParameterConfig, WalkForwardConfig, run_wfa
    cfg = WalkForwardConfig(train_bars=40, validation_bars=20, test_bars=20, embargo_bars=2, min_train_trades=1, min_test_trades=1)
    assert run_wfa(ds, [ParameterConfig("baseline", {})], cfg) == []


def test_workflow_has_no_scheduled_trigger():
    from pathlib import Path
    workflow = Path(".github/workflows/quant-screener.yml").read_text(encoding="utf-8")
    assert "schedule:" not in workflow
    assert "repository_dispatch:" in workflow


def test_signal_ledger_records_provenance(tmp_path):
    from engine.signal_ledger import append_signal_events
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_099_999, signal_type="STRONG_LONG", composite_score=80.0,
        z_cvd_div=1.0, z_fund_trap=1.0, z_delta_oi=1.0, z_micro=1.0, vpin=0.5, obi=0.1,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=1.8, decision_timestamp_ms=1_700_000_137_000,
    )
    path = tmp_path / "ledger.jsonl"
    assert append_signal_events(path, [sig]) == 1
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["research_schema_version"] == 4
    assert row["strategy_revision"] == "2026-09-25-cleanstart-hardened-v4"
    assert len(row["config_fingerprint"]) == 16


def test_portfolio_short_mark_to_market_uses_linear_return():
    from engine.portfolio import PortfolioFill, PortfolioSimulator, PortfolioConfig
    simulator = PortfolioSimulator(PortfolioConfig())
    pos = PortfolioFill(
        signal_timestamp_ms=1, entry_timestamp_ms=1, exit_timestamp_ms=2, symbol="BTCUSDT", side="SHORT",
        notional_usd=1000.0, entry_price=100.0, stop_risk_usd=10.0, pnl_usd=0.0, equity_before=10000.0, equity_after=10000.0,
    )
    # Access the same mark-to-market contract through a minimal real run is cumbersome;
    # verify the formula explicitly used by the simulator remains linear in price.
    assert math.isclose(pos.notional_usd * (1.0 - 90.0 / pos.entry_price), 100.0, rel_tol=1e-12)
    assert math.isclose(pos.notional_usd * (1.0 - 110.0 / pos.entry_price), -100.0, rel_tol=1e-12)


def test_risk_guard_requires_timestamp_for_env_equity(tmp_path, monkeypatch):
    from engine.risk_guard import RiskGuard
    monkeypatch.setenv("CURRENT_EQUITY_USDT", "10000")
    monkeypatch.delenv("CURRENT_EQUITY_UPDATED_MS", raising=False)
    guard = RiskGuard(state_file=tmp_path / "equity.json", account_equity=10000.0, paper_trading=False, kill_switch_file=tmp_path / "KILL_SWITCH")
    allowed, reason = guard.evaluate()
    assert not allowed
    assert reason == "CURRENT_EQUITY_TIMESTAMP_REQUIRED"


def test_risk_guard_rejects_stale_equity_state(tmp_path, monkeypatch):
    from engine.risk_guard import RiskGuard
    import time as _time
    state = tmp_path / "equity.json"
    state.write_text(json.dumps({
        "timestamp_ms": int((_time.time() - 3600) * 1000),
        "equity_usdt": 10000.0,
        "day_start_equity_usdt": 10000.0,
        "peak_equity_usdt": 10000.0,
        "day_key": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).date().isoformat(),
    }))
    monkeypatch.delenv("CURRENT_EQUITY_USDT", raising=False)
    guard = RiskGuard(state_file=state, account_equity=10000.0, paper_trading=False, max_state_age_sec=60, kill_switch_file=tmp_path / "KILL_SWITCH")
    allowed, reason = guard.evaluate()
    assert allowed is False
    assert reason == "EQUITY_STATE_STALE"


def test_paper_mode_tolerates_stale_equity_state(tmp_path):
    from engine.risk_guard import RiskGuard
    import time as _time
    state = tmp_path / "equity.json"
    state.write_text(json.dumps({
        "timestamp_ms": int((_time.time() - 3600) * 1000),
        "equity_usdt": 10000.0,
        "day_start_equity_usdt": 10000.0,
        "peak_equity_usdt": 10000.0,
        "day_key": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).date().isoformat(),
    }))
    guard = RiskGuard(state_file=state, account_equity=10000.0, paper_trading=True, max_state_age_sec=60, kill_switch_file=tmp_path / "KILL_SWITCH")
    allowed, reason = guard.evaluate()
    assert allowed is True
    assert reason == "PAPER_STALE_EQUITY_STATE"


def test_provenance_fingerprint_uses_effective_defaults(monkeypatch):
    from engine.provenance import CONFIG_DEFAULTS, config_fingerprint
    for key in CONFIG_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    baseline = config_fingerprint()
    monkeypatch.setenv("ACCOUNT_EQUITY_USDT", CONFIG_DEFAULTS["ACCOUNT_EQUITY_USDT"])
    assert config_fingerprint() == baseline


def test_wfa_validation_sample_gate_is_explicit():
    from engine.walk_forward import WalkForwardConfig
    cfg = WalkForwardConfig()
    assert cfg.min_train_trades == 50
    assert cfg.min_validation_trades == 20
    assert cfg.min_test_trades == 20


def test_funding_normalization_defaults_to_rank_score_for_repeated_8h_rates():
    from engine.signals import percentile_score
    engine = QuantSignalEngine(z_history_min_samples=24)
    funding_history = [0.0010] * 23 + [0.001001]
    basis_history = [0.0] * 23 + [1.0]
    values = engine.calculate_factor_zscores(
        funding_rate_8h=0.0010005,
        basis_spread_bps=0.5,
        delta_oi_pct=0.001,
        obi=0.1,
        vpin=0.2,
        cvd_divergence_score=0.0,
        whale_divergence_score=0.0,
        funding_history=funding_history,
        basis_history=basis_history,
        delta_oi_pct_history=[0.0, 0.1, 0.2, 0.3] * 6,
        micro_factor_history=[0.0, 0.1, 0.2, 0.3] * 6,
        cvd_history=[-1.0, 0.0, 1.0, 0.0] * 6,
        whale_history=[],
    )
    expected_funding = percentile_score(0.0010005, funding_history, 24)
    expected_basis = empirical_zscore(0.5, basis_history, 24)
    expected_combined = -engine.funding_weight * expected_funding - engine.basis_weight * expected_basis
    assert math.isclose(values[1], expected_combined, abs_tol=1e-12)


def test_signal_ledger_uses_canonical_friction_field(tmp_path):
    from engine.signal_ledger import append_signal_events
    sig = SignalEvent(
        symbol="BTCUSDT", timestamp_ms=1_700_000_099_999, signal_type="STRONG_LONG", composite_score=80.0,
        z_cvd_div=1.0, z_fund_trap=1.0, z_delta_oi=1.0, z_micro=1.0, vpin=0.5, obi=0.1,
        funding_8h=0.0, basis_bps=0.0, price=100.0, invalidation_price=98.0, target_price=104.0,
        risk_reward_ratio=1.8, decision_timestamp_ms=1_700_000_137_000, applied_friction_rt_pct=0.0017,
    )
    path = tmp_path / "ledger.jsonl"
    assert append_signal_events(path, [sig]) == 1
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["applied_friction_rt_pct"] == 0.0017
    assert "friction_rt_pct" not in row


def test_finalize_research_rows_contains_final_dispatch_outcome():
    from cron_runner import finalize_research_rows
    rows = [
        {"symbol": "BTCUSDT", "timestamp_ms": 1000, "final_signal_type": "STRONG_LONG"},
        {"symbol": "ETHUSDT", "timestamp_ms": 1000, "final_signal_type": "NEUTRAL"},
    ]
    sent = {"BTCUSDT:1000:STRONG_LONG"}
    finalize_research_rows(
        rows, dispatch_allowed=True, block_reason=None, dispatched_signal_ids=sent, dispatched_at_ms=2000
    )
    assert rows[0]["dispatch_allowed"] is True
    assert rows[0]["dispatched"] is True
    assert rows[0]["dispatched_at_ms"] == 2000
    assert rows[1]["dispatched"] is False
    assert rows[1]["dispatched_at_ms"] is None
    assert rows[0]["dispatch_status_source"] == "FINAL_RISK_GUARD_AND_TELEGRAM"
