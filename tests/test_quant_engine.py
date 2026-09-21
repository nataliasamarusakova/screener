# --- NEW tests to add ---

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
