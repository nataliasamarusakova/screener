"""Regression tests for the production quant/risk paths.

The tests intentionally assert concrete values and failure modes rather than
implementation ranges. They cover the Phase-1 root causes fixed in Phase 2.
"""
import asyncio
import json
import math

import numpy as np
import pytest

from binance_ingestion import BinanceFuturesIngestion, BinanceOrderBookFSM, OrderBookFSMState
from contracts import MarketStateSnapshot
from engine.divergence import detect_cvd_divergence_jit
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.microstructure_jit import compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.sentiment import SentimentEngine
from engine.signals import QuantSignalEngine, empirical_zscore
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
            return [{"a": 1, "T": 1700000100000, "q": "2.0", "m": True}]

    trades = asyncio.run(StubIngestion([]).fetch_5m_agg_trades(
        "BTCUSDT", 1700000000000, 1700000299999
    ))
    assert trades is not None
    assert len(trades) == 1
    assert trades[0]["T"] == 1700000100000
    assert trades[0]["a"] == 1


def test_explicit_5m_agg_trade_window_rejects_malformed_missing_trade_id():
    class StubIngestion(BinanceFuturesIngestion):
        async def _request_json(self, method, url, *, params=None, weight, symbol=""):
            return [{"T": 1700000100000, "q": "2.0", "m": True}]

    trades = asyncio.run(StubIngestion([]).fetch_5m_agg_trades(
        "BTCUSDT", 1700000000000, 1700000299999
    ))
    assert trades is None


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
    assert gapped_beta is None






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




def test_signal_rejects_nonfinite_z_override():
    engine = QuantSignalEngine()
    with pytest.raises(ValueError, match="Non-finite signal input"):
        engine.compute_signal(
            symbol="BTCUSDT", current_price=100.0, funding_rate_8h=0.0, basis_spread_bps=0.0,
            delta_oi=0.0, oi_total=1000.0, obi=0.0, vpin=0.5, cvd_divergence_score=0.0,
            recent_high=101.0, recent_low=99.0, z_cvd_override=float("nan"), z_fund_override=0.0,
            z_delta_oi_override=0.0, z_micro_override=0.0, z_whale_override=0.0, atr_pct=1.0,
        )








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






def _make_5m_kline(open_ms: int, open_price: float, high: float, low: float, close: float, volume: float = 10.0, taker_buy: float = 5.0):
    return [
        open_ms, str(open_price), str(high), str(low), str(close), str(volume),
        open_ms + 299_999, str(volume), 1, str(taker_buy), str(taker_buy), "0",
    ]


def test_atr_pct_uses_previous_close_for_first_true_range():
    base = 1_700_000_000_000
    step = 5 * 60 * 1000
    klines = [
        _make_5m_kline(base, 100.0, 101.0, 99.0, 100.0),
        _make_5m_kline(base + step, 115.0, 120.0, 110.0, 115.0),
        _make_5m_kline(base + 2 * step, 115.0, 116.0, 114.0, 114.5),
    ]
    atr_pct = __import__("engine.screener", fromlist=["QuantScreener"]).QuantScreener._atr_pct(klines, lookback=2)
    assert atr_pct == pytest.approx(11.0 / 114.5, rel=0.0, abs=1e-12)


def test_btc_regime_threshold_uses_unrounded_5m_change():
    engine = MarketRegimeEngine()
    regime = engine.evaluate_btc_regime(99.701, 100.0, 0.0)
    assert regime.status == "NEUTRAL_RANGING"
    assert regime.btc_change_5m_pct == pytest.approx(-0.299, abs=1e-12)


def test_rolling_beta_rejects_gap_inside_latest_window():
    step = 5 * 60 * 1000
    times = [1_700_000_000_000 + i * step for i in range(30)]
    btc = [60_000.0]
    alt = [3_000.0]
    returns = [0.0005, -0.0002, 0.0008, 0.0001, -0.0004] * 6
    for r in returns:
        btc.append(btc[-1] * (1.0 + r))
        alt.append(alt[-1] * (1.0 + 2.0 * r))
    gapped = times.copy()
    gapped[15] += step
    beta = MarketRegimeEngine.calculate_rolling_beta(gapped, alt, times, btc, min_samples=24)
    assert beta is None


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class _FakeHTTPSession:
    closed = False

    def __init__(self, responses):
        self.responses = responses

    def get(self, url, params=None):
        if "globalLongShortAccountRatio" in url:
            return _FakeHTTPResponse(self.responses["retail"])
        if "topLongShortPositionRatio" in url:
            return _FakeHTTPResponse(self.responses["whale"])
        return _FakeHTTPResponse(self.responses["taker"])



def test_sentiment_accepts_endpoint_specific_5m_timestamp_semantics():
    start = 1_700_000_000_000
    end = start + 300_000
    responses = {
        "retail": [{"timestamp": end, "longShortRatio": "1.2", "longAccount": "0.55"}],
        "whale": [{"timestamp": end, "longShortRatio": "1.5", "longAccount": "0.60"}],
        "taker": [{"timestamp": start, "buySellRatio": "1.1"}],
    }
    engine = SentimentEngine(session=_FakeHTTPSession(responses), api_key="test")
    result = asyncio.run(engine.fetch_sentiment_divergence("BTCUSDT", start_time_ms=start, end_time_ms=end))
    assert result is not None
    assert result.observation_timestamp_ms == end
    assert result.retail_ls_ratio == pytest.approx(1.2, abs=1e-12)


def test_sentiment_rejects_wrong_endpoint_boundary_timestamps():
    start = 1_700_000_000_000
    end = start + 300_000
    responses = {
        "retail": [{"timestamp": start, "longShortRatio": "1.2", "longAccount": "0.55"}],
        "whale": [{"timestamp": end, "longShortRatio": "1.5", "longAccount": "0.60"}],
        "taker": [{"timestamp": start, "buySellRatio": "1.1"}],
    }
    engine = SentimentEngine(session=_FakeHTTPSession(responses), api_key="test")
    result = asyncio.run(engine.fetch_sentiment_divergence("BTCUSDT", start_time_ms=start, end_time_ms=end))
    assert result is None


def test_rate_limiter_rejects_weight_larger_than_budget():
    limiter = __import__("binance_ingestion", fromlist=["WeightedRateLimiter"]).WeightedRateLimiter(max_weight=10)
    with pytest.raises(ValueError, match="exceeds limiter budget"):
        asyncio.run(limiter.acquire(11))


def test_screener_rejects_nonpositive_runtime_limits():
    from engine.screener import QuantScreener
    with pytest.raises(ValueError, match="SCAN_CONCURRENCY"):
        QuantScreener(concurrency_limit=0)
    with pytest.raises(ValueError, match="TOP_N_SYMBOLS"):
        QuantScreener(top_n_symbols=0)


def test_telegram_cache_skips_malformed_records(tmp_path, monkeypatch):
    from engine.telegram import TelegramAlerter
    import engine.telegram as telegram_module
    cache_path = tmp_path / ".alert_cache.json"
    cache_path.write_text(json.dumps({
        "BTCUSDT": {"time": 100.0, "score": 80.0},
        "BROKEN": {"time": "not-a-number", "score": 80.0},
        "MISSING": {"time": 100.0},
    }), encoding="utf-8")
    monkeypatch.setattr(telegram_module, "ALERT_CACHE_FILE", cache_path)
    alerter = TelegramAlerter(cooldown_sec=3600)
    assert alerter.cache == {"BTCUSDT": {"time": 100.0, "score": 80.0}}



def test_orderbook_snapshot_mismatch_resyncs_without_physical_disconnect():
    async def scenario():
        ingestion = BinanceFuturesIngestion(["BTCUSDT"])
        ingestion._running = True
        calls = 0

        async def fake_snapshot(symbol, limit=1000):
            nonlocal calls
            calls += 1
            return {
                "lastUpdateId": 100,
                "bids": [["60000", "1.0"]],
                "asks": [["60001", "1.0"]],
            }

        ingestion.fetch_l2_snapshot = fake_snapshot
        book = ingestion.books["BTCUSDT"]
        book.on_ws_connected()
        task = asyncio.create_task(ingestion._sync_symbol_book("BTCUSDT"))
        await asyncio.sleep(0)
        book.handle_depth_event({"e": "depthUpdate", "E": 100, "s": "BTCUSDT", "U": 90, "u": 100, "pu": 89, "b": [], "a": []})
        await asyncio.sleep(0.35)
        assert book.state == OrderBookFSMState.BUFFERING
        assert calls >= 1
        book.handle_depth_event({"e": "depthUpdate", "E": 101, "s": "BTCUSDT", "U": 101, "u": 105, "pu": 100, "b": [["60000", "2.0"]], "a": [["60001", "2.0"]]})
        await asyncio.wait_for(task, timeout=2.0)
        assert book.state == OrderBookFSMState.IN_SYNC
        assert calls == 2
        await ingestion.stop()

    asyncio.run(scenario())


def test_quality_filter_rejects_nonfinite_configuration():
    with pytest.raises(ValueError, match="finite"):
        QualityFilter(max_spread_bps=float("nan"))
    with pytest.raises(ValueError, match="invalid bounds"):
        QualityFilter(max_abs_funding_rate_8h=0.0)


def test_funding_filter_rejects_nonfinite_threshold_configuration():
    with pytest.raises(ValueError, match="finite and positive"):
        FundingFilterEngine(proximity_threshold_minutes=float("nan"))


def test_market_regime_rejects_nonfinite_or_reversed_thresholds():
    with pytest.raises(ValueError, match="finite"):
        MarketRegimeEngine(dump_threshold_5m_pct=float("nan"))
    with pytest.raises(ValueError, match="invalid bounds"):
        MarketRegimeEngine(dump_threshold_5m_pct=0.3, pump_threshold_5m_pct=-0.3)


def test_ingestion_rejects_nonfinite_rate_limit_backoff(monkeypatch):
    monkeypatch.setenv("BINANCE_429_BACKOFF_SEC", "nan")
    with pytest.raises(ValueError, match="BINANCE_429_BACKOFF_SEC"):
        BinanceFuturesIngestion([])


def test_screener_offline_single_symbol_scan_persists_closed_candle(tmp_path):
    from types import SimpleNamespace

    class FakeIngestion:
        def __init__(self):
            self.stopped = False

        async def fetch_universe_tickers(self):
            return {
                "BTCUSDT": {
                    "symbol": "BTCUSDT", "quoteVolume": "100000000",
                    "highPrice": "102", "lowPrice": "98", "priceChangePercent": "0.2",
                }
            }

        async def fetch_universe_premium_index(self):
            return {
                "BTCUSDT": {
                    "symbol": "BTCUSDT", "lastFundingRate": "0.0001",
                    "markPrice": "100", "indexPrice": "99.99",
                    "nextFundingTime": 9_999_999_999_999,
                }
            }

        async def fetch_universe_funding_info(self):
            return {}

        async def fetch_symbol_closed_5m_klines(self, symbol, closed_open_ms, *, history_bars):
            first = closed_open_ms - (history_bars - 1) * 300_000
            rows = []
            for i in range(history_bars):
                t = first + i * 300_000
                rows.append([t, "99.5", "100.5", "99.0", "100.0", "10", t + 299_999, "1000", "10", "5", "500", "0"])
            return rows

        async def fetch_symbol_closed_5m_open_interest(self, symbol, current_open_ms):
            return 1000.0

        async def fetch_symbol_orderbook_top(self, symbol, limit=20):
            return SimpleNamespace(
                bids=((99.99, 10.0), (99.98, 5.0)),
                asks=((100.00, 10.0), (100.01, 5.0)),
                spread_bps=1.5,
            )

        async def fetch_5m_agg_trades(self, symbol, start_time_ms, end_time_ms):
            return [{"a": 1, "T": start_time_ms, "q": 10.0, "m": False}]

        async def stop(self):
            self.stopped = True

    class FakeSentiment:
        async def fetch_sentiment_divergence(self, symbol, start_time_ms=None, end_time_ms=None):
            return SimpleNamespace(divergence_score=0.0)

        async def close(self):
            pass

    async def run():
        screener = QuantScreener(state_file=tmp_path / "state.bin", top_n_symbols=1, concurrency_limit=1)
        screener.ingestion = FakeIngestion()
        screener.sentiment_engine = FakeSentiment()
        _, _, summary = await screener.scan()
        assert summary.total_scanned == 1
        assert summary.successful_symbols == 1
        assert summary.failed_symbols == 0
        assert summary.signal_ready_symbols == 0
        state = screener.load_previous_state()
        assert state["BTCUSDT"].candle_open_time_ms == summary.timestamp_ms + 1 - 300_000
        assert screener.ingestion.stopped is True

    asyncio.run(run())


def _minimal_snapshot(symbol: str, candle_open_time_ms: int) -> MarketStateSnapshot:
    return MarketStateSnapshot(
        symbol=symbol,
        timestamp_ms=candle_open_time_ms + 299999,
        last_price=100.0,
        open_interest=1000.0,
        delta_oi_5m=0.0,
        cumulative_cvd_5m=0.0,
        funding_rate_8h=0.0,
        basis_bps=0.0,
        vpin_estimate=0.2,
        obi_score=0.0,
        composite_score=0.0,
        candle_open_time_ms=candle_open_time_ms,
    )


def test_screener_contiguous_state_helper_rejects_stale_btc_snapshot():
    expected = 1_700_000_000_000
    fresh = _minimal_snapshot("BTCUSDT", expected)
    stale = _minimal_snapshot("BTCUSDT", expected - 300_000)
    assert QuantScreener._state_is_for_candle(fresh, expected) is True
    assert QuantScreener._state_is_for_candle(stale, expected) is False
    assert QuantScreener._state_is_for_candle(None, expected) is False


def test_signal_rejects_nonfinite_configuration():
    with pytest.raises(ValueError, match="finite"):
        QuantSignalEngine(score_tanh_scale=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        QuantSignalEngine(w_cvd=-1.0)
    with pytest.raises(ValueError, match="strong_signal_threshold"):
        QuantSignalEngine(strong_signal_threshold=101.0)
