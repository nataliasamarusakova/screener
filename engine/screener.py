"""
Production 5-minute Quant Screener.
Fail-closed pipeline with robust lag recovery, friction-adjusted scoring,
per-symbol cost model, portfolio correlation limit, and position sizing.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import msgspec
import numpy as np

from binance_ingestion import BinanceFuturesIngestion
from contracts import MarketStateSnapshot, SignalEvent, SyntheticLiquidation
from engine.divergence import detect_cvd_divergence_jit
from engine.funding_filter import FundingFilterEngine
from engine.liquidations import SyntheticLiquidationDetector
from engine.liquidity_sweep import LiquiditySweepDetector
from engine.market_regime import MarketRegimeEngine
from engine.microstructure_jit import VPIN_ESTIMATE_METHOD, compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.sentiment import SentimentEngine
from engine.signals import QuantSignalEngine
from engine.research_recorder import ResearchRecorder
from engine.provenance import provenance
from engine.serialization import signal_to_dict

logger = logging.getLogger("screener")


class ScreenerResult(msgspec.Struct, gc=False):
    timestamp_ms: int
    duration_sec: float
    total_scanned: int
    strong_longs_count: int
    strong_shorts_count: int
    synthetic_liqs_count: int
    btc_regime: str
    btc_change_5m_pct: float
    successful_symbols: int = 0
    rejected_symbols: int = 0
    failed_symbols: int = 0
    signal_ready_symbols: int = 0
    portfolio_limited_symbols: int = 0
    signal_readiness_reasons: Dict[str, int] = msgspec.field(default_factory=dict)
    stale_state_symbols_dropped: int = 0
    state_recovery_sources: Dict[str, int] = msgspec.field(default_factory=dict)


def save_latest_scan_json(
    signals: List[SignalEvent],
    synthetic_liqs: List[SyntheticLiquidation],
    summary: ScreenerResult,
    target_path: Path = Path("data/signals_latest.json"),
) -> None:
    payload = {
        "summary": {
            "timestamp_ms": summary.timestamp_ms,
            "duration_sec": summary.duration_sec,
            "total_scanned": summary.total_scanned,
            "strong_longs": summary.strong_longs_count,
            "strong_shorts": summary.strong_shorts_count,
            "synthetic_liqs": summary.synthetic_liqs_count,
            "btc_regime": summary.btc_regime,
            "btc_change_5m_pct": summary.btc_change_5m_pct,
            "successful_symbols": summary.successful_symbols,
            "rejected_symbols": summary.rejected_symbols,
            "failed_symbols": summary.failed_symbols,
            "signal_ready_symbols": summary.signal_ready_symbols,
            "portfolio_limited_symbols": summary.portfolio_limited_symbols,
            "signal_readiness_reasons": summary.signal_readiness_reasons,
            "stale_state_symbols_dropped": summary.stale_state_symbols_dropped,
            "state_recovery_sources": summary.state_recovery_sources,
        },
        "signal_schema_version": 2,
        "provenance": provenance(),
        "signals": [signal_to_dict(s) for s in signals],
        "liquidations": [
            {
                "symbol": lq.symbol,
                "side": lq.side,
                "price": lq.price,
                "delta_oi": lq.delta_oi,
                "taker_volume": lq.taker_volume,
                "anomaly_ratio": lq.anomaly_ratio,
                "estimated_volume": lq.estimated_liquidation_volume,
            }
            for lq in synthetic_liqs
        ],
    }
    raw = msgspec.json.encode(payload)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as temp:
            temp.write(raw)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, target_path)
        dir_fd = os.open(target_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class QuantScreener:
    def __init__(
        self,
        state_file: Path = Path("data/market_state.bin"),
        concurrency_limit: int = 25,
        top_n_symbols: int = 100,
    ) -> None:
        self.state_file = state_file
        self._provenance = provenance()
        self.concurrency_limit = concurrency_limit
        self.top_n_symbols = top_n_symbols
        self.history_bars = int(os.getenv("SIGNAL_HISTORY_BARS", "60"))
        self.z_history_min_samples = int(os.getenv("SIGNAL_Z_MIN_SAMPLES", "24"))
        self.cvd_lookback = int(os.getenv("CVD_LOOKBACK_BARS", "8"))
        self.vpin_window_baskets = int(os.getenv("VPIN_WINDOW_BASKETS", "10"))
        self.beta_min_samples = int(os.getenv("BETA_MIN_SAMPLES", "24"))
        self.atr_lookback = int(os.getenv("ATR_LOOKBACK_BARS", "12"))
        self.obi_decay = float(os.getenv("OBI_DECAY", "0.85"))
        self.failure_ratio_limit = float(os.getenv("SCAN_FAILURE_RATIO_LIMIT", "0.10"))
        self.account_equity = float(os.getenv("ACCOUNT_EQUITY_USDT", "10000.0"))
        # NEW: correlation limit
        self.max_strong_per_direction = int(os.getenv("MAX_STRONG_PER_DIRECTION", "3"))
        self.max_portfolio_correlation = float(os.getenv("MAX_PORTFOLIO_CORRELATION", "0.80"))
        self.max_aggregate_risk_pct = float(os.getenv("MAX_AGGREGATE_RISK_PCT", "0.03"))
        self.max_gross_leverage = float(os.getenv("MAX_GROSS_LEVERAGE", "3.0"))
        self.funding_proximity_minutes = float(os.getenv("FUNDING_PROXIMITY_MINUTES", "20.0"))
        self.funding_extreme_threshold_8h = float(os.getenv("FUNDING_EXTREME_THRESHOLD_8H", "0.0010"))

        if self.concurrency_limit < 1:
            raise ValueError("SCAN_CONCURRENCY must be >= 1")
        if self.top_n_symbols < 1:
            raise ValueError("TOP_N_SYMBOLS must be >= 1")
        if self.history_bars <= self.z_history_min_samples:
            raise ValueError("SIGNAL_HISTORY_BARS must be greater than SIGNAL_Z_MIN_SAMPLES")
        if self.max_strong_per_direction < 1:
            raise ValueError("MAX_STRONG_PER_DIRECTION must be >= 1")
        if not math.isfinite(self.max_portfolio_correlation) or not (0.0 < self.max_portfolio_correlation <= 1.0):
            raise ValueError("MAX_PORTFOLIO_CORRELATION must be in (0, 1]")
        if not math.isfinite(self.max_aggregate_risk_pct) or not (0.0 < self.max_aggregate_risk_pct <= 1.0):
            raise ValueError("MAX_AGGREGATE_RISK_PCT must be in (0, 1]")
        if not math.isfinite(self.max_gross_leverage) or self.max_gross_leverage < 1.0:
            raise ValueError("MAX_GROSS_LEVERAGE must be >= 1")
        if not math.isfinite(self.funding_proximity_minutes) or self.funding_proximity_minutes <= 0.0:
            raise ValueError("FUNDING_PROXIMITY_MINUTES must be positive")
        if not math.isfinite(self.funding_extreme_threshold_8h) or self.funding_extreme_threshold_8h <= 0.0:
            raise ValueError("FUNDING_EXTREME_THRESHOLD_8H must be positive")
        if not math.isfinite(self.obi_decay) or not (0.0 < self.obi_decay <= 1.0):
            raise ValueError("OBI_DECAY must be in (0, 1]")

        self.ingestion = BinanceFuturesIngestion(symbols=[])
        # FIX: read min_effective_rrr and friction from env
        self.research_recorder = ResearchRecorder(Path(os.getenv("RESEARCH_DB", "data/research/features.sqlite3")))
        self.signal_engine = QuantSignalEngine(
            z_history_min_samples=self.z_history_min_samples,
            min_effective_rrr=float(os.getenv("MIN_EFFECTIVE_RRR", "1.30")),
            friction_round_trip_pct=float(os.getenv("FRICTION_RT_PCT", "0.0018")),
            atr_stop_multiplier=float(os.getenv("ATR_STOP_MULTIPLIER", "1.50")),
            max_atr_multiplier=float(os.getenv("MAX_ATR_MULTIPLIER", "2.50")),
            sweep_booster_points=float(os.getenv("SWEEP_BOOSTER_POINTS", "0.0")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.01")),
            max_leverage=int(os.getenv("MAX_LEVERAGE", "3")),
        )
        self.liq_detector = SyntheticLiquidationDetector()
        self.regime_engine = MarketRegimeEngine()
        self.funding_filter = FundingFilterEngine(
            proximity_threshold_minutes=self.funding_proximity_minutes,
            extreme_threshold_8h=self.funding_extreme_threshold_8h,
        )
        self.quality_filter = QualityFilter(
            min_24h_volume_usdt=float(os.getenv("MIN_24H_VOLUME_USDT", "10000000")),
            max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", "2.5")),
        )
        self.sweep_detector = LiquiditySweepDetector()
        self.sentiment_engine = SentimentEngine()

    def _state_is_compatible(self, snapshot: MarketStateSnapshot) -> bool:
        return (
            snapshot.strategy_revision == self._provenance["strategy_revision"]
            and snapshot.config_fingerprint == self._provenance["config_fingerprint"]
            and snapshot.research_schema_version == self._provenance["research_schema_version"]
        )

    def load_previous_state(self) -> Dict[str, MarketStateSnapshot]:
        target = self.state_file
        if not target.exists() and Path(".market_state.bin").exists():
            target = Path(".market_state.bin")
        if not target.exists():
            return {}
        try:
            raw = target.read_bytes()
            if raw:
                items = msgspec.json.decode(raw, type=List[MarketStateSnapshot])
                return {item.symbol: item for item in items}
        except Exception as exc:
            logger.error("state_load_failed path=%s error=%s", target, exc)
        return {}

    def save_current_state(self, snapshots: List[MarketStateSnapshot]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        encoded = msgspec.json.encode(snapshots)
        fd, temp_name = tempfile.mkstemp(
            dir=str(self.state_file.parent),
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as temp:
                temp.write(encoded)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_name, self.state_file)
            dir_fd = os.open(self.state_file.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _parse_kline(row: list) -> Tuple[int, float, float, float, float, float, float, float]:
        if not isinstance(row, list) or len(row) < 12:
            raise ValueError("Malformed kline")
        open_ms = int(row[0])
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])
        taker_buy = float(row[9])
        taker_sell = volume - taker_buy
        values = (open_price, high, low, close, volume, taker_buy, taker_sell)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Non-finite kline")
        if min(open_price, high, low, close) <= 0.0 or volume < 0.0 or taker_buy < 0.0 or taker_sell < 0.0:
            raise ValueError("Invalid kline values")
        return open_ms, open_price, high, low, close, volume, taker_buy, taker_sell

    @staticmethod
    def _state_is_for_candle(snapshot: Optional[MarketStateSnapshot], candle_open_ms: int) -> bool:
        return snapshot is not None and snapshot.candle_open_time_ms == candle_open_ms

    @classmethod
    def _atr_pct(cls, klines: List[list], lookback: int) -> float:
        parsed = [cls._parse_kline(row) for row in klines[-(lookback + 1):]]
        if len(parsed) < lookback + 1:
            raise ValueError("Insufficient ATR history")
        true_ranges: List[float] = []
        prev_close = parsed[0][4]
        for _, _, high, low, close, *_ in parsed[1:]:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            prev_close = close
        return (sum(true_ranges) / len(true_ranges)) / parsed[-1][4]

    @staticmethod
    def _compute_friction_rt(ob_spread_bps: float, atr_pct: float) -> float:
        """Estimate round-trip taker costs conservatively and per symbol.

        The model is deliberately a cost floor, not an alpha assumption: it adds
        two-sided spread, configured taker commission and a bounded ATR impact term.
        It must be treated as a research approximation until fill data is available.
        """
        commission_rt = float(os.getenv("TAKER_COMMISSION_RT_PCT", "0.0010"))
        slippage_atr_fraction = float(os.getenv("SLIPPAGE_ATR_FRACTION", "0.15"))
        max_slippage = float(os.getenv("MAX_SLIPPAGE_RT_PCT", "0.0030"))
        if not all(math.isfinite(x) and x >= 0.0 for x in (ob_spread_bps, atr_pct, commission_rt, slippage_atr_fraction, max_slippage)):
            raise ValueError("Invalid friction input/configuration")
        spread_cost_rt = ob_spread_bps / 10000.0
        slippage_rt = min(atr_pct * slippage_atr_fraction, max_slippage)
        return commission_rt + spread_cost_rt + slippage_rt

    async def close(self) -> None:
        await asyncio.gather(
            self.ingestion.stop(),
            self.sentiment_engine.close(),
            return_exceptions=False,
        )

    @staticmethod
    def _returns_from_prices(prices: tuple) -> Optional[np.ndarray]:
        if len(prices) < 3:
            return None
        arr = np.asarray(prices, dtype=np.float64)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            return None
        rets = arr[1:] / arr[:-1] - 1.0
        return rets if len(rets) >= 2 and np.std(rets) > 1e-12 else None

    @staticmethod
    def _downgrade_signal(signal: SignalEvent, reason: str) -> None:
        """Convert an otherwise-strong candidate into a non-actionable NEUTRAL event."""
        signal.signal_type = "NEUTRAL"
        signal.gate_status = reason
        signal.invalidation_price = signal.price
        signal.target_price = signal.price
        signal.risk_reward_ratio = 0.0
        signal.suggested_position_usd = 0.0
        signal.suggested_leverage = 1
        signal.trailing_stop_activation_pct = 0.0
        signal.trailing_stop_distance_pct = 0.0

    def _apply_portfolio_correlation_limit(
        self,
        signals: List[SignalEvent],
        snapshots: Optional[Dict[str, MarketStateSnapshot]] = None,
    ) -> Tuple[List[SignalEvent], int]:
        """Apply global portfolio risk, direction, gross-leverage and correlation caps."""
        snapshots = snapshots or {}
        downgraded = 0
        direction_counts = {"STRONG_LONG": 0, "STRONG_SHORT": 0}
        kept: List[SignalEvent] = []
        aggregate_risk_pct = 0.0
        aggregate_notional_usd = 0.0

        candidates = sorted(
            [s for s in signals if s.signal_type in ("STRONG_LONG", "STRONG_SHORT")],
            key=lambda s: abs(s.composite_score),
            reverse=True,
        )
        for signal in candidates:
            direction = signal.signal_type
            if direction_counts[direction] >= self.max_strong_per_direction:
                self._downgrade_signal(signal, "BLOCKED_PORTFOLIO_COUNT_LIMIT")
                downgraded += 1
                continue

            stop_risk = 0.0
            if signal.suggested_position_usd > 0.0 and signal.price > 0.0:
                stop_distance = abs(signal.price - signal.invalidation_price) / signal.price
                stop_risk = (signal.suggested_position_usd * stop_distance) / max(self.account_equity, 1e-12)
            if stop_risk <= 0.0:
                stop_risk = self.signal_engine.risk_per_trade_pct

            if aggregate_risk_pct + stop_risk > self.max_aggregate_risk_pct + 1e-12:
                self._downgrade_signal(signal, "BLOCKED_PORTFOLIO_RISK_LIMIT")
                downgraded += 1
                continue

            if aggregate_notional_usd + signal.suggested_position_usd > self.account_equity * self.max_gross_leverage + 1e-9:
                self._downgrade_signal(signal, "BLOCKED_PORTFOLIO_GROSS_LEVERAGE")
                downgraded += 1
                continue

            new_returns = self._returns_from_prices(
                tuple(snapshots.get(signal.symbol).price_history_5m) if snapshots.get(signal.symbol) else ()
            )
            blocked_corr = None
            if new_returns is not None:
                new_sign = 1 if direction == "STRONG_LONG" else -1
                for existing in kept:
                    old_returns = self._returns_from_prices(
                        tuple(snapshots.get(existing.symbol).price_history_5m) if snapshots.get(existing.symbol) else ()
                    )
                    if old_returns is None:
                        continue
                    common = min(len(old_returns), len(new_returns))
                    if common < max(self.beta_min_samples, 10):
                        continue
                    corr = float(np.corrcoef(old_returns[-common:], new_returns[-common:])[0, 1])
                    old_sign = 1 if existing.signal_type == "STRONG_LONG" else -1
                    effective_corr = corr * new_sign * old_sign
                    if math.isfinite(effective_corr) and effective_corr >= self.max_portfolio_correlation:
                        blocked_corr = effective_corr
                        break
            if blocked_corr is not None:
                self._downgrade_signal(signal, f"BLOCKED_PORTFOLIO_CORRELATION:{blocked_corr:+.2f}")
                downgraded += 1
                continue

            kept.append(signal)
            direction_counts[direction] += 1
            aggregate_risk_pct += stop_risk
            aggregate_notional_usd += signal.suggested_position_usd

        return signals, downgraded

    @staticmethod
    def _derive_cvd_series_and_divergence(
        parsed: List[tuple],
        lookback: int,
    ) -> Tuple[List[float], List[float]]:
        """Reconstruct CVD and point-in-time divergence from completed klines only."""
        cvd_values: List[float] = []
        running = 0.0
        for row in parsed:
            running += row[6] - row[7]
            cvd_values.append(running)

        divergence_values: List[float] = [0.0] * len(parsed)
        for idx in range(lookback, len(parsed)):
            prices = np.asarray([row[4] for row in parsed[: idx + 1]], dtype=np.float64)
            cvd = np.asarray(cvd_values[: idx + 1], dtype=np.float64)
            score, _ = detect_cvd_divergence_jit(
                prices,
                cvd,
                lookback=min(lookback, len(prices) - 1),
            )
            divergence_values[idx] = float(score)
        return cvd_values, divergence_values

    @staticmethod
    def _select_recoverable_history(
        state_values: tuple,
        persisted_rows: list[dict],
        key_fn,
        state_allowed: bool,
        expected_previous_timestamp_ms: Optional[int] = None,
        interval_ms: int = 5 * 60 * 1000,
    ) -> tuple[tuple, str]:
        """Choose the strongest contiguous history available for one factor.

        A contiguous previous state is preferred only when it is at least as
        complete as durable research history. This prevents partial state resets
        from pinning a factor to a short 3-4 sample tail forever.
        """
        state_hist = tuple(state_values) if state_allowed else ()
        research_values = []
        for row in persisted_rows:
            try:
                value = key_fn(row)
            except (KeyError, TypeError, ValueError):
                continue
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                research_values.append(value)
        research_hist = tuple(research_values)
        if expected_previous_timestamp_ms is not None and persisted_rows:
            valid_research_tail = None
            try:
                max_ts = max(int(row["timestamp_ms"]) for row in persisted_rows if "timestamp_ms" in row)
                if max_ts == expected_previous_timestamp_ms:
                    valid_research_tail = research_hist
            except (TypeError, ValueError, KeyError):
                valid_research_tail = None
            if valid_research_tail is None:
                research_hist = ()
        if len(research_hist) > len(state_hist):
            return research_hist, "RESEARCH"
        if state_hist:
            return state_hist, "STATE"
        if research_hist:
            return research_hist, "RESEARCH"
        return (), "EMPTY"

    async def scan(
        self,
    ) -> Tuple[List[SignalEvent], List[SyntheticLiquidation], ScreenerResult, List[dict]]:
        t0 = time.time()
        prev_state = self.load_previous_state()
        signals: List[SignalEvent] = []
        synthetic_liqs: List[SyntheticLiquidation] = []

        try:
            all_tickers = await self.ingestion.fetch_universe_tickers()
            if not all_tickers:
                raise RuntimeError("Failed to retrieve 24h tickers from Binance Futures")

            sorted_symbols = sorted(
                all_tickers.keys(),
                key=lambda s: float(all_tickers[s].get("quoteVolume", 0.0)),
                reverse=True,
            )[:self.top_n_symbols]
            all_premium = await self.ingestion.fetch_universe_premium_index()
            funding_info = await self.ingestion.fetch_universe_funding_info()
            if funding_info is None:
                raise RuntimeError("Funding interval metadata unavailable")

            if "BTCUSDT" not in all_tickers or "BTCUSDT" not in all_premium:
                raise RuntimeError("BTC market data unavailable")

            interval_ms = 5 * 60 * 1000
            now_ms = int(time.time() * 1000)
            current_open_ms = (now_ms // interval_ms) * interval_ms
            closed_open_ms = current_open_ms - interval_ms
            closed_end_ms = current_open_ms - 1
            kline_history = max(self.history_bars + 1, self.atr_lookback + 1)

            btc_klines = await self.ingestion.fetch_symbol_closed_5m_klines(
                "BTCUSDT", closed_open_ms, history_bars=kline_history
            )
            if btc_klines is None:
                raise RuntimeError("BTC 5m candle history unavailable")
            btc_parsed = [self._parse_kline(row) for row in btc_klines]
            btc_last_close = btc_parsed[-1][4]
            btc_prev_close = btc_parsed[-2][4]
            btc_change_5m_pct = (btc_last_close / btc_prev_close - 1.0) * 100.0
            btc_24h_chg = float(all_tickers["BTCUSDT"]["priceChangePercent"])

            btc_regime = self.regime_engine.evaluate_btc_regime(
                btc_last_price=btc_last_close,
                btc_prev_price=btc_prev_close,
                btc_change_24h_pct=btc_24h_chg,
            )

            # Durable research rows are the recovery source for state-only
            # histories after a missed/reset 5m cycle.
            research_history = self.research_recorder.load_recent_histories(
                before_timestamp_ms=closed_end_ms,
                symbols=sorted_symbols,
                limit=self.history_bars,
                expected_provenance={k: self._provenance[k] for k in ("strategy_revision", "config_fingerprint", "research_schema_version")},
            )

            sem = asyncio.Semaphore(self.concurrency_limit)
            Result = Union[None, Tuple[Optional[SignalEvent], MarketStateSnapshot, Optional[SyntheticLiquidation], dict], Exception]

            async def analyze_symbol(symbol: str) -> Result:
                async with sem:
                    try:
                        ticker = all_tickers.get(symbol)
                        premium = all_premium.get(symbol)
                        if not ticker or not premium:
                            return None

                        try:
                            quote_vol_24h = float(ticker["quoteVolume"])
                            high_24h = float(ticker["highPrice"])
                            low_24h = float(ticker["lowPrice"])
                            price_change_pct_24h = float(ticker["priceChangePercent"])
                            raw_funding = float(premium["lastFundingRate"])
                            mark_price = float(premium["markPrice"])
                            index_price = float(premium["indexPrice"])
                            next_funding_time_ms = int(premium["nextFundingTime"])
                        except (KeyError, TypeError, ValueError):
                            return None

                        if not all(math.isfinite(x) for x in (quote_vol_24h, high_24h, low_24h, price_change_pct_24h, raw_funding, mark_price, index_price)):
                            return None
                        if quote_vol_24h < 0.0 or index_price <= 0.0 or mark_price <= 0.0:
                            return None

                        funding_interval_h = funding_info.get(symbol, 8.0)
                        if not math.isfinite(funding_interval_h) or funding_interval_h <= 0.0 or 1.0 + raw_funding <= 0.0:
                            return None
                        norm_8h_funding = (1.0 + raw_funding) ** (8.0 / funding_interval_h) - 1.0
                        basis_bps = (mark_price - index_price) / index_price * 10000.0

                        p_snap = prev_state.get(symbol)
                        expected_prev_open_ms = closed_open_ms - interval_ms
                        if p_snap is not None and p_snap.candle_open_time_ms == closed_open_ms:
                            return None
                        prev_contiguous = self._state_is_for_candle(p_snap, expected_prev_open_ms) and (p_snap is not None and self._state_is_compatible(p_snap))

                        klines = await self.ingestion.fetch_symbol_closed_5m_klines(
                            symbol, closed_open_ms, history_bars=kline_history
                        )
                        if klines is None:
                            return None
                        parsed = [self._parse_kline(row) for row in klines]
                        if parsed[-1][0] != closed_open_ms:
                            return None

                        candle_open, _, candle_high, candle_low, candle_close, volume_5m, taker_buy, taker_sell = parsed[-1]
                        alt_change_5m_pct = (candle_close / parsed[-2][4] - 1.0) * 100.0
                        atr_pct = self._atr_pct(klines, self.atr_lookback)

                        oi_history = await self.ingestion.fetch_symbol_open_interest_history(
                            symbol,
                            closed_open_ms,
                            history_bars=max(self.history_bars, self.z_history_min_samples + 1),
                        )
                        if oi_history is None or len(oi_history) < self.z_history_min_samples + 1:
                            return None
                        oi_values = [oi for _, oi in oi_history]
                        curr_oi = oi_values[-1]
                        prev_oi = oi_values[-2]
                        delta_oi_5m = curr_oi - prev_oi
                        delta_oi_pct = delta_oi_5m / prev_oi if prev_oi > 0.0 else None

                        ob = await self.ingestion.fetch_symbol_orderbook_top(symbol, limit=20)
                        if ob is None:
                            return None
                        bids_qty = np.asarray([q for _, q in ob.bids], dtype=np.float64)
                        asks_qty = np.asarray([q for _, q in ob.asks], dtype=np.float64)
                        obi = float(compute_weighted_obi_jit(bids_qty, asks_qty, decay=self.obi_decay))

                        q_res = self.quality_filter.evaluate(
                            symbol=symbol,
                            quote_volume_24h=quote_vol_24h,
                            spread_bps=ob.spread_bps,
                            funding_rate_8h=norm_8h_funding,
                        )
                        if not q_res.is_valid:
                            return None

                        current_cvd_delta = taker_buy - taker_sell
                        cvd_series, div_series = self._derive_cvd_series_and_divergence(parsed, self.cvd_lookback)
                        current_cvd = (
                            (p_snap.cumulative_cvd_5m + current_cvd_delta)
                            if prev_contiguous and p_snap is not None
                            else cvd_series[-1]
                        )
                        cvd_hist = cvd_series[-self.history_bars:]
                        price_hist = [row[4] for row in parsed[-self.history_bars:]]
                        time_hist = [row[0] for row in parsed[-self.history_bars:]]
                        div_hist_from_klines = div_series[-self.history_bars:]
                        div_score: Optional[float] = div_hist_from_klines[-1] if div_hist_from_klines else None

                        closed_trades = await self.ingestion.fetch_5m_agg_trades(symbol, closed_open_ms, closed_end_ms)
                        if closed_trades is None or not closed_trades:
                            return None
                        trade_qtys = [float(t["q"]) for t in closed_trades]
                        is_buyer = [not bool(t["m"]) for t in closed_trades]
                        total_trade_qty = sum(trade_qtys)
                        bucket_volume = total_trade_qty / self.vpin_window_baskets
                        if bucket_volume <= 0.0:
                            return None
                        vpin = float(
                            compute_vpin_numba(
                                np.asarray(trade_qtys, dtype=np.float64),
                                np.asarray(is_buyer, dtype=np.bool_),
                                bucket_vol=bucket_volume,
                                window_baskets=self.vpin_window_baskets,
                            )
                        )

                        sweep_event = None
                        swing_bars = parsed[-(self.cvd_lookback + 1):-1]
                        if len(swing_bars) >= self.cvd_lookback:
                            recent_high_swing = max(bar[2] for bar in swing_bars)
                            recent_low_swing = min(bar[3] for bar in swing_bars)
                            sweep_event = self.sweep_detector.detect(
                                symbol=symbol,
                                current_price=candle_close,
                                current_high=candle_high,
                                current_low=candle_low,
                                recent_swing_high=recent_high_swing,
                                recent_swing_low=recent_low_swing,
                                cvd_delta=current_cvd_delta,
                            )
                        else:
                            # fallback: use 24h extremes from ticker
                            recent_high_swing = high_24h
                            recent_low_swing = low_24h
                        sweep_reclaim = sweep_event is not None and sweep_event.is_confirmed
                        sweep_pattern = sweep_event.pattern_type if sweep_event is not None else "NONE"

                        liq = None
                        if delta_oi_pct is not None:
                            liq = self.liq_detector.detect(
                                symbol=symbol,
                                current_price=candle_close,
                                price_change_pct=alt_change_5m_pct,
                                delta_oi=delta_oi_5m,
                                taker_buy_vol=taker_buy,
                                taker_sell_vol=taker_sell,
                            )

                        rolling_beta = None
                        rolling_beta_reason = "BTC_SELF"
                        if symbol != "BTCUSDT":
                            rolling_beta, rolling_beta_reason = self.regime_engine.calculate_rolling_beta_with_reason(
                                time_hist,
                                price_hist,
                                [row[0] for row in btc_parsed[-self.history_bars:]],
                                [row[4] for row in btc_parsed[-self.history_bars:]],
                                min_samples=self.beta_min_samples,
                            )

                        if symbol == "BTCUSDT":
                            rs_to_btc = 0.0
                        elif rolling_beta is not None:
                            rs_to_btc = self.regime_engine.calculate_relative_strength(
                                alt_change_5m_pct=alt_change_5m_pct,
                                btc_change_5m_pct=btc_change_5m_pct,
                                beta=rolling_beta,
                            )
                        else:
                            rs_to_btc = 0.0

                        btc_long_allowed, btc_long_reason = self.regime_engine.check_signal_gate(
                            symbol=symbol,
                            signal_type="STRONG_LONG",
                            btc_regime=btc_regime,
                            alt_change_5m_pct=alt_change_5m_pct,
                            beta=rolling_beta,
                        ) if symbol == "BTCUSDT" or rolling_beta is not None else (False, "INSUFFICIENT_ROLLING_BETA")

                        btc_short_allowed, btc_short_reason = self.regime_engine.check_signal_gate(
                            symbol=symbol,
                            signal_type="STRONG_SHORT",
                            btc_regime=btc_regime,
                            alt_change_5m_pct=alt_change_5m_pct,
                            beta=rolling_beta,
                        ) if symbol == "BTCUSDT" or rolling_beta is not None else (False, "INSUFFICIENT_ROLLING_BETA")

                        funding_gate_long = self.funding_filter.evaluate_funding_gate(
                            symbol=symbol,
                            signal_type="STRONG_LONG",
                            funding_rate_8h=norm_8h_funding,
                            next_funding_time_ms=next_funding_time_ms,
                        )
                        funding_gate_short = self.funding_filter.evaluate_funding_gate(
                            symbol=symbol,
                            signal_type="STRONG_SHORT",
                            funding_rate_8h=norm_8h_funding,
                            next_funding_time_ms=next_funding_time_ms,
                        )
                        gate_long_status = btc_long_reason if not btc_long_allowed else (funding_gate_long.gate_reason if not funding_gate_long.allow_long else "PASSED")
                        gate_short_status = btc_short_reason if not btc_short_allowed else (funding_gate_short.gate_reason if not funding_gate_short.allow_short else "PASSED")

                        sent = await self.sentiment_engine.fetch_sentiment_divergence(
                            symbol,
                            start_time_ms=closed_open_ms,
                            end_time_ms=current_open_ms,
                        )
                        sentiment_available = sent is not None
                        whale_divergence = float(sent.divergence_score) if sentiment_available else 0.0

                        persisted_rows = research_history.get(symbol, [])
                        state_allowed = prev_contiguous and p_snap is not None
                        base_funding_hist, funding_source = self._select_recoverable_history(
                            p_snap.funding_history_5m if p_snap is not None else (),
                            persisted_rows,
                            lambda row: row.get("funding_rate_8h"),
                            state_allowed,
                            expected_previous_timestamp_ms=closed_end_ms - interval_ms,
                        )
                        base_basis_hist, basis_source = self._select_recoverable_history(
                            p_snap.basis_history_5m if p_snap is not None else (),
                            persisted_rows,
                            lambda row: row.get("basis_bps"),
                            state_allowed,
                            expected_previous_timestamp_ms=closed_end_ms - interval_ms,
                        )
                        base_micro_hist, micro_source = self._select_recoverable_history(
                            p_snap.micro_factor_history_5m if p_snap is not None else (),
                            persisted_rows,
                            lambda row: float(row["obi"]) * (1.0 - float(row["vpin"])),
                            state_allowed,
                            expected_previous_timestamp_ms=closed_end_ms - interval_ms,
                        )
                        base_whale_hist, whale_source = self._select_recoverable_history(
                            p_snap.whale_divergence_history_5m if p_snap is not None else (),
                            [row for row in persisted_rows if bool(row.get("sentiment_available", False))],
                            lambda row: row.get("whale_divergence_score"),
                            state_allowed,
                            expected_previous_timestamp_ms=closed_end_ms - interval_ms,
                        )
                        factor_sources = {funding_source, basis_source, micro_source, whale_source}
                        if factor_sources == {"STATE"}:
                            recovery_source = "STATE"
                        elif factor_sources == {"EMPTY"}:
                            recovery_source = "EMPTY"
                        elif "RESEARCH" in factor_sources and "STATE" in factor_sources:
                            recovery_source = "MIXED"
                        elif "RESEARCH" in factor_sources:
                            recovery_source = "RESEARCH"
                        else:
                            recovery_source = "STATE"

                        funding_hist = base_funding_hist[-self.history_bars + 1:] + (norm_8h_funding,)
                        basis_hist = base_basis_hist[-self.history_bars + 1:] + (basis_bps,)
                        micro_factor = obi * (1.0 - vpin)
                        micro_hist = base_micro_hist[-self.history_bars + 1:] + (micro_factor,)
                        whale_hist = (
                            base_whale_hist[-self.history_bars + 1:] + (whale_divergence,)
                            if sentiment_available
                            else base_whale_hist[-self.history_bars:]
                        )
                        oi_delta_hist = []
                        for prev, cur in zip(oi_values[:-1], oi_values[1:]):
                            if prev > 0.0:
                                oi_delta_hist.append((cur - prev) / prev)
                        delta_hist = tuple(oi_delta_hist[-self.history_bars:])
                        div_hist = tuple(div_hist_from_klines[-self.history_bars:])

                        signal: Optional[SignalEvent] = None
                        readiness_reasons: List[str] = []
                        min_prior = self.z_history_min_samples
                        if delta_oi_pct is None:
                            readiness_reasons.append("MISSING_CURRENT_OI_DELTA")
                        if symbol != "BTCUSDT" and rolling_beta is None:
                            readiness_reasons.append(f"BETA_{rolling_beta_reason}")
                        if len(funding_hist) - 1 < min_prior:
                            readiness_reasons.append(f"FUNDING_HISTORY_{len(funding_hist) - 1}/{min_prior}")
                        if len(basis_hist) - 1 < min_prior:
                            readiness_reasons.append(f"BASIS_HISTORY_{len(basis_hist) - 1}/{min_prior}")
                        if len(delta_hist) - 1 < min_prior:
                            readiness_reasons.append(f"OI_HISTORY_{len(delta_hist) - 1}/{min_prior}")
                        if len(micro_hist) - 1 < min_prior:
                            readiness_reasons.append(f"MICRO_HISTORY_{len(micro_hist) - 1}/{min_prior}")
                        if len(div_hist) - 1 < min_prior:
                            readiness_reasons.append(f"CVD_HISTORY_{len(div_hist) - 1}/{min_prior}")
                        if div_score is None:
                            readiness_reasons.append("MISSING_CURRENT_CVD_DIVERGENCE")
                        signal_ready = not readiness_reasons

                        if signal_ready:
                            z_cvd, z_fund, z_oi, z_micro, z_whale = self.signal_engine.calculate_factor_zscores(
                                funding_rate_8h=norm_8h_funding,
                                basis_spread_bps=basis_bps,
                                delta_oi_pct=delta_oi_pct if delta_oi_pct is not None else 0.0,
                                obi=obi,
                                vpin=vpin,
                                cvd_divergence_score=div_score if div_score is not None else 0.0,
                                whale_divergence_score=whale_divergence,
                                funding_history=funding_hist[:-1],
                                basis_history=basis_hist[:-1],
                                delta_oi_pct_history=delta_hist[:-1],
                                micro_factor_history=micro_hist[:-1],
                                cvd_history=div_hist[:-1],
                                whale_history=whale_hist[:-1] if len(whale_hist) > 1 else (),
                            )
                            if not sentiment_available or len(whale_hist) - 1 < self.z_history_min_samples:
                                z_whale = 0.0

                            # NEW: per-symbol friction
                            friction_rt = self._compute_friction_rt(ob.spread_bps, atr_pct)

                            signal = self.signal_engine.compute_signal(
                                symbol=symbol,
                                current_price=candle_close,
                                funding_rate_8h=norm_8h_funding,
                                basis_spread_bps=basis_bps,
                                delta_oi=delta_oi_5m,
                                oi_total=curr_oi,
                                obi=obi,
                                vpin=vpin,
                                cvd_divergence_score=div_score if div_score is not None else 0.0,
                                recent_high=recent_high_swing,
                                recent_low=recent_low_swing,
                                z_whale_sentiment=whale_divergence,
                                relative_strength=rs_to_btc,
                                sweep_reclaim=sweep_reclaim,
                                sweep_pattern=sweep_pattern,
                                atr_pct=atr_pct,
                                gate_long_status=gate_long_status,
                                gate_short_status=gate_short_status,
                                z_cvd_override=z_cvd,
                                z_fund_override=z_fund,
                                z_delta_oi_override=z_oi,
                                z_micro_override=z_micro,
                                z_whale_override=z_whale,
                                timestamp_ms=closed_end_ms,
                                decision_timestamp_ms=int(time.time() * 1000),
                                account_equity=self.account_equity,
                                whale_history_length=len(whale_hist) - 1,
                                friction_round_trip_pct=friction_rt,
                                active_factors=self.signal_engine.factor_activity(
                                    funding_history=funding_hist[:-1],
                                    basis_history=basis_hist[:-1],
                                    delta_oi_pct_history=delta_hist[:-1],
                                    micro_factor_history=micro_hist[:-1],
                                    whale_history=whale_hist[:-1] if len(whale_hist) > 1 else (),
                                ),
                            )

                        snapshot = MarketStateSnapshot(
                            symbol=symbol,
                            timestamp_ms=closed_end_ms,
                            last_price=candle_close,
                            open_interest=curr_oi,
                            delta_oi_5m=delta_oi_5m,
                            cumulative_cvd_5m=current_cvd,
                            funding_rate_8h=norm_8h_funding,
                            basis_bps=basis_bps,
                            vpin_estimate=vpin,
                            obi_score=obi,
                            composite_score=signal.composite_score if signal is not None else 0.0,
                            low_24h=low_24h,
                            high_24h=high_24h,
                            whale_sentiment_z=signal.z_whale_sentiment if signal is not None else 0.0,
                            cvd_history_5m=tuple(cvd_hist[-self.history_bars:]),
                            price_history_5m=tuple(price_hist[-self.history_bars:]),
                            funding_history_5m=tuple(funding_hist[-self.history_bars:]),
                            basis_history_5m=tuple(basis_hist[-self.history_bars:]),
                            delta_oi_pct_history_5m=tuple(delta_hist[-self.history_bars:]),
                            micro_factor_history_5m=tuple(micro_hist[-self.history_bars:]),
                            whale_divergence_history_5m=tuple(whale_hist[-self.history_bars:]),
                            cvd_divergence_history_5m=tuple(div_hist[-self.history_bars:]),
                            candle_open_time_ms=candle_open,
                            candle_open_times_5m=tuple(time_hist[-self.history_bars:]),
                            candle_high_5m=candle_high,
                            candle_low_5m=candle_low,
                            signal_ready=signal is not None,
                            strategy_revision=str(self._provenance["strategy_revision"]),
                            code_revision=str(self._provenance["code_revision"]),
                            config_fingerprint=str(self._provenance["config_fingerprint"]),
                            research_schema_version=int(self._provenance["research_schema_version"]),
                        )
                        research_row = {
                            **provenance(),
                            "timestamp_ms": closed_end_ms,
                            "symbol": symbol,
                            "open": float(parsed[-1][1]),
                            "high": float(candle_high),
                            "low": float(candle_low),
                            "close": float(candle_close),
                            "volume": float(volume_5m),
                            "open_interest": float(curr_oi),
                            "delta_oi_pct": float(delta_oi_pct if delta_oi_pct is not None else 0.0),
                            "funding_rate_8h": float(norm_8h_funding),
                            "basis_bps": float(basis_bps),
                            "obi": float(obi),
                            "vpin": float(vpin),
                            "vpin_method": VPIN_ESTIMATE_METHOD,
                            "cvd_divergence_score": float(div_score if div_score is not None else 0.0),
                            "whale_divergence_score": float(whale_divergence),
                            "atr_pct": float(atr_pct),
                            "recent_high": float(recent_high_swing),
                            "recent_low": float(recent_low_swing),
                            "spread_bps": float(ob.spread_bps),
                            "gate_long_status": str(gate_long_status),
                            "gate_short_status": str(gate_short_status),
                            "relative_strength": float(rs_to_btc),
                            "sweep_reclaim": bool(sweep_reclaim),
                            "sweep_pattern": str(sweep_pattern),
                            "candidate_signal_type": signal.signal_type if signal is not None else "NONE",
                            "candidate_score": float(signal.composite_score) if signal is not None else 0.0,
                            "recorded_signal_type": signal.signal_type if signal is not None else "NONE",
                            "recorded_score": float(signal.composite_score) if signal is not None else 0.0,
                            "final_signal_type": signal.signal_type if signal is not None else "NONE",
                            "portfolio_action": "CANDIDATE_ONLY" if signal is not None else "NO_SIGNAL",
                            "actionable": bool(signal is not None and signal.signal_type in ("STRONG_LONG", "STRONG_SHORT")),
                            "dispatch_allowed": None,
                            "dispatch_block_reason": None,
                            "dispatched_at_ms": None,
                            "gate_status": signal.gate_status if signal is not None else "NOT_READY",
                            "signal_decision_timestamp_ms": int(signal.decision_timestamp_ms) if signal is not None else None,
                            "signal_event": signal_to_dict(signal) if signal is not None else None,
                            "signal_ready": bool(signal_ready),
                            "signal_readiness_reasons": list(readiness_reasons),
                            "sentiment_available": bool(sentiment_available),
                            "sentiment_observation_timestamp_ms": int(sent.observation_timestamp_ms) if sent is not None else None,
                            "state_recovery_source": recovery_source,
                            "persisted_research_rows_available": len(persisted_rows),
                        }
                        return signal, snapshot, liq, research_row
                    except Exception as exc:
                        logger.exception("symbol_analysis_failed symbol=%s error=%s", symbol, exc)
                        return exc

            results = await asyncio.gather(*(analyze_symbol(sym) for sym in sorted_symbols))
            successful_symbols = 0
            rejected_symbols = 0
            failed_symbols = 0
            signal_ready_symbols = 0
            readiness_counter: Counter[str] = Counter()
            recovery_counter: Counter[str] = Counter()
            new_snapshots: List[MarketStateSnapshot] = []
            research_rows: List[dict] = []

            for result in results:
                if isinstance(result, Exception):
                    failed_symbols += 1
                    continue
                if result is None:
                    rejected_symbols += 1
                    continue
                successful_symbols += 1
                signal, snapshot, liq, research_row = result
                new_snapshots.append(snapshot)
                research_rows.append(research_row)
                row_reasons = research_row.get("signal_readiness_reasons", [])
                for reason in row_reasons:
                    readiness_counter[str(reason)] += 1
                recovery_counter[str(research_row.get("state_recovery_source", "UNKNOWN"))] += 1
                if signal is not None:
                    signal_ready_symbols += 1
                    signals.append(signal)
                if liq is not None:
                    synthetic_liqs.append(liq)

            failure_ratio = failed_symbols / max(len(sorted_symbols), 1)
            if failure_ratio > self.failure_ratio_limit:
                raise RuntimeError(f"Failure ratio {failure_ratio:.2%} exceeds {self.failure_ratio_limit:.2%}")
            if sorted_symbols and successful_symbols == 0:
                raise RuntimeError("No symbol produced a valid market snapshot")

            previous_state_symbols = set(prev_state)
            successful_state_symbols = {snapshot.symbol for snapshot in new_snapshots}
            stale_state_symbols_dropped = len(previous_state_symbols - successful_state_symbols)
            self.save_current_state(new_snapshots)
            if stale_state_symbols_dropped:
                logger.warning(
                    "stale_state_symbols_dropped count=%s",
                    stale_state_symbols_dropped,
                )

            signals.sort(key=lambda s: s.composite_score, reverse=True)
            snapshot_map = {snapshot.symbol: snapshot for snapshot in new_snapshots}
            signals, portfolio_limited = self._apply_portfolio_correlation_limit(signals, snapshot_map)

            signal_map = {(s.symbol, s.timestamp_ms): s for s in signals}
            for row in research_rows:
                key = (str(row["symbol"]).upper(), int(row["timestamp_ms"]))
                candidate = row.get("signal_event") or {}
                final_signal = signal_map.get(key)
                row["candidate_signal_type"] = candidate.get("signal_type", "NONE")
                row["candidate_score"] = float(candidate.get("composite_score", 0.0))
                if final_signal is None:
                    row["final_signal_type"] = "NONE"
                    row["recorded_signal_type"] = "NONE"
                    row["portfolio_action"] = "NOT_ACTIONABLE"
                    row["actionable"] = False
                    row["gate_status"] = "NOT_READY_OR_REJECTED"
                else:
                    row["final_signal_type"] = final_signal.signal_type
                    row["recorded_signal_type"] = final_signal.signal_type
                    row["portfolio_action"] = ("ACTIONABLE" if final_signal.signal_type in ("STRONG_LONG", "STRONG_SHORT") else "BLOCKED_OR_NEUTRAL")
                    row["actionable"] = bool(final_signal.signal_type in ("STRONG_LONG", "STRONG_SHORT"))
                    row["gate_status"] = final_signal.gate_status
                    row["recorded_score"] = float(final_signal.composite_score)
                    row["signal_event_final"] = signal_to_dict(final_signal)
            strong_longs = [s for s in signals if s.signal_type == "STRONG_LONG"]
            strong_shorts = [s for s in signals if s.signal_type == "STRONG_SHORT"]
            duration = time.time() - t0

            summary = ScreenerResult(
                timestamp_ms=closed_end_ms,
                duration_sec=round(duration, 2),
                total_scanned=len(sorted_symbols),
                strong_longs_count=len(strong_longs),
                strong_shorts_count=len(strong_shorts),
                synthetic_liqs_count=len(synthetic_liqs),
                btc_regime=btc_regime.status,
                btc_change_5m_pct=btc_regime.btc_change_5m_pct,
                successful_symbols=successful_symbols,
                rejected_symbols=rejected_symbols,
                failed_symbols=failed_symbols,
                signal_ready_symbols=signal_ready_symbols,
                portfolio_limited_symbols=portfolio_limited,
                signal_readiness_reasons=dict(readiness_counter.most_common()),
                stale_state_symbols_dropped=stale_state_symbols_dropped,
                state_recovery_sources=dict(recovery_counter.most_common()),
            )
            return signals, synthetic_liqs, summary, research_rows
        except Exception:
            logger.exception("Screener scan failed")
            raise
