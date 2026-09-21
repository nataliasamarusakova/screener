"""
Production 5-minute Quant Screener.
Fail-closed pipeline with robust lag recovery, friction-adjusted scoring, and position sizing.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
import time
from pathlib import Path
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
from engine.microstructure_jit import compute_vpin_numba, compute_weighted_obi_jit
from engine.quality_filter import QualityFilter
from engine.sentiment import SentimentEngine
from engine.signals import QuantSignalEngine

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
        },
        "signals": [
            {
                "symbol": s.symbol,
                "type": s.signal_type,
                "score": s.composite_score,
                "price": s.price,
                "funding_8h": s.funding_8h,
                "basis_bps": s.basis_bps,
                "obi": s.obi,
                "vpin": s.vpin,
                "z_cvd_div": s.z_cvd_div,
                "z_fund_trap": s.z_fund_trap,
                "z_whale_sentiment": s.z_whale_sentiment,
                "relative_strength": s.relative_strength,
                "sweep_reclaim": s.sweep_reclaim,
                "sweep_pattern": s.sweep_pattern,
                "gate_status": s.gate_status,
                "invalidation_price": s.invalidation_price,
                "target_price": s.target_price,
                "risk_reward_ratio": s.risk_reward_ratio,
                "suggested_position_usd": s.suggested_position_usd,
                "suggested_leverage": s.suggested_leverage,
            }
            for s in signals
        ],
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
        self.concurrency_limit = concurrency_limit
        self.top_n_symbols = top_n_symbols
        self.history_bars = int(os.getenv("SIGNAL_HISTORY_BARS", "60"))
        self.z_history_min_samples = int(os.getenv("SIGNAL_Z_MIN_SAMPLES", "24"))
        self.cvd_lookback = int(os.getenv("CVD_LOOKBACK_BARS", "12")) # Raised from 6 to 12
        self.vpin_window_baskets = int(os.getenv("VPIN_WINDOW_BASKETS", "10"))
        self.beta_min_samples = int(os.getenv("BETA_MIN_SAMPLES", "24"))
        self.atr_lookback = int(os.getenv("ATR_LOOKBACK_BARS", "12"))
        self.failure_ratio_limit = float(os.getenv("SCAN_FAILURE_RATIO_LIMIT", "0.10"))
        self.account_equity = float(os.getenv("ACCOUNT_EQUITY_USDT", "10000.0"))

        if self.concurrency_limit < 1:
            raise ValueError("SCAN_CONCURRENCY must be >= 1")
        if self.top_n_symbols < 1:
            raise ValueError("TOP_N_SYMBOLS must be >= 1")
        if self.history_bars <= self.z_history_min_samples:
            raise ValueError("SIGNAL_HISTORY_BARS must be greater than SIGNAL_Z_MIN_SAMPLES")

        self.ingestion = BinanceFuturesIngestion(symbols=[])
        self.signal_engine = QuantSignalEngine(
            z_history_min_samples=self.z_history_min_samples,
            min_effective_rrr=float(os.getenv("MIN_EFFECTIVE_RRR", "1.30")),
            friction_round_trip_pct=float(os.getenv("FRICTION_RT_PCT", "0.0018")),
        )
        self.liq_detector = SyntheticLiquidationDetector()
        self.regime_engine = MarketRegimeEngine()
        self.funding_filter = FundingFilterEngine(proximity_threshold_minutes=20.0)
        self.quality_filter = QualityFilter(
            min_24h_volume_usdt=float(os.getenv("MIN_24H_VOLUME_USDT", "10000000")),
            max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", "2.5")), # Tightened to 2.5 bps
        )
        self.sweep_detector = LiquiditySweepDetector()
        self.sentiment_engine = SentimentEngine()

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
            dir_fd = os.open(target_parent := self.state_file.parent, os.O_RDONLY)
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

    async def close(self) -> None:
        await asyncio.gather(
            self.ingestion.stop(),
            self.sentiment_engine.close(),
            return_exceptions=False,
        )

    async def scan(self) -> Tuple[List[SignalEvent], List[SyntheticLiquidation], ScreenerResult]:
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

            sem = asyncio.Semaphore(self.concurrency_limit)
            Result = Union[None, Tuple[Optional[SignalEvent], MarketStateSnapshot, Optional[SyntheticLiquidation]], Exception]

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
                        prev_contiguous = self._state_is_for_candle(p_snap, expected_prev_open_ms)

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

                        curr_oi = await self.ingestion.fetch_symbol_closed_5m_open_interest(symbol, current_open_ms)
                        if curr_oi is None:
                            return None
                        delta_oi_5m = 0.0
                        delta_oi_pct: Optional[float] = None
                        if prev_contiguous and p_snap and p_snap.open_interest > 0.0:
                            delta_oi_5m = curr_oi - p_snap.open_interest
                            delta_oi_pct = delta_oi_5m / p_snap.open_interest

                        ob = await self.ingestion.fetch_symbol_orderbook_top(symbol, limit=20)
                        if ob is None:
                            return None
                        bids_qty = np.asarray([q for _, q in ob.bids], dtype=np.float64)
                        asks_qty = np.asarray([q for _, q in ob.asks], dtype=np.float64)
                        obi = float(compute_weighted_obi_jit(bids_qty, asks_qty, decay=0.85))

                        q_res = self.quality_filter.evaluate(
                            symbol=symbol,
                            quote_volume_24h=quote_vol_24h,
                            spread_bps=ob.spread_bps,
                            funding_rate_8h=norm_8h_funding,
                        )
                        if not q_res.is_valid:
                            return None

                        current_cvd_delta = taker_buy - taker_sell
                        current_cvd = (p_snap.cumulative_cvd_5m + current_cvd_delta) if prev_contiguous and p_snap else current_cvd_delta
                        prev_cvd_hist = tuple(p_snap.cvd_history_5m) if p_snap and prev_contiguous else ()
                        prev_price_hist = tuple(p_snap.price_history_5m) if p_snap and prev_contiguous else ()
                        prev_times = tuple(p_snap.candle_open_times_5m) if p_snap and prev_contiguous else ()
                        cvd_hist = prev_cvd_hist[-self.history_bars + 1:] + (current_cvd,)
                        price_hist = prev_price_hist[-self.history_bars + 1:] + (candle_close,)
                        time_hist = prev_times[-self.history_bars + 1:] + (closed_open_ms,)

                        div_score: Optional[float] = None
                        if prev_contiguous and len(prev_cvd_hist) >= self.cvd_lookback + 1 and len(prev_cvd_hist) == len(prev_price_hist) == len(prev_times):
                            if prev_times[-1] == expected_prev_open_ms:
                                div_window = max(self.cvd_lookback + 1, 4)
                                prices_arr = np.asarray(price_hist[-div_window:], dtype=np.float64)
                                cvd_arr = np.asarray(cvd_hist[-div_window:], dtype=np.float64)
                                div_score_raw, _ = detect_cvd_divergence_jit(
                                    prices_arr,
                                    cvd_arr,
                                    lookback=min(self.cvd_lookback, len(prices_arr) - 1),
                                )
                                div_score = float(div_score_raw)

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
                        btc_state = prev_state.get("BTCUSDT")
                        if (
                            symbol != "BTCUSDT"
                            and p_snap is not None
                            and btc_state is not None
                            and prev_contiguous
                            and self._state_is_for_candle(btc_state, expected_prev_open_ms)
                        ):
                            rolling_beta = self.regime_engine.calculate_rolling_beta(
                                p_snap.candle_open_times_5m,
                                p_snap.price_history_5m,
                                btc_state.candle_open_times_5m,
                                btc_state.price_history_5m,
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

                        base_funding_hist = p_snap.funding_history_5m if (p_snap and prev_contiguous) else ()
                        base_basis_hist = p_snap.basis_history_5m if (p_snap and prev_contiguous) else ()
                        base_micro_hist = p_snap.micro_factor_history_5m if (p_snap and prev_contiguous) else ()
                        base_whale_hist = p_snap.whale_divergence_history_5m if (p_snap and prev_contiguous) else ()
                        base_delta_hist = p_snap.delta_oi_pct_history_5m if (p_snap and prev_contiguous) else ()
                        base_div_hist = p_snap.cvd_divergence_history_5m if (p_snap and prev_contiguous) else ()

                        funding_hist = base_funding_hist[-self.history_bars + 1:] + (norm_8h_funding,)
                        basis_hist = base_basis_hist[-self.history_bars + 1:] + (basis_bps,)
                        micro_factor = obi * (1.0 - vpin)
                        micro_hist = base_micro_hist[-self.history_bars + 1:] + (micro_factor,)
                        
                        # Preserve existing contract: append whale sample only if fresh sentiment loaded
                        whale_hist = (
                            base_whale_hist[-self.history_bars + 1:] + (whale_divergence,)
                            if sentiment_available
                            else base_whale_hist[-self.history_bars:]
                        )
                        delta_hist = (base_delta_hist[-self.history_bars + 1:] + (delta_oi_pct,)) if delta_oi_pct is not None else base_delta_hist
                        div_hist = (base_div_hist[-self.history_bars + 1:] + (div_score,)) if div_score is not None else base_div_hist

                        signal: Optional[SignalEvent] = None
                        
                        # FIX: Do NOT block entire signal_ready state if whale sentiment is lagged/unavailable.
                        # As long as the core 5 factors (funding, basis, delta OI, micro, div) have 24 bars, signal is ready!
                        signal_ready = (
                            delta_oi_pct is not None
                            and (rolling_beta is not None or symbol == "BTCUSDT")
                        ) and (
                            len(funding_hist) - 1 >= self.z_history_min_samples
                            and len(basis_hist) - 1 >= self.z_history_min_samples
                            and len(delta_hist) - 1 >= self.z_history_min_samples
                            and len(micro_hist) - 1 >= self.z_history_min_samples
                            and len(div_hist) - 1 >= self.z_history_min_samples
                            and div_score is not None
                        )

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
                                timestamp_ms=int(time.time() * 1000),
                                account_equity=self.account_equity,
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
                        )
                        return signal, snapshot, liq
                    except Exception as exc:
                        logger.exception("symbol_analysis_failed symbol=%s error=%s", symbol, exc)
                        return exc

            results = await asyncio.gather(*(analyze_symbol(sym) for sym in sorted_symbols))
            successful_symbols = 0
            rejected_symbols = 0
            failed_symbols = 0
            signal_ready_symbols = 0
            new_snapshots: List[MarketStateSnapshot] = []

            for result in results:
                if isinstance(result, Exception):
                    failed_symbols += 1
                    continue
                if result is None:
                    rejected_symbols += 1
                    continue
                successful_symbols += 1
                signal, snapshot, liq = result
                new_snapshots.append(snapshot)
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

            current_symbols = set(sorted_symbols)
            merged_state: Dict[str, MarketStateSnapshot] = {
                symbol: snapshot
                for symbol, snapshot in prev_state.items()
                if symbol in current_symbols
            }
            merged_state.update({snapshot.symbol: snapshot for snapshot in new_snapshots})
            self.save_current_state(list(merged_state.values()))

            signals.sort(key=lambda s: s.composite_score, reverse=True)
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
            )
            return signals, synthetic_liqs, summary
        except Exception:
            logger.exception("Screener scan failed")
            raise
