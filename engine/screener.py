"""
Quantitative Screener Engine for 100+ Binance Futures Contracts.
Orchestrates:
1. Parallel async ingestion of 100+ contracts.
2. BTC Market Regime & Altcoin Relative Strength (Beta) Filter.
3. Liquidity & Anti-Spoofing Quality Filter.
4. Funding Settlement Epoch Countdown Gate.
5. Wyckoff Spring & Upthrust Liquidity Sweep Reclaim.
6. Smart Money vs. Retail Sentiment Divergence.
7. JIT Microstructure, Synthetic Liquidations & Factor Attribution.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class ScreenerResult(msgspec.Struct, gc=False):
    """Encapsulates the complete result of a 5-minute screener cycle."""
    timestamp_ms: int
    duration_sec: float
    total_scanned: int
    strong_longs_count: int
    strong_shorts_count: int
    synthetic_liqs_count: int
    btc_regime: str
    btc_change_5m_pct: float


class QuantScreener:
    """
    High-performance, pure-Python screener scanning 100+ contracts in parallel.
    """

    def __init__(
        self,
        state_file: Path = Path(".market_state.bin"),
        concurrency_limit: int = 25,
        top_n_symbols: int = 100,
    ) -> None:
        self.state_file = state_file
        self.concurrency_limit = concurrency_limit
        self.top_n_symbols = top_n_symbols
        self.ingestion = BinanceFuturesIngestion(symbols=[])
        self.signal_engine = QuantSignalEngine()
        self.liq_detector = SyntheticLiquidationDetector()
        self.regime_engine = MarketRegimeEngine()
        self.funding_filter = FundingFilterEngine(proximity_threshold_minutes=20.0)
        self.quality_filter = QualityFilter(min_24h_volume_usdt=10_000_000.0, max_spread_bps=3.5)
        self.sweep_detector = LiquiditySweepDetector()
        self.sentiment_engine = SentimentEngine()

    def load_previous_state(self) -> Dict[str, MarketStateSnapshot]:
        """Loads state snapshot from previous 5m cron cycle."""
        if not self.state_file.exists():
            return {}
        try:
            raw = self.state_file.read_bytes()
            if raw:
                items = msgspec.json.decode(raw, type=List[MarketStateSnapshot])
                return {item.symbol: item for item in items}
        except Exception as exc:
            print(f"⚠️ [SCREENER] Could not parse state file: {exc}")
        return {}

    def save_current_state(self, snapshots: List[MarketStateSnapshot]) -> None:
        """Atomically saves state snapshot to disk via temp file."""
        encoded = msgspec.json.encode(snapshots)
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_bytes(encoded)
        temp_file.replace(self.state_file)

    async def scan(self) -> Tuple[List[SignalEvent], List[SyntheticLiquidation], ScreenerResult]:
        """
        Executes full institutional screener cycle with all gates and filters.
        """
        t0 = time.time()
        prev_state = self.load_previous_state()

        # Step 1: Fetch 24hr tickers (Single HTTP call)
        all_tickers = await self.ingestion.fetch_universe_tickers()
        if not all_tickers:
            await self.ingestion.stop()
            raise RuntimeError("Failed to retrieve 24hr tickers from Binance Futures.")

        # Rank universe by quoteVolume
        sorted_symbols = sorted(
            all_tickers.keys(),
            key=lambda s: float(all_tickers[s].get("quoteVolume", 0.0)),
            reverse=True,
        )[:self.top_n_symbols]

        # Step 2: Fetch Premium Index (Single HTTP call)
        all_premium = await self.ingestion.fetch_universe_premium_index()

        # Step 3: Evaluate Bitcoin Macro Regime
        btc_ticker = all_tickers.get("BTCUSDT", {})
        btc_last_p = float(btc_ticker.get("lastPrice", 0.0))
        btc_prev_p = prev_state.get("BTCUSDT").last_price if "BTCUSDT" in prev_state else btc_last_p
        btc_24h_chg = float(btc_ticker.get("priceChangePercent", 0.0))

        btc_regime = self.regime_engine.evaluate_btc_regime(
            btc_last_price=btc_last_p,
            btc_prev_price=btc_prev_p,
            btc_change_24h_pct=btc_24h_chg,
        )

        # Step 4: Parallel bounded scan across universe
        sem = asyncio.Semaphore(self.concurrency_limit)
        signals: List[SignalEvent] = []
        synthetic_liqs: List[SyntheticLiquidation] = []
        new_snapshots: List[MarketStateSnapshot] = []

        async def analyze_symbol(symbol: str) -> None:
            async with sem:
                ticker = all_tickers.get(symbol, {})
                premium = all_premium.get(symbol, {})

                last_price = float(ticker.get("lastPrice", 0.0))
                if last_price <= 0.0:
                    return

                quote_vol_24h = float(ticker.get("quoteVolume", 0.0))
                high_price = float(ticker.get("highPrice", last_price * 1.01))
                low_price = float(ticker.get("lowPrice", last_price * 0.99))
                price_change_pct = float(ticker.get("priceChangePercent", 0.0))

                # Funding 8h & Basis spread
                raw_funding = float(premium.get("lastFundingRate", 0.0))
                mark_price = float(premium.get("markPrice", last_price))
                index_price = float(premium.get("indexPrice", last_price))
                next_funding_time_ms = int(premium.get("nextFundingTime", 0))
                norm_8h_funding = (1.0 + raw_funding) - 1.0
                basis_bps = ((mark_price - index_price) / index_price * 10000.0) if index_price > 0 else 0.0

                # Open Interest
                oi_data = await self.ingestion.fetch_symbol_open_interest(symbol)
                curr_oi = float(oi_data.get("openInterest", 0.0)) if oi_data else 0.0

                p_snap = prev_state.get(symbol)
                delta_oi_5m = (curr_oi - p_snap.open_interest) if p_snap else 0.0

                # Order Book Depth & Spread (Numba JIT)
                ob = await self.ingestion.fetch_symbol_orderbook_top(symbol, limit=20)
                spread_bps = ob.spread_bps if ob else 1.0
                if ob and len(ob.bids) > 0 and len(ob.asks) > 0:
                    bids_qty = np.array([q for _, q in ob.bids], dtype=np.float64)
                    asks_qty = np.array([q for _, q in ob.asks], dtype=np.float64)
                    obi = compute_weighted_obi_jit(bids_qty, asks_qty, decay=0.85)
                else:
                    obi = 0.0

                # Anti-Spoofing & Quality Gate
                q_res = self.quality_filter.evaluate(
                    symbol=symbol,
                    quote_volume_24h=quote_vol_24h,
                    spread_bps=spread_bps,
                    funding_rate_8h=norm_8h_funding,
                )
                if not q_res.is_valid:
                    # Skip low-liquidity or wide-spread spoofed books
                    return

                # Calculate 5m Price Change & Relative Strength vs BTC
                alt_change_5m_pct = ((last_price - p_snap.last_price) / p_snap.last_price * 100.0) if p_snap else 0.0
                rs_to_btc = self.regime_engine.calculate_relative_strength(
                    alt_change_5m_pct=alt_change_5m_pct,
                    btc_change_5m_pct=btc_regime.btc_change_5m_pct,
                )

                # Trades, VPIN & CVD (limit 100 for minimal API weight)
                session = await self.ingestion._get_session()
                url = f"{self.ingestion.REST_BASE_URL}/fapi/v1/aggTrades"
                trade_qtys = []
                is_buyer = []
                taker_buy = 0.0
                taker_sell = 0.0

                try:
                    async with session.get(url, params={"symbol": symbol, "limit": "100"}) as resp:
                        if resp.status == 200:
                            raw_trades = await resp.json()
                            for t in raw_trades:
                                q = float(t.get("q", 0.0))
                                m = bool(t.get("m", False))
                                trade_qtys.append(q)
                                b = not m
                                is_buyer.append(b)
                                if b:
                                    taker_buy += q
                                else:
                                    taker_sell += q
                        elif resp.status == 429:
                            await asyncio.sleep(0.5)
                except Exception:
                    pass

                tot_taker = taker_buy + taker_sell
                cvd = taker_buy - taker_sell

                if trade_qtys and tot_taker > 0.0:
                    t_arr = np.array(trade_qtys, dtype=np.float64)
                    b_arr = np.array(is_buyer, dtype=np.bool_)
                    bucket_vol = max(tot_taker / 10.0, 1e-4)
                    vpin = compute_vpin_numba(t_arr, b_arr, bucket_vol=bucket_vol, window_baskets=10)
                else:
                    vpin = 0.3

                # CVD Divergence Check
                # FIX [C1]: Need at least 4-12 points of history for meaningful divergence detection.
                # Previously: only 2 points (prev + current) were passed with lookback=2,
                # but detect_cvd_divergence_jit returns 0.0 when n < 4.
                # Solution: Build history from stored state if available, otherwise skip.
                div_score = 0.0
                if p_snap is not None and hasattr(p_snap, 'cvd_history_5m') and p_snap.cvd_history_5m:
                    # Use stored 5m CVD history (should have 4-12 points)
                    cvd_hist = list(p_snap.cvd_history_5m) + [cvd]
                    price_hist = list(p_snap.price_history_5m) + [last_price]
                    if len(cvd_hist) >= 4:
                        prices_arr = np.array(price_hist[-12:], dtype=np.float64)  # Last 12 points max
                        cvd_arr = np.array(cvd_hist[-12:], dtype=np.float64)
                        lookback = min(6, len(prices_arr) - 1)  # Use 6 or less depending on data
                        div_score, _ = detect_cvd_divergence_jit(prices_arr, cvd_arr, lookback=lookback)

                # Liquidity Sweep & Reclaim Check (Wyckoff Spring / Upthrust)
                # FIX [C2]: Use 5m OHLC from historical bars, not 24h ticker extremes.
                # Previously: high_price/low_price were 24h extremes from ticker, causing false
                # sweep detection at 24h window boundaries.
                # Solution: Store and use actual 5m swing highs/lows from recent bars.
                # For now, we use a conservative approach: only detect sweeps if we have
                # stored swing levels from previous cycles that are distinct from current 24h range.
                
                has_sweep_reclaim = False
                if p_snap and p_snap.low_24h > 0.0 and p_snap.high_24h > 0.0:
                    # Check if 24h extremes are meaningfully different from current bar
                    # to avoid false positives at 24h window boundaries
                    swing_range_pct = ((p_snap.high_24h - p_snap.low_24h) / p_snap.low_24h) * 100.0
                    curr_range_pct = ((high_price - low_price) / low_price) * 100.0
                    
                    # Only use 24h extremes as swing levels if they represent a wider range
                    # than the current bar (i.e., not just boundary artifacts)
                    if swing_range_pct > curr_range_pct * 1.5:  # 24h range should be significantly wider
                        recent_low_swing = p_snap.low_24h
                        recent_high_swing = p_snap.high_24h
                    else:
                        # Fall back to current bar - no sweep detection this cycle
                        recent_low_swing = low_price
                        recent_high_swing = high_price
                else:
                    # First run: no prior state, skip sweep detection
                    recent_low_swing = low_price
                    recent_high_swing = high_price

                # Only attempt sweep detection if swing levels are meaningful
                if recent_low_swing != low_price or recent_high_swing != high_price:
                    sweep_event = self.sweep_detector.detect(
                        symbol=symbol,
                        current_price=last_price,
                        current_high=high_price,
                        current_low=low_price,
                        recent_swing_high=recent_high_swing,
                        recent_swing_low=recent_low_swing,
                        cvd_delta=cvd,
                    )
                    has_sweep_reclaim = sweep_event is not None and sweep_event.is_confirmed

                # Synthetic Liquidation Check
                liq = self.liq_detector.detect(
                    symbol=symbol,
                    current_price=last_price,
                    price_change_pct=price_change_pct,
                    delta_oi=delta_oi_5m,
                    taker_buy_vol=taker_buy,
                    taker_sell_vol=taker_sell,
                )
                if liq:
                    synthetic_liqs.append(liq)

                # Gate Evaluations (BTC Correlation & Funding Epoch)
                # FIX Bug 3: Gates are evaluated for BOTH directions and combined.
                # Previously, we pre-guessed signal direction from div_score alone,
                # which could be wrong when whale sentiment overrides the CVD direction.
                # Now we evaluate both directions and let compute_signal() decide which applies.
                btc_long_allowed, btc_long_reason = self.regime_engine.check_signal_gate(
                    symbol=symbol,
                    signal_type="STRONG_LONG",
                    btc_regime=btc_regime,
                    alt_change_5m_pct=alt_change_5m_pct,
                )
                btc_short_allowed, btc_short_reason = self.regime_engine.check_signal_gate(
                    symbol=symbol,
                    signal_type="STRONG_SHORT",
                    btc_regime=btc_regime,
                    alt_change_5m_pct=alt_change_5m_pct,
                )

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

                # Build separate gate_status strings for each direction.
                # compute_signal() will apply the correct one based on the final signal type.
                if not btc_long_allowed:
                    gate_long_status = btc_long_reason
                elif not funding_gate_long.allow_long:
                    gate_long_status = funding_gate_long.gate_reason
                else:
                    gate_long_status = "PASSED"

                if not btc_short_allowed:
                    gate_short_status = btc_short_reason
                elif not funding_gate_short.allow_short:
                    gate_short_status = funding_gate_short.gate_reason
                else:
                    gate_short_status = "PASSED"

                # Sentiment Divergence:
                # Fetch for BTC/ETH/SOL always (macro anchors) and for any symbol
                # showing meaningful CVD divergence direction (|div_score| > 0.3).
                # Lower threshold vs. previous 0.5 ensures whale factor is included
                # early enough to boost/block signals before they cross the 75pt gate.
                z_whale = 0.0
                if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT") or abs(div_score) > 0.3:
                    sent = await self.sentiment_engine.fetch_sentiment_divergence(symbol)
                    if sent:
                        z_whale = sent.z_whale_sentiment

                # Compute Final Institutional Signal
                sig = self.signal_engine.compute_signal(
                    symbol=symbol,
                    current_price=last_price,
                    funding_rate_8h=norm_8h_funding,
                    basis_spread_bps=basis_bps,
                    delta_oi=delta_oi_5m,
                    oi_total=curr_oi,
                    obi=obi,
                    vpin=vpin,
                    cvd_divergence_score=div_score,
                    recent_high=high_price,
                    recent_low=low_price,
                    z_whale_sentiment=z_whale,
                    relative_strength=rs_to_btc,
                    sweep_reclaim=has_sweep_reclaim,
                    gate_long_status=gate_long_status,
                    gate_short_status=gate_short_status,
                )
                signals.append(sig)

                # FIX [C1]: Maintain rolling 5m CVD/price history for divergence detection
                # Keep last N=12 points (enough for lookback=6 with n>=4 requirement)
                MAX_HISTORY = 12
                if p_snap is not None and hasattr(p_snap, 'cvd_history_5m') and p_snap.cvd_history_5m:
                    cvd_hist = list(p_snap.cvd_history_5m)[-MAX_HISTORY+1:] + [cvd]
                    price_hist = list(p_snap.price_history_5m)[-MAX_HISTORY+1:] + [last_price]
                else:
                    cvd_hist = [cvd]
                    price_hist = [last_price]

                new_snapshots.append(
                    MarketStateSnapshot(
                        symbol=symbol,
                        timestamp_ms=int(time.time() * 1000),
                        last_price=last_price,
                        open_interest=curr_oi,
                        delta_oi_5m=delta_oi_5m,
                        cumulative_cvd_5m=cvd,
                        funding_rate_8h=norm_8h_funding,
                        basis_bps=basis_bps,
                        vpin_estimate=vpin,
                        obi_score=obi,
                        composite_score=sig.composite_score,
                        low_24h=low_price,
                        high_24h=high_price,
                        whale_sentiment_z=z_whale,
                        cvd_history_5m=tuple(cvd_hist[-MAX_HISTORY:]),
                        price_history_5m=tuple(price_hist[-MAX_HISTORY:]),
                    )
                )

        tasks = [analyze_symbol(sym) for sym in sorted_symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.ingestion.stop()
        await self.sentiment_engine.close()

        # Step 5: Persist state
        self.save_current_state(new_snapshots)

        # Sort signals descending by score
        signals.sort(key=lambda s: s.composite_score, reverse=True)

        strong_longs = [s for s in signals if s.signal_type == "STRONG_LONG"]
        strong_shorts = [s for s in signals if s.signal_type == "STRONG_SHORT"]
        duration = time.time() - t0

        summary = ScreenerResult(
            timestamp_ms=int(time.time() * 1000),
            duration_sec=round(duration, 2),
            total_scanned=len(signals),
            strong_longs_count=len(strong_longs),
            strong_shorts_count=len(strong_shorts),
            synthetic_liqs_count=len(synthetic_liqs),
            btc_regime=btc_regime.status,
            btc_change_5m_pct=btc_regime.btc_change_5m_pct,
        )

        return signals, synthetic_liqs, summary
