"""
Smart Money vs. Retail Crowd Sentiment Engine.
Analyzes three Binance Futures sentiment endpoints:
1. globalLongShortAccountRatio  — % of retail accounts long vs short (crowd positioning)
2. topLongShortPositionRatio    — net positions of top-20% whale traders (smart money)
3. takerlongshortRatio          — aggressive taker buy vs sell volume (momentum confirmation)

Detects Whale vs Retail divergence and taker flow confirmation traps.
"""
from __future__ import annotations

import math
from typing import Optional
import aiohttp
import msgspec


class SentimentDivergence(msgspec.Struct, gc=False):
    symbol: str
    retail_ls_ratio: float               # Retail long/short account ratio
    retail_long_pct: float               # Retail long accounts %
    top_traders_ls_ratio: float          # Whale top trader position ratio
    top_traders_long_pct: float          # Whale long position %
    taker_buy_sell_ratio: float          # Aggressive taker buy/sell volume ratio (>1 = net buy)
    divergence_score: float              # [-1.0, 1.0] (positive = whales long / retail short)
    z_whale_sentiment: float             # Z-score contribution [-3.0, 3.0]
    sentiment_bias: str                  # "SMART_MONEY_LONG", "RETAIL_TRAP_SHORT", "NEUTRAL"


class SentimentEngine:
    """
    Asynchronously queries Binance sentiment endpoints and computes Whale-Retail divergence.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def fetch_sentiment_divergence(self, symbol: str) -> Optional[SentimentDivergence]:
        """
        Fetches retail, top trader, and taker long/short ratios for a symbol
        and computes a composite Whale-Retail-Taker divergence score.

        Endpoints used:
          /futures/data/globalLongShortAccountRatio — retail crowd positioning
          /futures/data/topLongShortPositionRatio   — top-20% whale net positions
          /futures/data/takerlongshortRatio         — aggressive taker flow direction

        Divergence logic:
          Primary (70%): log(Whale_LS) − log(Retail_LS)
            > 0: Whales long, retail short → bearish retail trap → BULLISH
            < 0: Whales short, retail long (FOMO crowd) → liquidation cascade → BEARISH
          Secondary (30%): log(buySellRatio)
            > 0: Aggressive buy takers confirm bullish momentum
            < 0: Aggressive sell takers confirm bearish momentum
        """
        session = await self._get_session()
        retail_url = f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio"
        whale_url = f"{self.BASE_URL}/futures/data/topLongShortPositionRatio"
        taker_url = f"{self.BASE_URL}/futures/data/takerlongshortRatio"

        params = {"symbol": symbol, "period": "5m", "limit": "1"}

        try:
            # Parallel fetch of all three sentiment endpoints
            async with session.get(retail_url, params=params) as resp_r, \
                       session.get(whale_url, params=params) as resp_w, \
                       session.get(taker_url, params=params) as resp_t:

                if resp_r.status != 200 or resp_w.status != 200:
                    return None

                data_r = await resp_r.json()
                data_w = await resp_w.json()
                data_t = await resp_t.json() if resp_t.status == 200 else []

                if not data_r or not data_w:
                    return None

                r_item = data_r[0]
                w_item = data_w[0]

                retail_ls = float(r_item.get("longShortRatio", 1.0))
                retail_long = float(r_item.get("longAccount", 0.5)) * 100.0

                whale_ls = float(w_item.get("longShortRatio", 1.0))
                whale_long = float(w_item.get("longAccount", 0.5)) * 100.0

                # Taker long/short ratio (aggressive flow direction)
                # buySellRatio > 1.0 = more aggressive buy takers (bullish momentum)
                taker_buy_sell_ratio = 1.0
                if data_t:
                    t_item = data_t[0]
                    taker_buy_sell_ratio = float(t_item.get("buySellRatio", 1.0))

                # --- Primary Divergence Component (weight: 70%) ---
                # log(Whale_LS) - log(Retail_LS):
                #   Whales=2.0 (66% long), Retail=0.8 (44% long) → log(2.0)-log(0.8) = +0.916 (Bullish)
                #   Whales=0.5 (33% long), Retail=2.5 (71% long) → log(0.5)-log(2.5) = -1.609 (Bearish trap)
                log_whale = math.log(max(whale_ls, 0.01))
                log_retail = math.log(max(retail_ls, 0.01))
                whale_retail_diff = log_whale - log_retail

                # --- Secondary Taker Flow Component (weight: 30%) ---
                # log(buySellRatio):
                #   buySellRatio=1.5 → log(1.5) = +0.405 (net buy pressure)
                #   buySellRatio=0.7 → log(0.7) = -0.357 (net sell pressure)
                taker_log = math.log(max(taker_buy_sell_ratio, 0.01))

                # Composite blend: primary divergence + taker momentum confirmation
                raw_diff = 0.70 * whale_retail_diff + 0.30 * taker_log

                # Normalize to [-1.0, 1.0] range (1.5 = ~4.5 sigma normalization factor)
                div_score = max(-1.0, min(1.0, raw_diff / 1.5))
                z_whale = max(-3.0, min(3.0, div_score * 3.0))

                if div_score >= 0.25:
                    bias = "SMART_MONEY_LONG"
                elif div_score <= -0.25:
                    bias = "RETAIL_TRAP_SHORT"
                else:
                    bias = "NEUTRAL"

                return SentimentDivergence(
                    symbol=symbol,
                    retail_ls_ratio=round(retail_ls, 3),
                    retail_long_pct=round(retail_long, 1),
                    top_traders_ls_ratio=round(whale_ls, 3),
                    top_traders_long_pct=round(whale_long, 1),
                    taker_buy_sell_ratio=round(taker_buy_sell_ratio, 3),
                    divergence_score=round(div_score, 3),
                    z_whale_sentiment=round(z_whale, 2),
                    sentiment_bias=bias,
                )

        except Exception:
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
