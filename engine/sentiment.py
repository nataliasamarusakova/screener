"""
Smart Money vs. Retail Crowd Sentiment Engine.

The top-trader position endpoint is MARKET_DATA and requires X-MBX-APIKEY.
The returned divergence_score is a bounded normalized factor, not a statistical Z-score;
QuantSignalEngine performs the point-in-time empirical Z-score conversion.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import asyncio
import aiohttp
import msgspec

logger = logging.getLogger("sentiment")


class SentimentDivergence(msgspec.Struct, gc=False):
    symbol: str
    retail_ls_ratio: float
    retail_long_pct: float
    top_traders_ls_ratio: float
    top_traders_long_pct: float
    taker_buy_sell_ratio: float
    divergence_score: float
    z_whale_sentiment: float  # Deprecated compatibility field; not used by composite scoring.
    sentiment_bias: str
    observation_timestamp_ms: int = 0  # Backward-compatible persisted field.


class SentimentEngine:
    BASE_URL = "https://fapi.binance.com"

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        api_key: Optional[str] = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self._session = session
        self.api_key = api_key if api_key is not None else os.getenv("BINANCE_API_KEY")
        self.request_timeout_seconds = request_timeout_seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
            headers = {"User-Agent": "QuantEngine/1.0", "Accept-Encoding": "gzip"}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    @staticmethod
    def _select_observation(data: object, start_time_ms: int | None, end_time_ms: int | None) -> Optional[dict]:
        if not isinstance(data, list):
            return None
        candidates = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                ts = int(item["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if start_time_ms is not None and ts < start_time_ms:
                continue
            if end_time_ms is not None and ts > end_time_ms:
                continue
            candidates.append((ts, item))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    async def fetch_sentiment_divergence(
        self,
        symbol: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> Optional[SentimentDivergence]:
        """Fetch a single 5m sentiment observation inside an explicit PIT window."""
        if not self.api_key:
            logger.error("missing_binance_api_key endpoint=topLongShortPositionRatio symbol=%s", symbol)
            return None
        if (start_time_ms is None) != (end_time_ms is None):
            raise ValueError("start_time_ms and end_time_ms must be supplied together")
        if start_time_ms is not None and (start_time_ms < 0 or end_time_ms < start_time_ms):
            raise ValueError("invalid sentiment time window")

        session = await self._get_session()
        retail_url = f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio"
        whale_url = f"{self.BASE_URL}/futures/data/topLongShortPositionRatio"
        taker_url = f"{self.BASE_URL}/futures/data/takerlongshortRatio"
        params = {"symbol": symbol, "period": "5m", "limit": "1"}
        if start_time_ms is not None and end_time_ms is not None:
            params["startTime"] = str(start_time_ms)
            params["endTime"] = str(end_time_ms)

        try:
            async with session.get(retail_url, params=params) as resp_r, \
                       session.get(whale_url, params=params) as resp_w, \
                       session.get(taker_url, params=params) as resp_t:
                if resp_r.status != 200 or resp_w.status != 200 or resp_t.status != 200:
                    logger.error(
                        "sentiment_http_error symbol=%s retail=%s whale=%s taker=%s",
                        symbol, resp_r.status, resp_w.status, resp_t.status,
                    )
                    return None

                data_r = await resp_r.json()
                data_w = await resp_w.json()
                data_t = await resp_t.json()
                r_item = self._select_observation(data_r, start_time_ms, end_time_ms)
                w_item = self._select_observation(data_w, start_time_ms, end_time_ms)
                t_item = self._select_observation(data_t, start_time_ms, end_time_ms)
                if r_item is None or w_item is None or t_item is None:
                    logger.error("sentiment_missing_window symbol=%s start=%s end=%s", symbol, start_time_ms, end_time_ms)
                    return None

                retail_ts = int(r_item["timestamp"])
                whale_ts = int(w_item["timestamp"])
                taker_ts = int(t_item["timestamp"])

                # Binance uses different timestamp semantics for these 5m endpoints:
                # global/top-trader ratios are timestamped at period END, while the
                # taker buy/sell ratio is timestamped at period START.
                if start_time_ms is not None and end_time_ms is not None:
                    if retail_ts != end_time_ms or whale_ts != end_time_ms or taker_ts != start_time_ms:
                        logger.error(
                            "sentiment_timestamp_mismatch symbol=%s retail=%s whale=%s taker=%s expected_end=%s expected_start=%s",
                            symbol, retail_ts, whale_ts, taker_ts, end_time_ms, start_time_ms,
                        )
                        return None
                        
                elif retail_ts != whale_ts or retail_ts != taker_ts:
                    logger.error(
                        "sentiment_timestamp_mismatch symbol=%s timestamps=%s",
                        symbol, (retail_ts, whale_ts, taker_ts),
                    )
                    return None

                retail_ls = float(r_item["longShortRatio"])
                retail_long = float(r_item["longAccount"]) * 100.0
                whale_ls = float(w_item["longShortRatio"])
                whale_long = float(w_item["longAccount"]) * 100.0
                taker_buy_sell_ratio = float(t_item["buySellRatio"])

                values = (retail_ls, retail_long, whale_ls, whale_long, taker_buy_sell_ratio)
                if not all(math.isfinite(x) for x in values):
                    logger.error("sentiment_non_finite symbol=%s", symbol)
                    return None
                if min(retail_ls, whale_ls, taker_buy_sell_ratio) <= 0.0:
                    logger.error("sentiment_non_positive_ratio symbol=%s", symbol)
                    return None

                whale_retail_diff = math.log(whale_ls) - math.log(retail_ls)
                taker_log = math.log(taker_buy_sell_ratio)
                raw_diff = 0.70 * whale_retail_diff + 0.30 * taker_log
                div_score = max(-1.0, min(1.0, raw_diff / 1.5))

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
                    z_whale_sentiment=round(div_score * 3.0, 2),
                    sentiment_bias=bias,
                    observation_timestamp_ms=max(retail_ts, whale_ts, taker_ts),
                )

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as exc:
            logger.error("sentiment_fetch_failed symbol=%s error=%s", symbol, exc)
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
