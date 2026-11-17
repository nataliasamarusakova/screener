from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from . import __version__
from .models import Candle

log = logging.getLogger(__name__)

_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000)


class BinanceDataError(RuntimeError):
    pass


class BinanceScanAbortError(BinanceDataError):
    """A Binance rate-limit state where the rest of this scan must stop."""


class BinanceIPBanError(BinanceScanAbortError):
    """Fatal Binance IP ban; the current scan must stop issuing requests."""


class BinanceWAFBlockError(BinanceScanAbortError):
    """Binance WAF/HTTP 403 response; the current scan must stop."""


class BinanceRateLimitError(BinanceScanAbortError):
    """Repeated HTTP 429 responses; further requests in this scan are unsafe."""


class BinanceClient:
    """Public Binance USDⓈ-M Futures market-data client; never places orders."""

    def __init__(
        self,
        base_url: str,
        min_interval_sec: float = 0.12,
        timeout_sec: float = 10.0,
        *,
        http_proxy: str = "",
        vpn_enabled: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_sec = min_interval_sec
        self.timeout_sec = timeout_sec
        if vpn_enabled and not http_proxy:
            raise BinanceDataError("BINANCE_VPN_ENABLED=true requires BINANCE_HTTP_PROXY")
        if http_proxy and not http_proxy.startswith(("http://", "https://")):
            raise BinanceDataError("BINANCE_HTTP_PROXY must use http:// or https://")
        self.session = requests.Session()
        # Do not inherit runner-wide proxy variables: only Binance traffic uses WireProxy.
        self.session.trust_env = False
        if vpn_enabled:
            self.session.proxies.update({"http": http_proxy, "https": http_proxy})
        self.session.headers.update({"Accept": "application/json", "User-Agent": f"Market-Screener/{__version__}"})
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._symbols: set[str] | None = None
        self._ip_banned = False
        self._ip_ban_retry_after: float | None = None
        self._waf_blocked = False
        self._rate_limited = False
        self._rate_limit_retry_after: float | None = None

    def _pace(self) -> None:
        with self._lock:
            wait = self.min_interval_sec - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def _retry_after_seconds(headers: Any, fallback: float) -> float:
        """Parse Binance Retry-After without shortening the server's backoff."""
        raw = headers.get("Retry-After") if hasattr(headers, "get") else None
        if raw is None:
            return max(0.0, float(fallback))
        try:
            delay = float(str(raw).strip())
            if math.isfinite(delay) and delay >= 0:
                return delay
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
            if math.isfinite(delay):
                return max(0.0, delay)
        except (TypeError, ValueError, OverflowError):
            pass
        return max(0.0, float(fallback))

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self._ip_banned:
            detail = (
                f"; Retry-After={self._ip_ban_retry_after:g}s"
                if self._ip_ban_retry_after is not None else ""
            )
            raise BinanceIPBanError(
                "Binance client is blocked after HTTP 418; refusing all further requests" + detail
            )
        if self._waf_blocked:
            raise BinanceWAFBlockError(
                "Binance client is blocked after HTTP 403; refusing all further requests in this scan"
            )
        if self._rate_limited:
            detail = (
                f"; Retry-After={self._rate_limit_retry_after:g}s"
                if self._rate_limit_retry_after is not None else ""
            )
            raise BinanceRateLimitError(
                "Binance returned repeated HTTP 429 responses; refusing further requests in this scan" + detail
            )

        last_error: Exception | None = None
        for attempt in range(3):
            self._pace()
            try:
                response = self.session.get(f"{self.base_url}{path}", params=params or {}, timeout=self.timeout_sec)
                if response.status_code == 403:
                    self._waf_blocked = True
                    raise BinanceWAFBlockError(
                        "Binance returned HTTP 403 (WAF/security block); stopping the entire scan immediately."
                    )
                if response.status_code == 418:
                    self._ip_banned = True
                    self._ip_ban_retry_after = self._retry_after_seconds(response.headers, 0.0)
                    detail = (
                        f" Retry-After={self._ip_ban_retry_after:g}s."
                        if self._ip_ban_retry_after > 0 else ""
                    )
                    raise BinanceIPBanError(
                        "Binance returned HTTP 418 (IP ban); stopping the entire scan immediately."
                        + detail
                    )
                if response.status_code == 429:
                    delay = self._retry_after_seconds(response.headers, 2 ** attempt)
                    if attempt == 2:
                        self._rate_limited = True
                        self._rate_limit_retry_after = delay
                        raise BinanceRateLimitError(
                            "Binance returned HTTP 429 on all three attempts; stopping the entire scan to avoid escalation"
                            + (f". Retry-After={delay:g}s" if delay > 0 else "")
                        )
                    log.warning(
                        "Binance rate limit HTTP 429 for %s; waiting %.3fs before retry %d/3",
                        path, delay, attempt + 2,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
                if 400 <= response.status_code < 500:
                    raise BinanceDataError(f"Binance rejected {path} with HTTP {response.status_code}; not retrying a client error")
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(0.5 * (2 ** attempt))
        raise BinanceDataError(f"Binance request failed for {path}: {last_error}") from last_error

    def _load_symbols(self) -> set[str]:
        if self._symbols is not None:
            return self._symbols
        payload = self._get_json("/fapi/v1/exchangeInfo")
        symbols: set[str] = set()
        for item in payload.get("symbols", []) if isinstance(payload, dict) else []:
            if (
                item.get("status") == "TRADING"
                and item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT"
                and isinstance(item.get("symbol"), str)
            ):
                symbols.add(item["symbol"].upper())
        if not symbols:
            raise BinanceDataError("Binance exchangeInfo returned no active USDT perpetual symbols")
        self._symbols = symbols
        return symbols

    @staticmethod
    def _base_symbol(coinalyze_symbol: str) -> str:
        value = str(coinalyze_symbol or "").upper().strip().split(".", 1)[0]
        value = re.sub(r"[_-]?(PERPETUAL|PERP|SWAP)$", "", value)
        value = value.replace("/", "").replace("-", "").replace("_", "")
        # Historical Coinalyze market labels may use USD/USDC/BUSD quote suffixes;
        # P1 runs on Binance USD-M USDT perpetuals, so map the base only.
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if value.endswith(quote):
                value = value[:-len(quote)]
                break
        return value

    def resolve_symbol(self, coinalyze_symbol: str) -> str | None:
        raw = str(coinalyze_symbol or "").upper().strip().split(".", 1)[0]
        raw = re.sub(r"[_-]?(PERPETUAL|PERP|SWAP)$", "", raw).replace("/", "").replace("-", "").replace("_", "")
        symbols = self._load_symbols()
        if raw in symbols:
            return raw
        base = self._base_symbol(coinalyze_symbol)
        candidate = base + "USDT"
        if candidate in symbols:
            return candidate
        # Binance uses multiplier-prefixed contracts for some small-denomination
        # assets (e.g. 1000PEPEUSDT). Use only an unambiguous active listing.
        aliases = {prefix + candidate for prefix in ("1000", "10000", "100000", "1000000", "1M")}
        matches = sorted(aliases & symbols)
        return matches[0] if len(matches) == 1 else None

    def fetch_candles(self, coinalyze_symbol: str, interval: str, limit: int = 300) -> list[Candle]:
        resolved = self.resolve_symbol(coinalyze_symbol)
        if not resolved:
            raise BinanceDataError(f"No active Binance USDT perpetual for Coinalyze symbol {coinalyze_symbol}")
        durations = {
            "5m": 5 * 60_000, "15m": 15 * 60_000, "30m": 30 * 60_000,
            "1h": 60 * 60_000, "2h": 2 * 60 * 60_000, "4h": 4 * 60 * 60_000,
            "6h": 6 * 60 * 60_000, "12h": 12 * 60 * 60_000, "1d": 24 * 60 * 60_000,
        }
        if interval not in durations:
            raise ValueError(f"Unsupported Binance candle interval: {interval}")
        duration_ms = durations[interval]
        rows = self._get_json("/fapi/v1/klines", {"symbol": resolved, "interval": interval, "limit": min(max(int(limit), 100), 1500)})
        if not isinstance(rows, list):
            raise BinanceDataError(f"Unexpected Binance klines response for {resolved}/{interval}")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        candles: list[Candle] = []
        for raw in rows:
            if not isinstance(raw, (list, tuple)) or len(raw) < 9:
                continue
            try:
                candle = Candle(
                    open_time_ms=int(raw[0]),
                    close_time_ms=int(raw[6]),
                    open=float(raw[1]),
                    high=float(raw[2]),
                    low=float(raw[3]),
                    close=float(raw[4]),
                    volume=float(raw[5]),
                    quote_volume=float(raw[7]),
                    trades=int(raw[8]),
                )
            except (TypeError, ValueError):
                continue
            numeric = (candle.open, candle.high, candle.low, candle.close, candle.volume, candle.quote_volume)
            sane = (
                all(math.isfinite(value) for value in numeric)
                and candle.open > 0 and candle.high > 0 and candle.low > 0 and candle.close > 0
                and candle.volume >= 0 and candle.quote_volume >= 0 and candle.trades >= 0
                and candle.high >= max(candle.open, candle.close, candle.low)
                and candle.low <= min(candle.open, candle.close, candle.high)
                and candle.close_time_ms == candle.open_time_ms + duration_ms - 1
            )
            # The last API row may still be forming. Invalid rows are discarded,
            # malformed rows are omitted; continuity is evaluated on the latest tail below.
            if sane and candle.close_time_ms < now_ms:
                candles.append(candle)
        candles.sort(key=lambda c: c.open_time_ms)
        unique = {c.open_time_ms: c for c in candles}
        result = [unique[k] for k in sorted(unique)]
        # Older history can contain exchange/data-provider gaps. Keep only the
        # latest continuous segment: a gap before this segment must not discard
        # otherwise valid recent chart data, but no gap inside the analysis tail
        # may be silently compressed.
        last_gap = next((idx for idx in range(len(result) - 1, 0, -1)
                         if result[idx].open_time_ms - result[idx - 1].open_time_ms != duration_ms), None)
        if last_gap is not None:
            result = result[last_gap:]
        if len(result) < 20:
            raise BinanceDataError(f"Only {len(result)} consecutive valid closed {interval} candles for {resolved}; need at least 20")
        latest_age = now_ms - result[-1].close_time_ms
        if latest_age > duration_ms + 120_000:
            raise BinanceDataError(f"Stale {interval} candles for {resolved}: latest closed candle is {latest_age / 60_000:.1f} minutes old")
        return result

    def fetch_orderbook_diagnostics(
        self, coinalyze_symbol: str, *, limit: int = 100, notional_usd: float = 0.0
    ) -> dict[str, Any]:
        """Read-only top-of-book/depth diagnostics; this method never creates orders."""
        resolved = self.resolve_symbol(coinalyze_symbol)
        if not resolved:
            raise BinanceDataError(f"No active Binance USDT perpetual for Coinalyze symbol {coinalyze_symbol}")
        if not math.isfinite(float(notional_usd)) or float(notional_usd) < 0:
            raise BinanceDataError("Execution notional must be a finite non-negative USD amount")
        requested_limit = min(max(int(limit), _DEPTH_LIMITS[0]), _DEPTH_LIMITS[-1])
        # USDⓈ-M /fapi/v1/depth accepts only these discrete limits.
        safe_limit = next((allowed for allowed in _DEPTH_LIMITS if allowed >= requested_limit), _DEPTH_LIMITS[-1])
        payload = self._get_json("/fapi/v1/depth", {"symbol": resolved, "limit": safe_limit})
        if not isinstance(payload, dict):
            raise BinanceDataError(f"Unexpected Binance depth response for {resolved}")

        def levels(key: str) -> list[tuple[float, float]]:
            result: list[tuple[float, float]] = []
            for item in payload.get(key, []):
                try:
                    price, qty = float(item[0]), float(item[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(price) and math.isfinite(qty) and price > 0 and qty > 0:
                    result.append((price, qty))
            return result

        bids, asks = levels("bids"), levels("asks")
        if not bids or not asks:
            raise BinanceDataError(f"Empty bid/ask side in Binance depth for {resolved}")
        best_bid = max(price for price, _ in bids)
        best_ask = min(price for price, _ in asks)
        if best_ask <= best_bid:
            raise BinanceDataError(f"Crossed/invalid Binance order book for {resolved}: bid={best_bid}, ask={best_ask}")
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000.0

        def depth_within(level_data: list[tuple[float, float]], lower: float, upper: float) -> float:
            return sum(price * qty for price, qty in level_data if lower <= price <= upper)

        bid_depth_1pct = depth_within(bids, mid * 0.99, mid)
        ask_depth_1pct = depth_within(asks, mid, mid * 1.01)
        total_depth_1pct = bid_depth_1pct + ask_depth_1pct
        depth_imbalance = (bid_depth_1pct - ask_depth_1pct) / total_depth_1pct if total_depth_1pct else None

        def estimate_vwap(level_data: list[tuple[float, float]], requested_usd: float) -> dict[str, float | None]:
            if requested_usd <= 0:
                return {"filled_usd": None, "average_price": None, "slippage_bps": None, "unfilled_usd": None}
            remaining = requested_usd
            filled_usd = 0.0
            base_qty = 0.0
            for price, qty in sorted(level_data, key=lambda x: x[0], reverse=(level_data is bids)):
                available_usd = price * qty
                take_usd = min(remaining, available_usd)
                if take_usd > 0:
                    filled_usd += take_usd
                    base_qty += take_usd / price
                    remaining -= take_usd
                if remaining <= 1e-9:
                    break
            average_price = filled_usd / base_qty if base_qty else None
            reference = best_ask if level_data is asks else best_bid
            if average_price is None or not reference:
                slippage = None
            elif level_data is asks:
                slippage = (average_price / reference - 1.0) * 10_000.0
            else:
                # Sell-side shortfall is measured against the best bid, using
                # the same benchmark denominator as buy-side shortfall.
                slippage = (reference - average_price) / reference * 10_000.0
            return {
                "filled_usd": filled_usd,
                "average_price": average_price,
                "slippage_bps": slippage,
                "unfilled_usd": max(0.0, remaining),
            }

        buy = estimate_vwap(asks, notional_usd)
        sell = estimate_vwap(bids, notional_usd)
        return {
            "symbol": resolved,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Binance USDⓈ-M Futures depth snapshot",
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "bid_depth_1pct_usd": bid_depth_1pct,
            "ask_depth_1pct_usd": ask_depth_1pct,
            "depth_imbalance_1pct": depth_imbalance,
            "notional_usd": notional_usd if notional_usd > 0 else None,
            "market_buy_estimate": buy,
            "market_sell_estimate": sell,
            "estimated_only": True,
        }

