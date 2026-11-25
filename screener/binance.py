from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from . import __version__
from .models import Candle

log = logging.getLogger(__name__)

_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000)
_SPOT_MULTIPLIER_PREFIXES: tuple[tuple[str, float], ...] = (
    ("1000000", 1_000_000.0),
    ("100000", 100_000.0),
    ("10000", 10_000.0),
    ("1000", 1_000.0),
    ("1M", 1_000_000.0),
)


@dataclass(frozen=True, slots=True)
class SpotSymbolResolution:
    futures_symbol: str
    spot_symbol: str
    futures_base_asset: str
    spot_base_asset: str
    quote_asset: str
    base_multiplier: float = 1.0


class SpotDataError(RuntimeError):
    """A non-fatal Spot market-data error; it must not abort the Futures scan."""


class BinanceDataError(RuntimeError):
    pass


class BinanceScanAbortError(BinanceDataError):
    """A Binance rate-limit state where the rest of this scan must stop."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class BinanceIPBanError(BinanceScanAbortError):
    """Fatal Binance IP ban; the current scan must stop issuing requests."""


class BinanceWAFBlockError(BinanceScanAbortError):
    """Binance WAF/HTTP 403 response; the current scan must stop."""


class BinanceRateLimitError(BinanceScanAbortError):
    """Repeated HTTP 429 responses; further requests in this scan are unsafe."""


class BinanceGeoRestrictionError(BinanceScanAbortError):
    """HTTP 451 means the API is unavailable from this network/region."""


class BinanceCooldownActiveError(BinanceScanAbortError):
    """A previously persisted Binance Retry-After window is still active."""


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
                        + detail,
                        retry_after_seconds=self._ip_ban_retry_after,
                    )
                if response.status_code == 451:
                    raise BinanceGeoRestrictionError(
                        "Binance returned HTTP 451 (Unavailable For Legal Reasons); "
                        "stopping the entire scan because market data is unavailable from this network/region."
                    )
                if response.status_code == 429:
                    delay = self._retry_after_seconds(response.headers, 2 ** attempt)
                    if attempt == 2:
                        self._rate_limited = True
                        self._rate_limit_retry_after = delay
                        raise BinanceRateLimitError(
                            "Binance returned HTTP 429 on all three attempts; stopping the entire scan to avoid escalation"
                            + (f". Retry-After={delay:g}s" if delay > 0 else ""),
                            retry_after_seconds=delay,
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

    def fetch_open_interest_history(
        self,
        symbol: str,
        *,
        period: str = "5m",
        limit: int = 500,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch USDⓈ-M open-interest history without an API key.

        Base-unit OI and USD-valued OI are kept as separate source fields. The
        response is not forward-filled: missing intervals remain missing.
        """
        valid_periods = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        if period not in valid_periods:
            raise ValueError(f"Unsupported Binance OI period: {period}")
        safe_limit = min(max(int(limit), 1), 500)
        params: dict[str, Any] = {"symbol": str(symbol).upper(), "period": period, "limit": safe_limit}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        rows = self._get_json("/futures/data/openInterestHist", params)
        if not isinstance(rows, list):
            raise BinanceDataError(f"Unexpected Binance openInterestHist response for {symbol}")
        result: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                timestamp = int(item["timestamp"])
                oi_base_raw = str(item["sumOpenInterest"])
                oi_usd_raw = str(item["sumOpenInterestValue"])
                oi_base = float(oi_base_raw)
                oi_usd = float(oi_usd_raw)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                timestamp < 0 or not math.isfinite(oi_base) or not math.isfinite(oi_usd)
                or oi_base < 0 or oi_usd < 0
            ):
                continue
            result.append({
                "symbol": str(item.get("symbol") or symbol).upper(),
                "timestamp": timestamp,
                "sumOpenInterest": oi_base_raw,
                "sumOpenInterestValue": oi_usd_raw,
            })
        unique = {item["timestamp"]: item for item in result}
        return [unique[key] for key in sorted(unique)]

    def fetch_funding_rate_history(
        self,
        symbol: str,
        *,
        limit: int = 200,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical funding records, preserving Regular/Special rateType."""
        params: dict[str, Any] = {"symbol": str(symbol).upper(), "limit": min(max(int(limit), 1), 1000)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        rows = self._get_json("/fapi/v1/fundingRate", params)
        if not isinstance(rows, list):
            raise BinanceDataError(f"Unexpected Binance fundingRate response for {symbol}")
        result: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                funding_time = int(item["fundingTime"])
                rate_raw = str(item["fundingRate"])
                rate = float(rate_raw)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if funding_time < 0 or not math.isfinite(rate):
                continue
            mark_price_raw = item.get("markPrice")
            if mark_price_raw not in (None, ""):
                try:
                    mark_price = float(mark_price_raw)
                except (TypeError, ValueError, OverflowError):
                    mark_price = math.nan
                if not math.isfinite(mark_price) or mark_price < 0:
                    mark_price_raw = None
                else:
                    mark_price_raw = str(mark_price_raw)
            else:
                mark_price_raw = None
            rate_type = item.get("rateType")
            if rate_type is not None:
                rate_type = str(rate_type)
                if rate_type not in {"Regular", "Special"}:
                    log.warning("Unknown Binance funding rateType=%r for %s; preserving raw value", rate_type, symbol)
            result.append({
                "symbol": str(item.get("symbol") or symbol).upper(),
                "fundingTime": funding_time,
                "fundingRate": rate_raw,
                "markPrice": mark_price_raw,
                "rateType": rate_type,
            })
        unique = {item["fundingTime"]: item for item in result}
        return [unique[key] for key in sorted(unique)]

    def fetch_candles(
        self,
        coinalyze_symbol: str,
        interval: str,
        limit: int = 300,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Candle]:
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
        params: dict[str, Any] = {
            "symbol": resolved, "interval": interval,
            "limit": min(max(int(limit), 1), 1500),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        rows = self._get_json("/fapi/v1/klines", params)
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
                    taker_buy_base_volume=float(raw[9]) if len(raw) > 9 else None,
                    taker_buy_quote_volume=float(raw[10]) if len(raw) > 10 else None,
                )
            except (TypeError, ValueError):
                continue
            numeric = (
                candle.open, candle.high, candle.low, candle.close,
                candle.volume, candle.quote_volume,
                *(() if candle.taker_buy_base_volume is None else (candle.taker_buy_base_volume,)),
                *(() if candle.taker_buy_quote_volume is None else (candle.taker_buy_quote_volume,)),
            )
            sane = (
                all(math.isfinite(value) for value in numeric)
                and candle.open > 0 and candle.high > 0 and candle.low > 0 and candle.close > 0
                and candle.volume >= 0 and candle.quote_volume >= 0 and candle.trades >= 0
                and (candle.taker_buy_base_volume is None or 0 <= candle.taker_buy_base_volume <= candle.volume)
                and (candle.taker_buy_quote_volume is None or 0 <= candle.taker_buy_quote_volume <= candle.quote_volume)
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



class SpotMarketClient:
    """Unauthenticated Binance Spot market-data client using a direct session.

    It deliberately does not inherit BinanceClient's WireProxy configuration.
    Spot availability/errors are non-fatal and must never cancel a Futures scan.
    """

    def __init__(
        self,
        base_url: str = "https://data-api.binance.vision",
        min_interval_sec: float = 0.08,
        timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"Accept": "application/json", "User-Agent": f"Market-Screener/{__version__}"})
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._exchange_info: list[dict[str, Any]] | None = None
        self._symbols_by_base_quote: dict[tuple[str, str], list[str]] | None = None

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            wait = self.min_interval_sec - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(
                    f"{self.base_url}{path}", params=params or {}, timeout=self.timeout_sec
                )
                if response.status_code in {418, 429}:
                    raise SpotDataError(f"Binance Spot rate limit HTTP {response.status_code} for {path}")
                if response.status_code == 400:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    if isinstance(payload, dict) and payload.get("code") == -1121:
                        raise SpotDataError(f"SPOT_UNAVAILABLE: Invalid symbol (-1121) for {params or {}}")
                    raise SpotDataError(f"Binance Spot rejected {path}: {payload}")
                response.raise_for_status()
                return response.json()
            except SpotDataError:
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(0.25 * (2 ** attempt))
        raise SpotDataError(f"Binance Spot request failed for {path}: {last_error}") from last_error

    def _load_exchange_info(self) -> list[dict[str, Any]]:
        if self._exchange_info is not None:
            return self._exchange_info
        payload = self._get_json("/api/v3/exchangeInfo", {
            "permissions": "SPOT",
            "symbolStatus": "TRADING",
            "showPermissionSets": "false",
        })
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise SpotDataError("Unexpected Binance Spot exchangeInfo response")
        entries: list[dict[str, Any]] = []
        by_base_quote: dict[tuple[str, str], list[str]] = {}
        for item in payload["symbols"]:
            if not isinstance(item, dict) or item.get("status") != "TRADING":
                continue
            permissions = item.get("permissions")
            if isinstance(permissions, list) and "SPOT" not in permissions:
                continue
            if permissions is None:
                # The request filters SPOT on the server. If an older response
                # still includes permissionSets, use it as a defensive check.
                permission_sets = item.get("permissionSets")
                if permission_sets and not any(
                    "SPOT" in group for group in permission_sets if isinstance(group, list)
                ):
                    continue
            symbol = item.get("symbol")
            base = item.get("baseAsset")
            quote = item.get("quoteAsset")
            if not all(isinstance(value, str) and value for value in (symbol, base, quote)):
                continue
            symbol = symbol.upper()
            base, quote = base.upper(), quote.upper()
            entries.append({"symbol": symbol, "baseAsset": base, "quoteAsset": quote})
            by_base_quote.setdefault((base, quote), []).append(symbol)
        # An empty (but well-formed) TRADING/SPOT universe means no Spot
        # mapping is available; it is not a Futures scan failure.
        self._exchange_info = entries
        self._symbols_by_base_quote = by_base_quote
        return entries

    @staticmethod
    def _futures_base_asset(futures_symbol: str) -> str:
        value = str(futures_symbol or "").upper().strip()
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if value.endswith(quote):
                return value[:-len(quote)]
        return value

    def resolve_symbol(self, futures_symbol: str, *, quote_asset: str = "USDT") -> SpotSymbolResolution | None:
        """Map a tradable Futures contract to one active Spot pair.

        A known multiplier prefix is resolved to its unscaled underlying first;
        exact base-asset matching is the fallback. Either candidate is accepted
        only when exchangeInfo confirms one active Spot symbol for base/quote.
        """
        entries = self._load_exchange_info()
        assert self._symbols_by_base_quote is not None
        futures = str(futures_symbol or "").upper().strip()
        base = self._futures_base_asset(futures)
        candidates: list[tuple[str, float]] = []
        for prefix, multiplier in _SPOT_MULTIPLIER_PREFIXES:
            if base.startswith(prefix) and len(base) > len(prefix):
                # Prefer the unscaled underlying Spot asset for multiplier
                # Futures contracts (e.g. 1000PEPE -> PEPE).
                candidates.append((base[len(prefix):], multiplier))
                break
        candidates.append((base, 1.0))
        for candidate_base, multiplier in candidates:
            matches = self._symbols_by_base_quote.get((candidate_base, quote_asset.upper()), [])
            if len(matches) == 1:
                return SpotSymbolResolution(
                    futures_symbol=futures,
                    spot_symbol=matches[0],
                    futures_base_asset=base,
                    spot_base_asset=candidate_base,
                    quote_asset=quote_asset.upper(),
                    base_multiplier=multiplier,
                )
            if len(matches) > 1:
                # Ambiguous exchange metadata is not resolved by guessing.
                return None
        return None

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "5m",
        *,
        limit: int = 300,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Candle]:
        durations = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
            "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
            "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
        }
        if interval not in durations:
            raise ValueError(f"Unsupported Binance Spot candle interval: {interval}")
        params: dict[str, Any] = {"symbol": str(symbol).upper(), "interval": interval, "limit": min(max(int(limit), 1), 1000)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        rows = self._get_json("/api/v3/klines", params)
        if not isinstance(rows, list):
            raise SpotDataError(f"Unexpected Binance Spot klines response for {symbol}/{interval}")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        candles: list[Candle] = []
        duration_ms = durations[interval]
        for raw in rows:
            if not isinstance(raw, (list, tuple)) or len(raw) < 9:
                continue
            try:
                candle = Candle(
                    open_time_ms=int(raw[0]), close_time_ms=int(raw[6]),
                    open=float(raw[1]), high=float(raw[2]), low=float(raw[3]), close=float(raw[4]),
                    volume=float(raw[5]), quote_volume=float(raw[7]), trades=int(raw[8]),
                    taker_buy_base_volume=float(raw[9]) if len(raw) > 9 else None,
                    taker_buy_quote_volume=float(raw[10]) if len(raw) > 10 else None,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            values = [candle.open, candle.high, candle.low, candle.close, candle.volume, candle.quote_volume]
            values.extend(value for value in (candle.taker_buy_base_volume, candle.taker_buy_quote_volume) if value is not None)
            sane = (
                all(math.isfinite(value) for value in values)
                and candle.open > 0 and candle.high >= max(candle.open, candle.close, candle.low)
                and candle.low <= min(candle.open, candle.close, candle.high)
                and candle.close > 0 and candle.volume >= 0 and candle.quote_volume >= 0 and candle.trades >= 0
                and (candle.taker_buy_base_volume is None or 0 <= candle.taker_buy_base_volume <= candle.volume)
                and (candle.taker_buy_quote_volume is None or 0 <= candle.taker_buy_quote_volume <= candle.quote_volume)
                and candle.close_time_ms == candle.open_time_ms + duration_ms - 1
            )
            if sane and candle.close_time_ms < now_ms:
                candles.append(candle)
        unique = {item.open_time_ms: item for item in candles}
        result = [unique[key] for key in sorted(unique)]
        last_gap = next((idx for idx in range(len(result) - 1, 0, -1)
                         if result[idx].open_time_ms - result[idx - 1].open_time_ms != duration_ms), None)
        if last_gap is not None:
            result = result[last_gap:]
        return result
