from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .models import Candle

log = logging.getLogger(__name__)


class BinanceDataError(RuntimeError):
    pass


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
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Market-Screener/1.0"})
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._symbols: set[str] | None = None

    def _pace(self) -> None:
        with self._lock:
            wait = self.min_interval_sec - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            self._pace()
            try:
                response = self.session.get(f"{self.base_url}{path}", params=params or {}, timeout=self.timeout_sec)
                if response.status_code in (418, 429):
                    retry = response.headers.get("Retry-After")
                    delay = float(retry) if retry and retry.replace(".", "", 1).isdigit() else 2 ** attempt
                    if attempt == 2:
                        response.raise_for_status()
                    time.sleep(min(max(delay, 1.0), 30.0))
                    continue
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
        value = str(coinalyze_symbol or "").upper().strip()
        # Examples: BTCUSDT_PERP.A, BTCUSDT_PERP, BTCUSDT.A
        value = value.split(".", 1)[0]
        value = re.sub(r"[_-](PERP|SWAP)$", "", value)
        value = value.replace("/", "").replace("-", "").replace("_", "")
        return value

    def resolve_symbol(self, coinalyze_symbol: str) -> str | None:
        candidate = self._base_symbol(coinalyze_symbol)
        symbols = self._load_symbols()
        if candidate in symbols:
            return candidate
        if not candidate.endswith("USDT"):
            candidate += "USDT"
        return candidate if candidate in symbols else None

    def fetch_candles(self, coinalyze_symbol: str, interval: str, limit: int = 300) -> list[Candle]:
        resolved = self.resolve_symbol(coinalyze_symbol)
        if not resolved:
            raise BinanceDataError(f"No active Binance USDT perpetual for Coinalyze symbol {coinalyze_symbol}")
        interval_ms = {
            "5m": 5 * 60_000,
            "15m": 15 * 60_000,
            "30m": 30 * 60_000,
            "1h": 60 * 60_000,
            "2h": 2 * 60 * 60_000,
            "4h": 4 * 60 * 60_000,
            "6h": 6 * 60 * 60_000,
            "12h": 12 * 60 * 60_000,
            "1d": 24 * 60 * 60_000,
        }
        if interval not in interval_ms:
            raise ValueError(f"Unsupported Binance candle interval: {interval}")
        duration = interval_ms[interval]
        rows = self._get_json(
            "/fapi/v1/klines",
            {"symbol": resolved, "interval": interval, "limit": min(max(int(limit), 100), 1500)},
        )
        if not isinstance(rows, list):
            raise BinanceDataError(f"Unexpected Binance klines response for {resolved}/{interval}")

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        candles: list[Candle] = []
        malformed = 0
        for raw in rows:
            if not isinstance(raw, (list, tuple)) or len(raw) < 9:
                malformed += 1
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
            except (TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            numeric_values = (candle.open, candle.high, candle.low, candle.close, candle.volume, candle.quote_volume)
            valid = (
                all(math.isfinite(value) for value in numeric_values)
                and candle.open > 0
                and candle.high > 0
                and candle.low > 0
                and candle.close > 0
                and candle.volume >= 0
                and candle.quote_volume >= 0
                and candle.trades >= 0
                and candle.high >= max(candle.open, candle.close, candle.low)
                and candle.low <= min(candle.open, candle.close, candle.high)
                and candle.close_time_ms == candle.open_time_ms + duration - 1
            )
            if not valid:
                malformed += 1
                continue
            # Binance's final array element may be a still-forming candle.
            if candle.close_time_ms < now_ms:
                candles.append(candle)

        candles.sort(key=lambda candle: candle.open_time_ms)
        # De-duplicate by open time and keep deterministic ordering.
        unique = {candle.open_time_ms: candle for candle in candles}
        result = [unique[key] for key in sorted(unique)]
        if len(result) < 20:
            raise BinanceDataError(
                f"Only {len(result)} valid closed {interval} candles for {resolved}; need at least 20"
            )

        # The structure indicators assume adjacent bars are consecutive periods.
        # Do not calculate pivots/ATR over a silently compressed series with gaps.
        gaps = [
            (previous.open_time_ms, current.open_time_ms)
            for previous, current in zip(result, result[1:])
            if current.open_time_ms - previous.open_time_ms != duration
        ]
        if gaps:
            previous_open, current_open = gaps[0]
            gap_periods = (current_open - previous_open) / duration
            raise BinanceDataError(
                f"Non-consecutive {interval} candles for {resolved}: gap spans {gap_periods:g} intervals; "
                "1H/4H structure and ATR are not evaluated on discontinuous data"
            )

        latest_age_ms = now_ms - result[-1].close_time_ms
        if latest_age_ms > duration + 120_000:
            raise BinanceDataError(
                f"Stale {interval} candles for {resolved}: latest closed candle is "
                f"{latest_age_ms / 60_000:.1f} minutes old"
            )
        if malformed:
            log.warning("Ignored %d malformed Binance %s candles for %s", malformed, interval, resolved)
        return result

    def fetch_orderbook_diagnostics(
        self, coinalyze_symbol: str, *, limit: int = 100, notional_usd: float = 0.0
    ) -> dict[str, Any]:
        """Read-only top-of-book/depth diagnostics; this method never creates orders."""
        resolved = self.resolve_symbol(coinalyze_symbol)
        if not resolved:
            raise BinanceDataError(f"No active Binance USDT perpetual for Coinalyze symbol {coinalyze_symbol}")
        safe_limit = min(max(int(limit), 5), 1000)
        try:
            requested_notional = float(notional_usd)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("notional_usd must be a finite non-negative number") from exc
        if not math.isfinite(requested_notional) or requested_notional < 0:
            raise ValueError("notional_usd must be a finite non-negative number")
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
                slippage = (reference / average_price - 1.0) * 10_000.0
            return {
                "filled_usd": filled_usd,
                "average_price": average_price,
                "slippage_bps": slippage,
                "unfilled_usd": max(0.0, remaining),
            }

        buy = estimate_vwap(asks, requested_notional)
        sell = estimate_vwap(bids, requested_notional)
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
            "notional_usd": requested_notional if requested_notional > 0 else None,
            "market_buy_estimate": buy,
            "market_sell_estimate": sell,
            "estimated_only": True,
        }

