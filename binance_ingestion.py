"""
Production-ready Binance Futures Ingestion Layer.
Implements:
1. Low-latency WebSocket client with automatic reconnection and exponential backoff.
2. Robust Finite State Machine (FSM) for L2 Order Book synchronization with sequence integrity checks (pu == u_prev).
3. Typed msgspec serialization with bounded Python object overhead; numeric hot paths may allocate NumPy buffers.
4. Normalization of trades, order book depth, funding rates (8h basis), and mark price basis spread.
5. High-concurrency REST batch collector for universe-wide 5-minute cron executions.
"""
from __future__ import annotations

import asyncio
import collections
import enum
import logging
import math
import os
import random
import time
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

import aiohttp
import msgspec

from contracts import (
    NormalizedFunding,
    NormalizedTrade,
    OrderBookSnapshot,
)

# Optional uvloop integration for maximum event-loop throughput
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = logging.getLogger("binance_ingestion")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BinanceRestrictedLocationError(RuntimeError):
    """Binance rejected this request because the client egress is geo-restricted."""


class OrderBookFSMState(enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    BUFFERING = "BUFFERING"
    REST_SNAPSHOT = "REST_SNAPSHOT"
    APPLYING = "APPLYING"
    IN_SYNC = "IN_SYNC"


class BinanceOrderBookFSM:
    """
    State Machine managing Binance Futures Depth (L2) integrity.
    Enforces official Binance sequence constraints:
    1. Buffer depth events from WS stream.
    2. Fetch REST snapshot (/fapi/v1/depth?limit=1000).
    3. Drop events where event.u < snapshot.lastUpdateId.
    4. First event must satisfy: event.U <= snapshot.lastUpdateId + 1 <= event.u.
    5. Each subsequent event must strictly satisfy: event.pu == previous_event.u.
    6. Any sequence gap immediately forces book reset and snapshot refetch.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol.upper()
        self.state: OrderBookFSMState = OrderBookFSMState.DISCONNECTED
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        self.buffer: Deque[Dict[str, Any]] = collections.deque()
        self.last_update_id: int = 0
        self.prev_u: int = 0
        self.last_sync_timestamp_ms: int = 0
        self.resync_count: int = 0
        self._buffer_event = asyncio.Event()

    def reset(self) -> None:
        """Reset internal book state upon sequence disconnect or gap."""
        self.state = OrderBookFSMState.DISCONNECTED
        self.bids.clear()
        self.asks.clear()
        self.buffer.clear()
        self.last_update_id = 0
        self.prev_u = 0
        self.resync_count += 1
        self._buffer_event.clear()
        logger.warning(f"[{self.symbol}] OrderBook FSM reset triggered. Resync count: {self.resync_count}")

    def on_ws_connected(self) -> None:
        """Called when WebSocket connection is opened."""
        self.state = OrderBookFSMState.BUFFERING
        self.buffer.clear()
        self._buffer_event.clear()
        logger.info(f"[{self.symbol}] OrderBook FSM transitioned to BUFFERING")

    def handle_depth_event(self, event: Dict[str, Any]) -> None:
        """Receive depthUpdate event from WebSocket stream."""
        if self.state == OrderBookFSMState.DISCONNECTED:
            return

        if self.state in (OrderBookFSMState.BUFFERING, OrderBookFSMState.REST_SNAPSHOT, OrderBookFSMState.APPLYING):
            self.buffer.append(event)
            self._buffer_event.set()
            return

        if self.state == OrderBookFSMState.IN_SYNC:
            pu = event.get("pu")
            u = event.get("u")
            if pu is None or u is None:
                logger.error(f"[{self.symbol}] Malformed depth event without pu/u: {event}")
                self.reset()
                return

            # Strict sequence continuity check for Binance Futures
            if pu != self.prev_u:
                logger.error(
                    f"[{self.symbol}] Sequence ID gap detected! Expected pu={self.prev_u}, got pu={pu}. Forcing resync."
                )
                self.reset()
                return

            self._apply_diff(event.get("b", []), event.get("a", []))
            self.prev_u = u
            self.last_sync_timestamp_ms = event.get("E", int(time.time() * 1000))

    async def wait_for_buffered_event(self) -> None:
        """Wait until at least one depth event is buffered."""
        await self._buffer_event.wait()
        self._buffer_event.clear()

    def apply_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Apply REST depth snapshot and process buffered events.
        Returns True if successfully synchronized, False if resync needed.
        """
        self.state = OrderBookFSMState.APPLYING
        last_update_id = snapshot.get("lastUpdateId", 0)
        if last_update_id <= 0:
            logger.error(f"[{self.symbol}] Invalid snapshot lastUpdateId={last_update_id}")
            self.reset()
            return False

        self.last_update_id = last_update_id
        self.bids.clear()
        self.asks.clear()

        # Seed initial bids & asks
        for p_str, q_str in snapshot.get("bids", []):
            qty = float(q_str)
            if qty > 0.0:
                self.bids[float(p_str)] = qty

        for p_str, q_str in snapshot.get("asks", []):
            qty = float(q_str)
            if qty > 0.0:
                self.asks[float(p_str)] = qty

        # Discard events where u < lastUpdateId
        while self.buffer and self.buffer[0].get("u", 0) < self.last_update_id:
            self.buffer.popleft()

        if not self.buffer:
            # Buffer was drained or snapshot was ahead of buffer
            logger.warning(f"[{self.symbol}] Buffer exhausted after snapshot. Needs re-buffering.")
            self._buffer_event.clear()
            self.state = OrderBookFSMState.BUFFERING
            return False

        # Binance local-book contract: the first buffered event must contain
        # the update immediately after the REST snapshot: U <= lastUpdateId + 1 <= u.
        first_event = self.buffer[0]
        U = first_event.get("U", 0)
        u = first_event.get("u", 0)

        if not (U <= self.last_update_id + 1 <= u):
            logger.warning(
                f"[{self.symbol}] First event sequence mismatch: U={U} <= lastUpdateId+1={self.last_update_id + 1} <= u={u} violated. Re-buffering."
            )
            # A snapshot mismatch does not mean the WS connection is dead.
            # Keep the stream in BUFFERING so fresh events can accumulate for
            # the next snapshot instead of requiring a physical reconnect.
            self.bids.clear()
            self.asks.clear()
            self.buffer.clear()
            self.last_update_id = 0
            self.prev_u = 0
            self.state = OrderBookFSMState.BUFFERING
            self._buffer_event.clear()
            self.resync_count += 1
            return False

        # Apply first event
        self.buffer.popleft()
        self._apply_diff(first_event.get("b", []), first_event.get("a", []))
        self.prev_u = u

        # Apply remaining buffered events verifying continuity
        while self.buffer:
            event = self.buffer.popleft()
            e_pu = event.get("pu", 0)
            e_u = event.get("u", 0)
            if e_pu != self.prev_u:
                logger.error(
                    f"[{self.symbol}] Gap in buffered events: expected pu={self.prev_u}, got {e_pu}. Resetting."
                )
                self.reset()
                return False
            self._apply_diff(event.get("b", []), event.get("a", []))
            self.prev_u = e_u

        self.state = OrderBookFSMState.IN_SYNC
        self.last_sync_timestamp_ms = int(time.time() * 1000)
        logger.info(f"[{self.symbol}] OrderBook successfully IN_SYNC. Bids: {len(self.bids)}, Asks: {len(self.asks)}")
        return True

    def _apply_diff(self, bids: List[List[str]], asks: List[List[str]]) -> None:
        """Update local book with price-level diffs."""
        for p_str, q_str in bids:
            price = float(p_str)
            qty = float(q_str)
            if not math.isfinite(price) or price <= 0.0 or not math.isfinite(qty) or qty < 0.0:
                raise ValueError("Invalid bid depth level")
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for p_str, q_str in asks:
            price = float(p_str)
            qty = float(q_str)
            if not math.isfinite(price) or price <= 0.0 or not math.isfinite(qty) or qty < 0.0:
                raise ValueError("Invalid ask depth level")
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_snapshot(self, depth: int = 10) -> Optional[OrderBookSnapshot]:
        """
        Extract a normalized OrderBookSnapshot with OBI and spread calculations.
        Returns None if book is not IN_SYNC or either side is empty.
        """
        if self.state != OrderBookFSMState.IN_SYNC or not self.bids or not self.asks:
            return None

        # Sorted bids (descending) and asks (ascending)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

        if not sorted_bids or not sorted_asks:
            return None

        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]

        if best_bid >= best_ask:
            # Crossed order book anomaly
            logger.warning(f"[{self.symbol}] Crossed book detected: best_bid={best_bid} >= best_ask={best_ask}")
            self.reset()
            return None

        mid_price = (best_bid + best_ask) * 0.5
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000.0 if mid_price > 0.0 else 0.0

        # Calculate Order Book Imbalance (OBI) for top 5 levels
        bid_vol_5 = sum(qty for _, qty in sorted_bids[:5])
        ask_vol_5 = sum(qty for _, qty in sorted_asks[:5])
        tot_vol_5 = bid_vol_5 + ask_vol_5
        obi_5 = (bid_vol_5 - ask_vol_5) / tot_vol_5 if tot_vol_5 > 0.0 else 0.0

        # Calculate OBI for top 10 levels
        bid_vol_10 = sum(qty for _, qty in sorted_bids[:10])
        ask_vol_10 = sum(qty for _, qty in sorted_asks[:10])
        tot_vol_10 = bid_vol_10 + ask_vol_10
        obi_10 = (bid_vol_10 - ask_vol_10) / tot_vol_10 if tot_vol_10 > 0.0 else 0.0

        return OrderBookSnapshot(
            symbol=self.symbol,
            last_update_id=self.prev_u,
            timestamp_ms=self.last_sync_timestamp_ms or int(time.time() * 1000),
            bids=tuple(sorted_bids),
            asks=tuple(sorted_asks),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            obi_depth5=obi_5,
            obi_depth10=obi_10,
        )


class WeightedRateLimiter:
    """Sliding-window REQUEST_WEIGHT limiter with explicit 429/418 cooldowns."""

    def __init__(self, max_weight: int, window_seconds: float = 60.0) -> None:
        if max_weight <= 0 or window_seconds <= 0.0:
            raise ValueError("Rate limiter configuration must be positive")
        self.max_weight = max_weight
        self.window_seconds = window_seconds
        self._events: Deque[Tuple[float, int]] = collections.deque()
        self._used_weight = 0
        self._lock = asyncio.Lock()
        self._cooldown_until = 0.0

    async def acquire(self, weight: int) -> None:
        if weight <= 0:
            return
        if weight > self.max_weight:
            raise ValueError(f"request weight {weight} exceeds limiter budget {self.max_weight}")
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= self.window_seconds:
                    _, old_weight = self._events.popleft()
                    self._used_weight -= old_weight

                wait_for_cooldown = max(0.0, self._cooldown_until - now)
                if wait_for_cooldown > 0.0:
                    sleep_for = wait_for_cooldown
                elif self._used_weight + weight <= self.max_weight:
                    self._events.append((now, weight))
                    self._used_weight += weight
                    return
                else:
                    oldest_at, _ = self._events[0]
                    sleep_for = max(0.001, self.window_seconds - (now - oldest_at))
            await asyncio.sleep(sleep_for)

    def cooldown(self, seconds: float) -> None:
        if seconds > 0.0:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)



class BinanceFuturesIngestion:
    """
    Asynchronous Binance Futures Ingestion Layer.
    Supports both:
    1. Real-time persistent WebSocket streaming (L2 depth FSM, aggTrades, markPrice).
    2. High-speed REST batch polling for 5-minute cron executions over 100+ symbols.
    """

    WS_BASE_URL = "wss://fstream.binance.com/stream"
    REST_BASE_URL = "https://fapi.binance.com"

    def __init__(
        self,
        symbols: List[str],
        on_trade: Optional[Callable[[NormalizedTrade], Coroutine[Any, Any, None]]] = None,
        on_book: Optional[Callable[[OrderBookSnapshot], Coroutine[Any, Any, None]]] = None,
        on_funding: Optional[Callable[[NormalizedFunding], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        self.symbols: List[str] = [s.upper() for s in symbols]
        self.on_trade = on_trade
        self.on_book = on_book
        self.on_funding = on_funding

        self.books: Dict[str, BinanceOrderBookFSM] = {
            sym: BinanceOrderBookFSM(sym) for sym in self.symbols
        }

        self._session: Optional[aiohttp.ClientSession] = None
        self._running: bool = False
        self._ws_task: Optional[asyncio.Task[None]] = None
        self._book_sync_tasks: Dict[str, asyncio.Task[None]] = {}
        self._funding_intervals_hours: Dict[str, float] = {}
        self._funding_info_loaded = False
        self._msg_decoder = msgspec.json.Decoder()
        # Production cron route: 40 ticker + 10 premiumIndex + 100*20 aggTrades
        # + 100*2 depth + 100*1 kline + 100*1 OI history = 2450 worst-case if
        # every endpoint used the historical path. The actual scanner does not
        # request current OI separately. Keep a 5% safety reserve by default and
        # let the limiter pace beyond one window rather than violating Binance limits.
        self._rate_limiter = WeightedRateLimiter(
            max_weight=int(os.getenv("BINANCE_REQUEST_WEIGHT_BUDGET", "2280"))
        )
        self._backoff_429_seconds = float(os.getenv("BINANCE_429_BACKOFF_SEC", "1.0"))
        self._cooldown_418_seconds = float(os.getenv("BINANCE_418_COOLDOWN_SEC", "60.0"))
        self._max_request_retries = int(os.getenv("BINANCE_MAX_REQUEST_RETRIES", "3"))
        self._retry_backoff_cap_seconds = float(os.getenv("BINANCE_RETRY_BACKOFF_CAP_SEC", "8.0"))
        if not math.isfinite(self._backoff_429_seconds) or self._backoff_429_seconds <= 0.0:
            raise ValueError("BINANCE_429_BACKOFF_SEC must be finite and positive")
        if not math.isfinite(self._cooldown_418_seconds) or self._cooldown_418_seconds <= 0.0:
            raise ValueError("BINANCE_418_COOLDOWN_SEC must be finite and positive")
        if self._max_request_retries < 0 or self._max_request_retries > 10:
            raise ValueError("BINANCE_MAX_REQUEST_RETRIES must be between 0 and 10")
        if not math.isfinite(self._retry_backoff_cap_seconds) or self._retry_backoff_cap_seconds <= 0.0:
            raise ValueError("BINANCE_RETRY_BACKOFF_CAP_SEC must be finite and positive")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            headers = {
                "User-Agent": "QuantEngine/1.0",
                "Accept-Encoding": "gzip",
            }
            api_key = os.getenv("BINANCE_API_KEY")
            if api_key:
                headers["X-MBX-APIKEY"] = api_key
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, str]] = None,
        weight: int,
        symbol: Optional[str] = None,
    ) -> Optional[Any]:
        if weight < 0:
            raise ValueError("request weight must be non-negative")
        if method.upper() not in {"GET", "HEAD"}:
            raise ValueError("_request_json is only safe for idempotent GET/HEAD requests")

        for attempt in range(self._max_request_retries + 1):
            await self._rate_limiter.acquire(weight)
            session = await self._get_session()
            try:
                async with session.request(method, url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)

                    if resp.status == 451:
                        body = await resp.text()
                        logger.critical(
                            "binance_restricted_location status=451 symbol=%s body=%s",
                            symbol, body[:500],
                        )
                        raise BinanceRestrictedLocationError(
                            "Binance Futures rejected the request with HTTP 451 "
                            "(restricted location)"
                        )

                    retry_after_raw = resp.headers.get("Retry-After")
                    try:
                        retry_after = float(retry_after_raw) if retry_after_raw else 0.0
                    except ValueError:
                        retry_after = 0.0

                    retryable = resp.status == 429 or 500 <= resp.status <= 599
                    if resp.status == 418:
                        cooldown = max(self._cooldown_418_seconds, retry_after)
                        self._rate_limiter.cooldown(cooldown)
                        logger.critical(
                            "binance_ip_banned status=418 symbol=%s retry_after=%s cooldown=%.2fs",
                            symbol, retry_after_raw, cooldown,
                        )
                        return None
                    if retryable:
                        if resp.status == 429:
                            cooldown = max(self._backoff_429_seconds, retry_after)
                            self._rate_limiter.cooldown(cooldown)
                            logger.warning(
                                "binance_rate_limited status=429 symbol=%s retry_after=%s cooldown=%.2fs attempt=%d/%d",
                                symbol, retry_after_raw, cooldown, attempt + 1, self._max_request_retries + 1,
                            )
                        else:
                            body = await resp.text()
                            logger.warning(
                                "binance_retryable_http_error status=%s symbol=%s body=%s attempt=%d/%d",
                                resp.status, symbol, body[:300], attempt + 1, self._max_request_retries + 1,
                            )
                        if attempt < self._max_request_retries:
                            delay = min(
                                self._retry_backoff_cap_seconds,
                                max(self._backoff_429_seconds, 2.0 ** attempt),
                            )
                            delay += random.uniform(0.0, 0.25 * delay)
                            await asyncio.sleep(delay)
                            continue
                    else:
                        body = await resp.text()
                        logger.error(
                            "binance_http_error status=%s symbol=%s body=%s",
                            resp.status, symbol, body[:500],
                        )
                    return None
            except BinanceRestrictedLocationError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "binance_request_failed symbol=%s url=%s error=%s attempt=%d/%d",
                    symbol, url, exc, attempt + 1, self._max_request_retries + 1,
                )
                if attempt < self._max_request_retries:
                    delay = min(self._retry_backoff_cap_seconds, max(self._backoff_429_seconds, 2.0 ** attempt))
                    delay += random.uniform(0.0, 0.25 * delay)
                    await asyncio.sleep(delay)
                    continue
                return None
            except (TypeError, ValueError) as exc:
                logger.error("binance_response_invalid symbol=%s url=%s error=%s", symbol, url, exc)
                return None
        return None

    async def fetch_l2_snapshot(self, symbol: str, limit: int = 1000) -> Optional[Dict[str, Any]]:
        """Fetch REST depth snapshot for order book FSM synchronization."""
        if limit not in (5, 10, 20, 50, 100, 500, 1000):
            raise ValueError("unsupported Binance depth limit")
        weight = 20 if limit == 1000 else (10 if limit == 500 else (5 if limit == 100 else 2))
        url = f"{self.REST_BASE_URL}/fapi/v1/depth"
        data = await self._request_json(
            "GET", url, params={"symbol": symbol, "limit": str(limit)}, weight=weight, symbol=symbol
        )
        if not isinstance(data, dict):
            return None
        return data

    def _build_stream_url(self) -> str:
        """Construct multi-stream WebSocket URL for all requested contracts."""
        stream_names: List[str] = []
        for sym in self.symbols:
            s_lower = sym.lower()
            stream_names.append(f"{s_lower}@depth@100ms")
            stream_names.append(f"{s_lower}@aggTrade")
            stream_names.append(f"{s_lower}@markPrice@1s")
        return f"{self.WS_BASE_URL}?streams={'/'.join(stream_names)}"

    async def start_streaming(self) -> None:
        """Start long-running WebSocket streaming loop with automatic reconnect."""
        self._running = True
        backoff_sec = 1.0
        max_backoff_sec = 30.0

        while self._running:
            try:
                session = await self._get_session()
                stream_url = self._build_stream_url()
                logger.info(f"Connecting to Binance Futures WS streams for {len(self.symbols)} symbols...")

                async with session.ws_connect(
                    stream_url,
                    autoping=True,
                    heartbeat=20.0,
                    receive_timeout=30.0,
                    max_msg_size=16 * 1024 * 1024,
                ) as ws:
                    logger.info("WebSocket connected successfully!")
                    backoff_sec = 1.0  # Reset backoff on successful connection

                    funding_info = await self.fetch_universe_funding_info()
                    self._funding_info_loaded = funding_info is not None
                    if funding_info is not None:
                        self._funding_intervals_hours = funding_info
                    else:
                        logger.error("funding_info_unavailable streaming funding normalization is disabled until metadata loads")

                    # Set books into BUFFERING state and schedule snapshots.
                    for sym in self.books:
                        self._schedule_book_sync(sym)

                    async for msg in ws:
                        if not self._running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._dispatch_ws_message(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning(f"WebSocket closed or errored: {msg.type}")
                            break

            except asyncio.CancelledError:
                logger.info("Streaming task cancelled.")
                break
            except Exception as exc:
                logger.error(f"WebSocket loop error: {exc}. Reconnecting in {backoff_sec:.1f}s...")
                await asyncio.sleep(backoff_sec + random.uniform(0.1, 0.5))
                backoff_sec = min(backoff_sec * 2.0, max_backoff_sec)

        logger.info("Streaming stopped.")

    def _schedule_book_sync(self, symbol: str) -> None:
        """Schedule one snapshot resync for a symbol after connect or sequence gap."""
        book = self.books.get(symbol)
        if book is None:
            return
        task = self._book_sync_tasks.get(symbol)
        if task is not None and not task.done():
            return
        if book.state == OrderBookFSMState.DISCONNECTED:
            book.on_ws_connected()
        self._book_sync_tasks[symbol] = asyncio.create_task(self._sync_symbol_book(symbol))

    async def _sync_symbol_book(self, symbol: str) -> None:
        """Fetch snapshots until a buffered event range can be synchronized."""
        book = self.books.get(symbol)
        if not book:
            return
        current_task = asyncio.current_task()
        try:
            while self._running:
                if book.state == OrderBookFSMState.DISCONNECTED:
                    book.on_ws_connected()
                if book.state == OrderBookFSMState.IN_SYNC:
                    return
                if book.state == OrderBookFSMState.BUFFERING and not book.buffer:
                    await book.wait_for_buffered_event()
                    if not self._running:
                        return

                book.state = OrderBookFSMState.REST_SNAPSHOT
                await asyncio.sleep(0.3)
                snapshot = await self.fetch_l2_snapshot(symbol, limit=1000)
                if not snapshot:
                    book.state = OrderBookFSMState.BUFFERING
                    await asyncio.sleep(self._backoff_429_seconds)
                    continue

                success = book.apply_snapshot(snapshot)
                if success:
                    if self.on_book:
                        snap = book.get_snapshot(depth=10)
                        if snap:
                            await self.on_book(snap)
                    return
                if book.state == OrderBookFSMState.DISCONNECTED:
                    book.on_ws_connected()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("book_sync_failed symbol=%s error=%s", symbol, exc)
            if book.state != OrderBookFSMState.DISCONNECTED:
                book.reset()
        finally:
            if self._book_sync_tasks.get(symbol) is current_task:
                self._book_sync_tasks.pop(symbol, None)

    async def _dispatch_ws_message(self, raw_text: str) -> None:
        """Parse and dispatch multi-stream combined WebSocket messages."""
        try:
            payload = self._msg_decoder.decode(raw_text.encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed to decode message: {exc}")
            return

        stream_name = payload.get("stream", "")
        data = payload.get("data", {})
        event_type = data.get("e")

        if event_type == "depthUpdate":
            symbol = data.get("s", "")
            book = self.books.get(symbol)
            if book:
                try:
                    book.handle_depth_event(data)
                except (TypeError, ValueError, KeyError) as exc:
                    logger.error("malformed_depth_event symbol=%s error=%s", symbol, exc)
                    book.reset()
                if book.state == OrderBookFSMState.DISCONNECTED:
                    self._schedule_book_sync(symbol)
                if self.on_book and book.state == OrderBookFSMState.IN_SYNC:
                    snap = book.get_snapshot(depth=10)
                    if snap:
                        await self.on_book(snap)

        elif event_type == "aggTrade":
            symbol = data.get("s")
            try:
                is_maker = bool(data["m"])
                price = float(data["p"])
                qty = float(data["q"])
                timestamp_ms = int(data["T"])
                trade_id = int(data["a"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("malformed_agg_trade error=%s payload=%s", exc, data)
                return
            if (not isinstance(symbol, str) or not symbol
                    or not math.isfinite(price) or price <= 0.0
                    or not math.isfinite(qty) or qty <= 0.0
                    or timestamp_ms <= 0 or trade_id <= 0):
                logger.error("invalid_agg_trade payload=%s", data)
                return
            side = "SELL" if is_maker else "BUY"
            trade = NormalizedTrade(
                symbol=symbol,
                price=price,
                quantity=qty,
                quote_quantity=price * qty,
                side=side,
                timestamp_ms=timestamp_ms,
                is_buyer_maker=is_maker,
                trade_id=trade_id,
            )
            if self.on_trade:
                await self.on_trade(trade)

        elif event_type == "markPriceUpdate":
            symbol = data.get("s")
            try:
                raw_rate = float(data["r"])
                mark_price = float(data["p"])
                index_price = float(data["i"])
                next_time = int(data["T"])
                event_time = int(data["E"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("malformed_mark_price error=%s payload=%s", exc, data)
                return
            if (not isinstance(symbol, str) or not symbol
                    or not math.isfinite(raw_rate)
                    or not math.isfinite(mark_price) or mark_price <= 0.0
                    or not math.isfinite(index_price) or index_price <= 0.0
                    or next_time <= 0 or event_time <= 0
                    or not self._funding_info_loaded):
                logger.error("invalid_or_unready_mark_price symbol=%s payload=%s", symbol, data)
                return

            interval_h = self._funding_intervals_hours.get(symbol, 8.0)
            if not math.isfinite(interval_h) or interval_h <= 0.0 or 1.0 + raw_rate <= 0.0:
                logger.error("invalid_funding_metadata symbol=%s interval_h=%s raw_rate=%s", symbol, interval_h, raw_rate)
                return
            norm_8h = (1.0 + raw_rate) ** (8.0 / interval_h) - 1.0
            annualized = norm_8h * 3.0 * 365.0 * 100.0
            basis_spread = mark_price - index_price
            basis_bps = basis_spread / index_price * 10000.0

            funding = NormalizedFunding(
                symbol=symbol,
                raw_rate=raw_rate,
                interval_hours=interval_h,
                normalized_8h_rate=norm_8h,
                annualized_rate=annualized,
                mark_price=mark_price,
                index_price=index_price,
                basis_spread=basis_spread,
                basis_spread_bps=basis_bps,
                next_funding_time_ms=next_time,
                timestamp_ms=event_time,
            )
            if self.on_funding:
                await self.on_funding(funding)

    async def stop(self) -> None:
        """Gracefully stop ingestion and close sessions."""
        self._running = False
        tasks = list(self._book_sync_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._book_sync_tasks.clear()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Binance ingestion cleanly stopped.")

    # -------------------------------------------------------------------------
    # BATCH REST MODE (Optimized for 5-Minute Cron Executions without Docker)
    # -------------------------------------------------------------------------

    async def fetch_universe_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch 24h ticker data for all USDT-M symbols in one request."""
        url = f"{self.REST_BASE_URL}/fapi/v1/ticker/24hr"
        data = await self._request_json("GET", url, weight=40)
        if not isinstance(data, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            try:
                quote_volume = float(item["quoteVolume"])
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(symbol, str) or not symbol.endswith("USDT"):
                continue
            if not math.isfinite(quote_volume) or quote_volume < 0.0:
                continue
            out[symbol] = item
        return out

    async def fetch_universe_premium_index(self) -> Dict[str, Dict[str, Any]]:
        """Fetch premium/funding data for all USDT-M symbols in one request."""
        url = f"{self.REST_BASE_URL}/fapi/v1/premiumIndex"
        data = await self._request_json("GET", url, weight=10)
        if not isinstance(data, list):
            return {}
        return {
            item["symbol"]: item
            for item in data
            if isinstance(item, dict)
            and isinstance(item.get("symbol"), str)
            and item["symbol"].endswith("USDT")
        }

    async def fetch_universe_funding_info(self) -> Optional[Dict[str, float]]:
        """Fetch funding-interval metadata; absence of a symbol means default 8h only when the request succeeded."""
        url = f"{self.REST_BASE_URL}/fapi/v1/fundingInfo"
        data = await self._request_json("GET", url, weight=0)
        if not isinstance(data, list):
            return None
        out: Dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            if not isinstance(symbol, str) or not symbol.endswith("USDT"):
                continue
            try:
                interval_h = float(item["fundingIntervalHours"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(interval_h) or interval_h <= 0.0:
                continue
            out[symbol] = interval_h
        return out

    async def fetch_symbol_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current open interest; retained for external compatibility."""
        url = f"{self.REST_BASE_URL}/fapi/v1/openInterest"
        data = await self._request_json(
            "GET", url, params={"symbol": symbol}, weight=1, symbol=symbol
        )
        return data if isinstance(data, dict) else None

    async def fetch_symbol_closed_5m_klines(
        self,
        symbol: str,
        closed_open_ms: int,
        *,
        history_bars: int,
    ) -> Optional[List[list]]:
        """Fetch exactly `history_bars` completed 5m bars ending at `closed_open_ms`."""
        interval_ms = 5 * 60 * 1000
        if history_bars < 2 or closed_open_ms <= 0 or closed_open_ms % interval_ms != 0:
            raise ValueError("invalid completed-5m kline request")
        start_ms = closed_open_ms - (history_bars - 1) * interval_ms
        end_ms = closed_open_ms + interval_ms - 1
        limit = history_bars
        if limit > 1500:
            raise ValueError("history_bars exceeds Binance kline limit")
        weight = 1 if limit < 100 else (2 if limit < 500 else (5 if limit <= 1000 else 10))
        url = f"{self.REST_BASE_URL}/fapi/v1/klines"
        data = await self._request_json(
            "GET",
            url,
            params={
                "symbol": symbol,
                "interval": "5m",
                "startTime": str(start_ms),
                "endTime": str(end_ms),
                "limit": str(limit),
            },
            weight=weight,
            symbol=symbol,
        )
        if not isinstance(data, list) or len(data) != history_bars:
            logger.warning(
                "kline_window_incomplete symbol=%s expected=%d actual=%s",
                symbol, history_bars, len(data) if isinstance(data, list) else None,
            )
            return None

        opens: List[int] = []
        for row in data:
            if not isinstance(row, list) or len(row) < 12:
                return None
            try:
                open_ms = int(row[0])
                close_ms = int(row[6])
                numeric = [float(row[i]) for i in (1, 2, 3, 4, 5, 9)]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in numeric):
                return None
            if min(numeric[0:4]) <= 0.0 or numeric[4] < 0.0 or numeric[5] < 0.0:
                return None
            if close_ms != open_ms + interval_ms - 1:
                return None
            opens.append(open_ms)
        expected_opens = [start_ms + i * interval_ms for i in range(history_bars)]
        if opens != expected_opens or opens[-1] != closed_open_ms:
            logger.warning("kline_window_gap symbol=%s expected_last=%d actual=%s", symbol, closed_open_ms, opens[-1] if opens else None)
            return None
        return data

    async def fetch_symbol_open_interest_history(
        self,
        symbol: str,
        closed_open_ms: int,
        *,
        history_bars: int,
    ) -> Optional[List[Tuple[int, float]]]:
        """Return contiguous 5m OI observations ending at the just-closed candle."""
        interval_ms = 5 * 60 * 1000
        if history_bars < 2 or closed_open_ms <= 0 or closed_open_ms % interval_ms != 0:
            raise ValueError("invalid OI history request")
        if history_bars > 500:
            raise ValueError("history_bars exceeds Binance openInterestHist limit")
        start_ms = closed_open_ms - (history_bars - 1) * interval_ms
        end_ms = closed_open_ms + interval_ms - 1
        url = f"{self.REST_BASE_URL}/futures/data/openInterestHist"
        data = await self._request_json(
            "GET",
            url,
            params={
                "symbol": symbol,
                "period": "5m",
                "startTime": str(start_ms),
                "endTime": str(end_ms),
                "limit": str(history_bars),
            },
            weight=1,
            symbol=symbol,
        )
        if not isinstance(data, list):
            return None
        observations: List[Tuple[int, float]] = []
        for item in data:
            if not isinstance(item, dict):
                return None
            try:
                ts = int(item["timestamp"])
                oi = float(item["sumOpenInterest"])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(oi) or oi <= 0.0:
                return None
            # Binance openInterestHist timestamps are treated as 5m bucket anchors.
            # Floor to the canonical candle-open grid so OI and kline state align.
            normalized_open = (ts // interval_ms) * interval_ms
            if normalized_open < start_ms or normalized_open > closed_open_ms:
                continue
            observations.append((normalized_open, oi))
        observations.sort(key=lambda pair: pair[0])
        dedup: Dict[int, float] = {}
        for ts, oi in observations:
            dedup[ts] = oi
        observations = sorted(dedup.items())
        expected = [start_ms + i * interval_ms for i in range(history_bars)]
        if [ts for ts, _ in observations] != expected:
            logger.warning(
                "oi_history_gap symbol=%s expected=%d actual=%d",
                symbol, history_bars, len(observations),
            )
            return None
        return observations

    async def fetch_symbol_closed_5m_open_interest(
        self, symbol: str, current_open_ms: int
    ) -> Optional[float]:
        """Fetch the OI observation belonging to the just-closed 5m period."""
        interval_ms = 5 * 60 * 1000
        if current_open_ms <= 0 or current_open_ms % interval_ms != 0:
            raise ValueError("invalid completed-5m OI boundary")
        closed_open_ms = current_open_ms - interval_ms
        url = f"{self.REST_BASE_URL}/futures/data/openInterestHist"
        data = await self._request_json(
            "GET",
            url,
            params={
                "symbol": symbol,
                "period": "5m",
                "startTime": str(closed_open_ms),
                "endTime": str(current_open_ms),
                "limit": "10",
            },
            weight=1,
            symbol=symbol,
        )
        if not isinstance(data, list):
            return None
        candidates: List[Tuple[int, float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                ts = int(item["timestamp"])
                oi = float(item["sumOpenInterest"])
            except (KeyError, TypeError, ValueError):
                continue
            if closed_open_ms <= ts <= current_open_ms and math.isfinite(oi) and oi >= 0.0:
                candidates.append((ts, oi))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0])
        return candidates[-1][1]

    async def fetch_symbol_orderbook_top(self, symbol: str, limit: int = 20) -> Optional[OrderBookSnapshot]:
        """Fetch REST depth snapshot and compute top-10 OBI for cron cycle."""
        snapshot = await self.fetch_l2_snapshot(symbol, limit=limit)
        if not snapshot:
            return None
        try:
            raw_bids = [(float(p), float(q)) for p, q in snapshot.get("bids", [])]
            raw_asks = [(float(p), float(q)) for p, q in snapshot.get("asks", [])]
            last_update_id = int(snapshot["lastUpdateId"])
        except (KeyError, TypeError, ValueError):
            return None
        if not raw_bids or not raw_asks or last_update_id <= 0:
            return None
        if not all(math.isfinite(p) and math.isfinite(q) and p > 0.0 and q >= 0.0 for p, q in raw_bids + raw_asks):
            return None
        raw_bids.sort(key=lambda x: x[0], reverse=True)
        raw_asks.sort(key=lambda x: x[0])
        best_bid = raw_bids[0][0]
        best_ask = raw_asks[0][0]
        if best_bid >= best_ask:
            logger.warning("crossed_rest_book symbol=%s bid=%s ask=%s", symbol, best_bid, best_ask)
            return None
        top_bids = raw_bids[:10]
        top_asks = raw_asks[:10]
        mid = (best_bid + best_ask) * 0.5
        spread = best_ask - best_bid
        if mid <= 0.0:
            return None
        spread_bps = spread / mid * 10000.0
        bid_vol_5 = sum(q for _, q in top_bids[:5])
        ask_vol_5 = sum(q for _, q in top_asks[:5])
        tot_5 = bid_vol_5 + ask_vol_5
        bid_vol_10 = sum(q for _, q in top_bids)
        ask_vol_10 = sum(q for _, q in top_asks)
        tot_10 = bid_vol_10 + ask_vol_10
        if tot_5 <= 0.0 or tot_10 <= 0.0:
            return None
        return OrderBookSnapshot(
            symbol=symbol,
            last_update_id=last_update_id,
            timestamp_ms=int(time.time() * 1000),
            bids=tuple(top_bids),
            asks=tuple(top_asks),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            spread_bps=spread_bps,
            obi_depth5=(bid_vol_5 - ask_vol_5) / tot_5,
            obi_depth10=(bid_vol_10 - ask_vol_10) / tot_10,
        )

    async def fetch_5m_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch the complete aggregate-trade set inside a closed 5m window."""
        if start_time_ms < 0 or end_time_ms < start_time_ms:
            raise ValueError("invalid aggregate-trade window")
        if end_time_ms - start_time_ms >= 60 * 60 * 1000:
            raise ValueError("aggregate-trade query window must be < 1 hour")

        url = f"{self.REST_BASE_URL}/fapi/v1/aggTrades"
        params = {
            "symbol": symbol,
            "startTime": str(start_time_ms),
            "endTime": str(end_time_ms),
            "limit": "1000",
        }
        data = await self._request_json("GET", url, params=params, weight=20, symbol=symbol)
        if not isinstance(data, list):
            return None

        out: List[Dict[str, Any]] = []
        next_from_id: Optional[int] = None
        while data:
            last_id: Optional[int] = None
            for item in data:
                if not isinstance(item, dict):
                    return None
                try:
                    trade_id = int(item["a"])
                    ts = int(item["T"])
                    qty = float(item["q"])
                    maker = bool(item["m"])
                except (KeyError, TypeError, ValueError):
                    return None
                if trade_id <= 0 or ts < 0 or not math.isfinite(qty) or qty <= 0.0:
                    return None
                if last_id is not None and trade_id <= last_id:
                    return None
                last_id = trade_id
                if ts < start_time_ms:
                    continue
                if ts > end_time_ms:
                    data = []
                    break
                out.append({"a": trade_id, "T": ts, "q": qty, "m": maker})

            if not data or len(data) < 1000:
                break
            if last_id is None:
                return None
            next_from_id = last_id + 1
            if next_from_id <= last_id:
                return None
            data = await self._request_json(
                "GET",
                url,
                params={"symbol": symbol, "fromId": str(next_from_id), "limit": "1000"},
                weight=20,
                symbol=symbol,
            )
            if not isinstance(data, list):
                return None

        # IDs are monotonic in the API response; preserve that order for deterministic VPIN.
        if any(b["a"] <= a["a"] for a, b in zip(out, out[1:])):
            return None
        return out

    async def fetch_recent_agg_trades_cvd(self, symbol: str, limit: int = 100) -> Tuple[float, float, float]:
        """Deprecated: unbounded latest-N trade windows are not valid 5m aggregates."""
        raise RuntimeError(
            "fetch_recent_agg_trades_cvd is deprecated; use fetch_5m_agg_trades with explicit timestamps"
        )
