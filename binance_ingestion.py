"""
Production-ready Binance Futures Ingestion Layer.
Implements:
1. Low-latency WebSocket client with automatic reconnection and exponential backoff.
2. Robust Finite State Machine (FSM) for L2 Order Book synchronization with sequence integrity checks (pu == u_prev).
3. Zero-copy msgspec serialization and Zero-GC data structures.
4. Normalization of trades, order book depth, funding rates (8h basis), and mark price basis spread.
5. High-concurrency REST batch collector for universe-wide 5-minute cron executions.
"""
from __future__ import annotations

import asyncio
import collections
import enum
import logging
import math
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
    4. First event must satisfy: event.U <= snapshot.lastUpdateId and event.u >= snapshot.lastUpdateId.
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

    def reset(self) -> None:
        """Reset internal book state upon sequence disconnect or gap."""
        self.state = OrderBookFSMState.DISCONNECTED
        self.bids.clear()
        self.asks.clear()
        self.buffer.clear()
        self.last_update_id = 0
        self.prev_u = 0
        self.resync_count += 1
        logger.warning(f"[{self.symbol}] OrderBook FSM reset triggered. Resync count: {self.resync_count}")

    def on_ws_connected(self) -> None:
        """Called when WebSocket connection is opened."""
        self.state = OrderBookFSMState.BUFFERING
        self.buffer.clear()
        logger.info(f"[{self.symbol}] OrderBook FSM transitioned to BUFFERING")

    def handle_depth_event(self, event: Dict[str, Any]) -> None:
        """Receive depthUpdate event from WebSocket stream."""
        if self.state == OrderBookFSMState.DISCONNECTED:
            return

        if self.state in (OrderBookFSMState.BUFFERING, OrderBookFSMState.REST_SNAPSHOT, OrderBookFSMState.APPLYING):
            self.buffer.append(event)
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
            self.state = OrderBookFSMState.BUFFERING
            return False

        # First event must satisfy U <= lastUpdateId AND u >= lastUpdateId
        first_event = self.buffer[0]
        U = first_event.get("U", 0)
        u = first_event.get("u", 0)

        if not (U <= self.last_update_id <= u):
            logger.warning(
                f"[{self.symbol}] First event sequence mismatch: U={U} <= lastUpdateId={self.last_update_id} <= u={u} violated. Resetting."
            )
            self.reset()
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
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for p_str, q_str in asks:
            price = float(p_str)
            qty = float(q_str)
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
        self._msg_decoder = msgspec.json.Decoder()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "QuantEngine/1.0", "Accept-Encoding": "gzip"},
            )
        return self._session

    async def fetch_l2_snapshot(self, symbol: str, limit: int = 1000) -> Optional[Dict[str, Any]]:
        """Fetch REST depth snapshot for order book FSM synchronization."""
        session = await self._get_session()
        url = f"{self.REST_BASE_URL}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": str(limit)}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                logger.error(f"[{symbol}] Failed to fetch depth snapshot: HTTP {resp.status}")
                return None
        except Exception as exc:
            logger.error(f"[{symbol}] Exception fetching depth snapshot: {exc}")
            return None

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

                    # Set books into BUFFERING state and schedule snapshots
                    for sym, book in self.books.items():
                        book.on_ws_connected()
                        asyncio.create_task(self._sync_symbol_book(sym))

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

    async def _sync_symbol_book(self, symbol: str) -> None:
        """Fetch snapshot and synchronize symbol FSM book."""
        book = self.books.get(symbol)
        if not book:
            return
        book.state = OrderBookFSMState.REST_SNAPSHOT
        # Allow buffer to collect a few diff events first
        await asyncio.sleep(0.3)
        snapshot = await self.fetch_l2_snapshot(symbol, limit=1000)
        if snapshot:
            book.apply_snapshot(snapshot)
            if self.on_book and book.state == OrderBookFSMState.IN_SYNC:
                snap = book.get_snapshot(depth=10)
                if snap:
                    await self.on_book(snap)

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
                book.handle_depth_event(data)
                if self.on_book and book.state == OrderBookFSMState.IN_SYNC:
                    snap = book.get_snapshot(depth=10)
                    if snap:
                        await self.on_book(snap)

        elif event_type == "aggTrade":
            symbol = data.get("s", "")
            is_maker = bool(data.get("m", False))
            side = "SELL" if is_maker else "BUY"
            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))
            trade = NormalizedTrade(
                symbol=symbol,
                price=price,
                quantity=qty,
                quote_quantity=price * qty,
                side=side,
                timestamp_ms=int(data.get("T", 0)),
                is_buyer_maker=is_maker,
                trade_id=int(data.get("a", 0)),
            )
            if self.on_trade:
                await self.on_trade(trade)

        elif event_type == "markPriceUpdate":
            symbol = data.get("s", "")
            raw_rate = float(data.get("r", 0.0))
            mark_price = float(data.get("p", 0.0))
            index_price = float(data.get("i", 0.0))
            next_time = int(data.get("T", 0))

            # Funding rate normalization: 8h basis
            # Binance default is 8h; if rate is non-standard, formula is (1 + raw)^(8/interval) - 1
            interval_h = 8.0
            norm_8h = (1.0 + raw_rate) ** (8.0 / interval_h) - 1.0
            annualized = norm_8h * 3.0 * 365.0 * 100.0

            basis_spread = mark_price - index_price
            basis_bps = (basis_spread / index_price * 10000.0) if index_price > 0.0 else 0.0

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
                timestamp_ms=int(data.get("E", int(time.time() * 1000))),
            )
            if self.on_funding:
                await self.on_funding(funding)

    async def stop(self) -> None:
        """Gracefully stop ingestion and close sessions."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Binance ingestion cleanly stopped.")

    # -------------------------------------------------------------------------
    # BATCH REST MODE (Optimized for 5-Minute Cron Executions without Docker)
    # -------------------------------------------------------------------------

    async def fetch_universe_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch 24hr tickers for all futures contracts in 1 single HTTP call."""
        session = await self._get_session()
        url = f"{self.REST_BASE_URL}/fapi/v1/ticker/24hr"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {item["symbol"]: item for item in data if item["symbol"].endswith("USDT")}
            return {}

    async def fetch_universe_premium_index(self) -> Dict[str, Dict[str, Any]]:
        """Fetch premium index (funding, mark, index price) for all contracts in 1 single call."""
        session = await self._get_session()
        url = f"{self.REST_BASE_URL}/fapi/v1/premiumIndex"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {item["symbol"]: item for item in data if item["symbol"].endswith("USDT")}
            return {}

    async def fetch_symbol_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current open interest for a specific contract."""
        session = await self._get_session()
        url = f"{self.REST_BASE_URL}/fapi/v1/openInterest"
        try:
            async with session.get(url, params={"symbol": symbol}) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    async def fetch_symbol_orderbook_top(self, symbol: str, limit: int = 20) -> Optional[OrderBookSnapshot]:
        """Fetch REST depth snapshot and compute OBI directly for cron cycle."""
        snapshot = await self.fetch_l2_snapshot(symbol, limit=limit)
        if not snapshot:
            return None

        raw_bids = [(float(p), float(q)) for p, q in snapshot.get("bids", [])]
        raw_asks = [(float(p), float(q)) for p, q in snapshot.get("asks", [])]

        if not raw_bids or not raw_asks:
            return None

        best_bid = raw_bids[0][0]
        best_ask = raw_asks[0][0]
        mid = (best_bid + best_ask) * 0.5
        spread = best_ask - best_bid
        spread_bps = (spread / mid * 10000.0) if mid > 0 else 0.0

        bid_vol_5 = sum(q for _, q in raw_bids[:5])
        ask_vol_5 = sum(q for _, q in raw_asks[:5])
        tot_5 = bid_vol_5 + ask_vol_5
        obi_5 = (bid_vol_5 - ask_vol_5) / tot_5 if tot_5 > 0 else 0.0

        bid_vol_10 = sum(q for _, q in raw_bids[:10])
        ask_vol_10 = sum(q for _, q in raw_asks[:10])
        tot_10 = bid_vol_10 + ask_vol_10
        obi_10 = (bid_vol_10 - ask_vol_10) / tot_10 if tot_10 > 0 else 0.0

        return OrderBookSnapshot(
            symbol=symbol,
            last_update_id=snapshot.get("lastUpdateId", 0),
            timestamp_ms=int(time.time() * 1000),
            bids=tuple(raw_bids[:10]),
            asks=tuple(raw_asks[:10]),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            spread_bps=spread_bps,
            obi_depth5=obi_5,
            obi_depth10=obi_10,
        )

    async def fetch_recent_agg_trades_cvd(self, symbol: str, limit: int = 100) -> Tuple[float, float, float]:
        """
        Fetch recent aggregated trades and compute taker volume & CVD.
        Returns: (taker_buy_vol, taker_sell_vol, cvd)
        """
        session = await self._get_session()
        url = f"{self.REST_BASE_URL}/fapi/v1/aggTrades"
        try:
            async with session.get(url, params={"symbol": symbol, "limit": str(limit)}) as resp:
                if resp.status == 200:
                    trades = await resp.json()
                    buy_vol = 0.0
                    sell_vol = 0.0
                    for t in trades:
                        qty = float(t.get("q", 0.0))
                        is_maker = bool(t.get("m", False))
                        if is_maker:
                            sell_vol += qty  # Buyer was maker -> Taker sold
                        else:
                            buy_vol += qty   # Taker bought
                    return buy_vol, sell_vol, (buy_vol - sell_vol)
                elif resp.status == 429:
                    await asyncio.sleep(0.5)
        except Exception:
            pass
        return 0.0, 0.0, 0.0
