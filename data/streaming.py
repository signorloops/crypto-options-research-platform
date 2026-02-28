"""
Real-time WebSocket streaming for market data.
Supports multiple exchanges with unified interface.
"""
import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Union

import websockets
from websockets.exceptions import ConnectionClosed

from core.types import OrderBook, OrderBookLevel, Tick, Trade
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StreamConfig:
    """Configuration for WebSocket stream."""
    reconnect_interval: float = 5.0  # Seconds between reconnection attempts
    max_reconnects: int = 10  # Maximum reconnection attempts
    ping_interval: float = 20.0  # Ping interval in seconds
    ping_timeout: float = 10.0  # Ping timeout in seconds


class WebSocketStream(ABC):
    """
    Abstract base class for WebSocket market data streams.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._websocket: Optional[Any] = None
        self._running = False
        self._reconnect_count = 0
        self._callbacks: Dict[str, List[Callable]] = {
            'tick': [],
            'trade': [],
            'orderbook': [],
            'error': []
        }

    @abstractmethod
    def get_ws_url(self, instruments: List[str]) -> str:
        """Get WebSocket URL for given instruments."""
        pass

    @abstractmethod
    def parse_message(self, message: str) -> Optional[Dict]:
        """Parse WebSocket message into standardized format."""
        pass

    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for specific event type."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable) -> None:
        """Remove callback for specific event type."""
        if event_type in self._callbacks and callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    async def _emit(self, event_type: str, data: Any) -> None:
        """Emit event to all registered callbacks."""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}")

    async def connect(self, instruments: List[str]) -> None:
        """Connect to WebSocket and start streaming."""
        self._running = True
        self._cancel_event = asyncio.Event()
        url = self.get_ws_url(instruments)

        while self._running and self._reconnect_count < self.config.max_reconnects:
            try:
                logger.info(f"Connecting to {url}...")
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=5.0,
                    open_timeout=10.0  # Timeout for connection establishment
                ) as websocket:
                    self._websocket = websocket
                    self._reconnect_count = 0
                    logger.info("Connected!")

                    try:
                        await self._handle_messages(websocket)
                    except asyncio.CancelledError:
                        logger.info("Message handling cancelled")
                        break

            except ConnectionClosed:
                if not self._running:
                    break
                logger.warning("Connection closed, reconnecting...")
                self._reconnect_count += 1
                # Exponential backoff with max limit
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0  # Max 60 second backoff
                )
                logger.info(f"Reconnecting in {backoff:.1f}s (attempt {self._reconnect_count})")
                await asyncio.sleep(backoff)

            except asyncio.CancelledError:
                logger.info("WebSocket connection cancelled")
                break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._reconnect_count += 1
                # Exponential backoff with max limit
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0
                )
                await asyncio.sleep(backoff)

    async def _handle_messages(self, websocket: Any) -> None:
        """Handle incoming WebSocket messages with backpressure control."""
        # Use a queue to implement backpressure
        message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        queue_full_logged = False

        async def producer():
            """Produce messages from WebSocket to queue."""
            nonlocal queue_full_logged
            try:
                async for message in websocket:
                    try:
                        message_queue.put_nowait(message)
                        queue_full_logged = False
                    except asyncio.QueueFull:
                        # Drop oldest message and add new one
                        if not queue_full_logged:
                            logger.warning("Message queue full, dropping old messages")
                            queue_full_logged = True
                        try:
                            message_queue.get_nowait()  # Remove oldest
                            message_queue.put_nowait(message)  # Add new
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass
            except ConnectionClosed:
                logger.info("WebSocket connection closed in producer")
            finally:
                # Signal consumer to stop
                await message_queue.put(None)

        async def consumer():
            """Consume messages from queue."""
            while True:
                try:
                    message = await message_queue.get()
                    if message is None:  # Shutdown signal
                        break

                    try:
                        parsed = self.parse_message(message)
                        if parsed:
                            await self._route_message(parsed)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        await self._emit('error', e)
                    except asyncio.CancelledError:
                        raise
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        try:
            await consumer()
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

    async def _route_message(self, parsed: Dict) -> None:
        """Route parsed message to appropriate callback."""
        # Handle orderbook reconstruction
        if isinstance(parsed, dict) and parsed.get('type') == 'orderbook':
            if hasattr(self, '_handle_orderbook_message'):
                reconstructed = self._handle_orderbook_message(parsed)
                if reconstructed:
                    await self._emit('orderbook', reconstructed)
                return

        msg_type = parsed.get('type')
        data = parsed.get('data')

        if msg_type and msg_type in self._callbacks:
            await self._emit(msg_type, data)

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if hasattr(self, '_cancel_event'):
            self._cancel_event.set()

        if self._websocket:
            try:
                # Wait for close with timeout
                await asyncio.wait_for(self._websocket.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket close timeout, forcing close")
            finally:
                self._websocket = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


from data.orderbook_reconstructor import (
    OrderBookReconstructor, MultiInstrumentReconstructor,
    OrderBookDelta, ReconstructionState
)


class DeribitStream(WebSocketStream):
    """
    Deribit WebSocket stream for real-time market data.
    Supports order book reconstruction with delta updates.
    """

    WS_URL = os.getenv("DERIBIT_WS_URL", "wss://www.deribit.com/ws/api/v2/public")

    def __init__(self, config: Optional[StreamConfig] = None, enable_reconstruction: bool = True):
        super().__init__(config)
        self._subscribed_instruments: Set[str] = set()
        self._reconstructors = MultiInstrumentReconstructor() if enable_reconstruction else None
        self._enable_reconstruction = enable_reconstruction

    def get_ws_url(self, instruments: List[str]) -> str:
        """Deribit uses single URL with subscription messages."""
        return self.WS_URL

    def parse_message(self, message: str) -> Optional[Dict]:
        """Parse Deribit message format with orderbook reconstruction support."""
        try:
            data = json.loads(message)

            # Handle different message types
            if 'params' in data:
                params = data['params']
                channel = params.get('channel', '')
                received_data = params.get('data', {})

                if 'trades' in channel:
                    return {'type': 'trade', 'data': self._parse_trade(received_data)}
                elif 'ticker' in channel:
                    return {'type': 'tick', 'data': self._parse_tick(received_data)}
                elif 'book' in channel:
                    # Return extended format for reconstruction
                    return self._parse_orderbook(received_data)

            return None

        except json.JSONDecodeError:
            return None

    def _parse_trade(self, data: Dict) -> Trade:
        """Parse trade data from Deribit format."""
        from core.types import OrderSide

        return Trade(
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0) / 1000, tz=timezone.utc),
            instrument=data.get('instrument_name', ''),
            price=data.get('price', 0),
            size=data.get('amount', 0),
            side=OrderSide.BUY if data.get('direction') == 'buy' else OrderSide.SELL,
            trade_id=data.get('trade_id')
        )

    def _parse_tick(self, data: Dict) -> Tick:
        """Parse tick data from Deribit format."""
        return Tick(
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0) / 1000, tz=timezone.utc),
            instrument=data.get('instrument_name', ''),
            bid=data.get('best_bid_price', 0),
            ask=data.get('best_ask_price', 0),
            bid_size=data.get('best_bid_amount', 0),
            ask_size=data.get('best_ask_amount', 0)
        )

    def _parse_orderbook(self, data: Dict) -> Dict:
        """Parse order book data from Deribit format with reconstruction support."""
        instrument = data.get('instrument_name', '')
        timestamp = datetime.fromtimestamp(data.get('timestamp', 0) / 1000, tz=timezone.utc)

        # Extract sequence numbers for gap detection
        change_id = data.get('change_id')
        prev_change_id = data.get('prev_change_id')

        # Parse bids and asks
        bids = [
            OrderBookLevel(price=float(b[0]), size=float(b[1]))
            for b in data.get('bids', [])
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), size=float(a[1]))
            for a in data.get('asks', [])
        ]

        order_book = OrderBook(
            timestamp=timestamp,
            instrument=instrument,
            bids=bids,
            asks=asks
        )

        # Build deltas for reconstruction
        deltas = []
        for bid in bids:
            deltas.append(OrderBookDelta(
                price=bid.price,
                size=bid.size,
                side='bid',
                sequence=change_id or 0,
                timestamp=timestamp
            ))
        for ask in asks:
            deltas.append(OrderBookDelta(
                price=ask.price,
                size=ask.size,
                side='ask',
                sequence=change_id or 0,
                timestamp=timestamp
            ))

        return {
            'type': 'orderbook',
            'order_book': order_book,
            'deltas': deltas,
            'instrument': instrument,
            'change_id': change_id,
            'prev_change_id': prev_change_id,
            'timestamp': timestamp
        }

    def _handle_orderbook_message(self, parsed: Dict) -> Optional[OrderBook]:
        """Handle order book message with reconstruction."""
        if not self._enable_reconstruction:
            return parsed.get('order_book')

        instrument = parsed.get('instrument', '')
        change_id = parsed.get('change_id')
        deltas = parsed.get('deltas', [])

        reconstructor = self._reconstructors.get_or_create(instrument)

        # Check if this is a snapshot (initial) or delta update
        if change_id is not None:
            # Check for gap
            if reconstructor.state.last_sequence is not None:
                expected = reconstructor.state.last_sequence + 1
                if change_id != expected:
                    logger.warning(
                        f"{instrument}: Gap detected! Expected {expected}, got {change_id}"
                    )
                    # Trigger resync by resetting and treating as snapshot
                    reconstructor.reset()

            # Apply deltas
            if deltas:
                reconstructor.apply_deltas(deltas)

            # Return reconstructed order book
            return reconstructor.get_order_book()

        return parsed.get('order_book')

    def _build_subscriptions(self, instruments: List[str]) -> List[str]:
        """Build subscription channels for instruments."""
        subscriptions = []
        for instrument in instruments:
            subscriptions.append(f"trades.{instrument}.100ms")
            subscriptions.append(f"ticker.{instrument}.100ms")
            subscriptions.append(f"book.{instrument}.none.10.100ms")
        return subscriptions

    async def _subscribe(self, websocket, instruments: List[str]) -> None:
        """Send subscription message."""
        subscriptions = self._build_subscriptions(instruments)
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {"channels": subscriptions}
        }
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(instruments)} instruments")

    async def connect(self, instruments: List[str]) -> None:
        """Override to send subscription messages after connection."""
        self._subscribed_instruments = set(instruments)

        # Connect and subscribe
        self._running = True
        url = self.WS_URL

        while self._running and self._reconnect_count < self.config.max_reconnects:
            try:
                logger.info("Connecting to Deribit WebSocket...")
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout
                ) as websocket:
                    self._websocket = websocket
                    self._reconnect_count = 0
                    logger.info("Connected!")

                    # Send subscription message on EVERY connection (including reconnects)
                    await self._subscribe(websocket, instruments)

                    await self._handle_messages(websocket)

            except ConnectionClosed:
                logger.warning("Connection closed, reconnecting...")
                self._reconnect_count += 1
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0
                )
                await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._reconnect_count += 1
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0
                )
                await asyncio.sleep(backoff)


class OKXStream(WebSocketStream):
    """
    OKX WebSocket stream for real-time market data (coin-margined options only).
    """

    WS_URL = os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")

    def __init__(
        self,
        instruments: Optional[List[str]] = None,
        config: Optional[StreamConfig] = None,
        order_book_callback: Optional[Callable[[OrderBook], None]] = None,
        trade_callback: Optional[Callable[[Trade], None]] = None
    ):
        super().__init__(config)
        self.instruments = instruments or []
        self._order_book_callback = order_book_callback
        self._trade_callback = trade_callback

    def get_ws_url(self, instruments: List[str]) -> str:
        """OKX uses single URL with subscription messages."""
        return self.WS_URL

    def parse_message(self, message: str) -> Optional[Dict]:
        """Parse OKX message format."""
        try:
            data = json.loads(message)

            # Handle event messages
            if data.get("event"):
                return None

            # Handle data messages
            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            payload = data.get("data", [])

            if not payload:
                return None

            item = payload[0]

            if "trades" in channel:
                return {"type": "trade", "data": self._parse_trade(item)}
            elif "tickers" in channel:
                return {"type": "tick", "data": self._parse_tick(item)}
            elif "books" in channel:
                return {"type": "orderbook", "data": self._parse_orderbook(item)}

            return None

        except json.JSONDecodeError:
            return None

    def _parse_trade(self, data: Dict[str, Any]) -> Trade:
        """Parse trade data from OKX format.

        Args:
            data: Raw trade data from OKX WebSocket

        Returns:
            Parsed Trade object
        """
        from core.types import OrderSide

        return Trade(
            timestamp=datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc),
            instrument=data.get("instId", ""),
            price=float(data.get("px", 0)),
            size=float(data.get("sz", 0)),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            trade_id=data.get("tradeId")
        )

    def _parse_tick(self, data: Dict[str, Any]) -> Tick:
        """Parse tick data from OKX format.

        Args:
            data: Raw ticker data from OKX WebSocket

        Returns:
            Parsed Tick object
        """
        return Tick(
            timestamp=datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc),
            instrument=data.get("instId", ""),
            bid=float(data.get("bidPx", 0)),
            ask=float(data.get("askPx", 0)),
            bid_size=float(data.get("bidSz", 0)),
            ask_size=float(data.get("askSz", 0))
        )

    def _parse_orderbook(self, data: Dict[str, Any]) -> OrderBook:
        """Parse order book data from OKX format.

        Args:
            data: Raw order book data from OKX WebSocket

        Returns:
            Parsed OrderBook object
        """
        bids = [
            OrderBookLevel(price=float(b[0]), size=float(b[1]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), size=float(a[1]))
            for a in data.get("asks", [])
        ]

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument=data.get("instId", ""),
            bids=bids,
            asks=asks
        )

    async def connect(self, instruments: Optional[List[str]] = None) -> None:
        """Override to send subscription messages after connection."""
        instruments = instruments or self.instruments

        # Build subscription messages
        subscriptions = []
        for instrument in instruments:
            subscriptions.append({"channel": "trades", "instId": instrument})
            subscriptions.append({"channel": "tickers", "instId": instrument})
            subscriptions.append({"channel": "books", "instId": instrument})

        # Connect and subscribe
        self._running = True
        url = self.WS_URL

        while self._running and self._reconnect_count < self.config.max_reconnects:
            try:
                logger.info("Connecting to OKX WebSocket...")
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout
                ) as websocket:
                    self._websocket = websocket
                    self._reconnect_count = 0
                    logger.info("Connected!")

                    # Send subscription message
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": subscriptions
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {len(instruments)} instruments")

                    await self._handle_messages(websocket)

            except ConnectionClosed:
                logger.warning("Connection closed, reconnecting...")
                self._reconnect_count += 1
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0
                )
                await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._reconnect_count += 1
                backoff = min(
                    self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
                    60.0
                )
                await asyncio.sleep(backoff)


class MultiExchangeStream:
    """
    Aggregate streams from multiple exchanges.
    """

    def __init__(self):
        self.streams: Dict[str, WebSocketStream] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            'tick': [],
            'trade': [],
            'orderbook': []
        }
        # Python 3.8 compatibility: asyncio.Task is not subscriptable.
        self._callback_tasks: Set[asyncio.Task] = set()

    def add_exchange(self, name: str, stream: WebSocketStream) -> None:
        """Add an exchange stream."""
        self.streams[name] = stream

        # Forward callbacks
        stream.add_callback('tick', lambda x, n=name: self._forward('tick', x, n))
        stream.add_callback('trade', lambda x, n=name: self._forward('trade', x, n))
        stream.add_callback('orderbook', lambda x, n=name: self._forward('orderbook', x, n))

    def _track_callback_task(
        self, task: asyncio.Task, event_type: str, exchange: str
    ) -> None:
        """Track async callback task and log failures."""
        self._callback_tasks.add(task)

        def _on_done(done_task: asyncio.Task) -> None:
            self._callback_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"Async {event_type} callback failed for {exchange}: {exc}")

        task.add_done_callback(_on_done)

    def _forward(self, event_type: str, data: Any, exchange: str) -> None:
        """Forward event with exchange info."""
        for callback in self._callbacks.get(event_type, []):
            try:
                result = callback(data, exchange)
                if asyncio.iscoroutine(result):
                    task = asyncio.create_task(result)
                    self._track_callback_task(task, event_type, exchange)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}")

    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for specific event type."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    async def connect_all(self, exchange_instruments: Dict[str, List[str]]) -> None:
        """Connect to all exchanges."""
        task_map: Dict[str, asyncio.Task] = {}
        for exchange_name, instruments in exchange_instruments.items():
            if exchange_name in self.streams:
                stream = self.streams[exchange_name]
                task_map[exchange_name] = asyncio.create_task(stream.connect(instruments))

        if not task_map:
            return

        results = await asyncio.gather(*task_map.values(), return_exceptions=True)
        failures: List[str] = []

        for exchange_name, result in zip(task_map.keys(), results):
            if isinstance(result, Exception):
                failures.append(exchange_name)
                logger.error(f"Exchange stream '{exchange_name}' failed: {result}")

        if failures:
            failed = ", ".join(failures)
            raise RuntimeError(f"Failed to start exchange stream(s): {failed}")

    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges."""
        for stream in self.streams.values():
            await stream.disconnect()

        if self._callback_tasks:
            pending = list(self._callback_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._callback_tasks.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect_all()
