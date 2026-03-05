"""Real-time WebSocket streaming for market data."""
import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.types import OrderBook, OrderBookLevel, Tick, Trade
from utils.logging_config import get_logger

logger = get_logger(__name__)

STREAM_CONNECTION_EXCEPTIONS = (
    OSError,
    ConnectionError,
    RuntimeError,
    asyncio.TimeoutError,
    WebSocketException,
)


async def _run_websocket_session(
    *,
    stream: "WebSocketStream",
    url: str,
    on_connected: Optional[Callable[[Any], Any]] = None,
) -> None:
    logger.info(f"Connecting to {url}...")
    async with websockets.connect(
        url,
        ping_interval=stream.config.ping_interval,
        ping_timeout=stream.config.ping_timeout,
        close_timeout=5.0,
        open_timeout=10.0,
    ) as websocket:
        stream._websocket = websocket
        stream._reconnect_count = 0
        logger.info("Connected!")
        if on_connected is not None:
            await on_connected(websocket)
        await stream._handle_messages(websocket)


async def _sleep_after_reconnect_signal(
    stream: "WebSocketStream", *, message: str, include_attempt_log: bool = False
) -> None:
    logger.warning(message)
    backoff = stream._next_reconnect_backoff()
    if include_attempt_log:
        logger.info(f"Reconnecting in {backoff:.1f}s (attempt {stream._reconnect_count})")
    await asyncio.sleep(backoff)


def _okx_subscription_args(instruments: List[str]) -> List[Dict[str, str]]:
    subscriptions: List[Dict[str, str]] = []
    for instrument in instruments:
        subscriptions.append({"channel": "trades", "instId": instrument})
        subscriptions.append({"channel": "tickers", "instId": instrument})
        subscriptions.append({"channel": "books", "instId": instrument})
    return subscriptions


async def _send_okx_subscribe(
    websocket: Any, *, instruments: List[str], subscriptions: List[Dict[str, str]]
) -> None:
    subscribe_msg = {"op": "subscribe", "args": subscriptions}
    await websocket.send(json.dumps(subscribe_msg))
    logger.info(f"Subscribed to {len(instruments)} instruments")


async def _deribit_on_connected(
    stream: "DeribitStream", websocket: Any, instruments: List[str]
) -> None:
    await stream._subscribe(websocket, instruments)


def _okx_payload_item(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if data.get("event"):
        return None
    payload = data.get("data", [])
    if not payload:
        return None
    return payload[0]


def _okx_channel_parser(
    stream: "OKXStream", channel: str
) -> Tuple[Optional[str], Optional[Callable[[Dict[str, Any]], Union[Trade, Tick, OrderBook]]]]:
    if "trades" in channel:
        return "trade", stream._parse_trade
    if "tickers" in channel:
        return "tick", stream._parse_tick
    if "books" in channel:
        return "orderbook", stream._parse_orderbook
    return None, None


@dataclass
class StreamConfig:
    """Configuration for WebSocket stream."""
    reconnect_interval: float = 5.0  # Seconds between reconnection attempts
    max_reconnects: int = 10  # Maximum reconnection attempts
    ping_interval: float = 20.0  # Ping interval in seconds
    ping_timeout: float = 10.0  # Ping timeout in seconds


class WebSocketStream(ABC):
    """Abstract base class for WebSocket market data streams."""

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

    def _next_reconnect_backoff(self) -> float:
        """Increment reconnect count and return exponential backoff seconds."""
        self._reconnect_count += 1
        return min(
            self.config.reconnect_interval * (2 ** (self._reconnect_count - 1)),
            60.0,  # Max 60 second backoff
        )

    @staticmethod
    def _enqueue_with_drop_oldest(message_queue: asyncio.Queue, message: Any) -> bool:
        """Push message to queue, dropping oldest on overflow. True when no drop was needed."""
        try:
            message_queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            try:
                message_queue.get_nowait()
                message_queue.put_nowait(message)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
            return False

    async def _emit(self, event_type: str, data: Any) -> None:
        """Emit event to all registered callbacks."""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                # Callback handlers are user-supplied; isolate failures per callback.
                logger.error(f"Error in {event_type} callback: {e}")

    async def connect(self, instruments: List[str]) -> None:
        """Connect to WebSocket and start streaming."""
        self._running = True
        self._cancel_event = asyncio.Event()
        url = self.get_ws_url(instruments)
        while self._running and self._reconnect_count < self.config.max_reconnects:
            try:
                await _run_websocket_session(stream=self, url=url)
            except ConnectionClosed:
                if not self._running:
                    break
                await _sleep_after_reconnect_signal(
                    self,
                    message="Connection closed, reconnecting...",
                    include_attempt_log=True,
                )
            except asyncio.CancelledError:
                logger.info("WebSocket connection cancelled")
                break
            except STREAM_CONNECTION_EXCEPTIONS as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self._next_reconnect_backoff())

    async def _produce_messages(self, websocket: Any, message_queue: asyncio.Queue) -> None:
        """Read from websocket and enqueue messages with drop-oldest backpressure."""
        queue_full_logged = False
        try:
            async for message in websocket:
                no_drop = self._enqueue_with_drop_oldest(message_queue, message)
                if no_drop:
                    queue_full_logged = False
                elif not queue_full_logged:
                    logger.warning("Message queue full, dropping old messages")
                    queue_full_logged = True
        except ConnectionClosed:
            logger.info("WebSocket connection closed in producer")
        finally:
            await message_queue.put(None)

    async def _process_message(self, message: Any) -> None:
        """Parse one websocket message and route events/errors."""
        try:
            parsed = self.parse_message(message)
            if parsed:
                await self._route_message(parsed)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            await self._emit("error", e)

    async def _consume_messages(self, message_queue: asyncio.Queue) -> None:
        """Consume queued messages until producer sentinel is received."""
        while True:
            try:
                if (message := await message_queue.get()) is None:
                    break
                await self._process_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _handle_messages(self, websocket: Any) -> None:
        message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        producer_task = asyncio.create_task(self._produce_messages(websocket, message_queue))
        try:
            await self._consume_messages(message_queue)
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


def _reconstructed_orderbook(
    *,
    reconstructor: OrderBookReconstructor,
    instrument: str,
    change_id: Optional[int],
    deltas: list[OrderBookDelta],
    fallback: Optional[OrderBook],
) -> Optional[OrderBook]:
    if change_id is None:
        return fallback
    if reconstructor.state.last_sequence is not None:
        expected = reconstructor.state.last_sequence + 1
        if change_id != expected:
            logger.warning(f"{instrument}: Gap detected! Expected {expected}, got {change_id}")
            reconstructor.reset()
    if deltas:
        reconstructor.apply_deltas(deltas)
    return reconstructor.get_order_book()


class DeribitStream(WebSocketStream):
    """Deribit WebSocket stream for real-time market data."""

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

    @staticmethod
    def _parse_levels(levels: list[Any]) -> list[OrderBookLevel]:
        return [OrderBookLevel(price=float(level[0]), size=float(level[1])) for level in levels]

    @staticmethod
    def _levels_to_deltas(
        *,
        levels: list[OrderBookLevel],
        side: str,
        sequence: int,
        timestamp: datetime,
    ) -> list[OrderBookDelta]:
        return [
            OrderBookDelta(
                price=level.price,
                size=level.size,
                side=side,
                sequence=sequence,
                timestamp=timestamp,
            )
            for level in levels
        ]

    def _parse_orderbook(self, data: Dict) -> Dict:
        """Parse order book data from Deribit format with reconstruction support."""
        instrument = data.get('instrument_name', '')
        timestamp = datetime.fromtimestamp(data.get('timestamp', 0) / 1000, tz=timezone.utc)
        change_id = data.get('change_id')
        prev_change_id = data.get('prev_change_id')
        bids = self._parse_levels(data.get('bids', []))
        asks = self._parse_levels(data.get('asks', []))
        order_book = OrderBook(
            timestamp=timestamp,
            instrument=instrument,
            bids=bids,
            asks=asks
        )
        sequence = change_id or 0
        deltas = self._levels_to_deltas(
            levels=bids, side='bid', sequence=sequence, timestamp=timestamp
        ) + self._levels_to_deltas(
            levels=asks, side='ask', sequence=sequence, timestamp=timestamp
        )

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
        return _reconstructed_orderbook(
            reconstructor=reconstructor,
            instrument=instrument,
            change_id=change_id,
            deltas=deltas,
            fallback=parsed.get('order_book'),
        )

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
                await _run_websocket_session(
                    stream=self,
                    url=url,
                    on_connected=lambda websocket: _deribit_on_connected(
                        self, websocket, instruments
                    ),
                )
            except ConnectionClosed:
                await _sleep_after_reconnect_signal(
                    self,
                    message="Connection closed, reconnecting...",
                )
            except STREAM_CONNECTION_EXCEPTIONS as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self._next_reconnect_backoff())


class OKXStream(WebSocketStream):
    """OKX WebSocket stream for real-time market data."""

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
            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            item = _okx_payload_item(data)
            if item is None:
                return None
            event_type, parser = _okx_channel_parser(self, channel)
            if event_type is None or parser is None:
                return None
            return {"type": event_type, "data": parser(item)}
        except json.JSONDecodeError:
            return None

    def _parse_trade(self, data: Dict[str, Any]) -> Trade:
        """Parse trade data from OKX format."""
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
        """Parse tick data from OKX format."""
        return Tick(
            timestamp=datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc),
            instrument=data.get("instId", ""),
            bid=float(data.get("bidPx", 0)),
            ask=float(data.get("askPx", 0)),
            bid_size=float(data.get("bidSz", 0)),
            ask_size=float(data.get("askSz", 0))
        )

    def _parse_orderbook(self, data: Dict[str, Any]) -> OrderBook:
        """Parse order book data from OKX format."""
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
        subscriptions = _okx_subscription_args(instruments)
        self._running = True
        url = self.WS_URL
        while self._running and self._reconnect_count < self.config.max_reconnects:
            try:
                await _run_websocket_session(
                    stream=self,
                    url=url,
                    on_connected=lambda websocket: _send_okx_subscribe(
                        websocket,
                        instruments=instruments,
                        subscriptions=subscriptions,
                    ),
                )
            except ConnectionClosed:
                await _sleep_after_reconnect_signal(
                    self,
                    message="Connection closed, reconnecting...",
                )
            except STREAM_CONNECTION_EXCEPTIONS as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self._next_reconnect_backoff())


class MultiExchangeStream:
    """Aggregate streams from multiple exchanges."""

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
                # Async callbacks are extension points and should not crash stream fanout.
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
                # Synchronous callback failures are isolated per callback.
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
