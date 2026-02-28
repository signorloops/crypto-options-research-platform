"""
Tests for WebSocket streaming functionality.
"""
import asyncio
import json
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import OrderBook, OrderSide, Tick, Trade
from data.streaming import (
    DeribitStream,
    MultiExchangeStream,
    OKXStream,
    StreamConfig,
    WebSocketStream,
)


class TestStreamConfig:
    """Test StreamConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamConfig()
        assert config.reconnect_interval == 5.0
        assert config.max_reconnects == 10
        assert config.ping_interval == 20.0
        assert config.ping_timeout == 10.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StreamConfig(
            reconnect_interval=2.0,
            max_reconnects=5,
            ping_interval=15.0,
            ping_timeout=5.0
        )
        assert config.reconnect_interval == 2.0
        assert config.max_reconnects == 5
        assert config.ping_interval == 15.0
        assert config.ping_timeout == 5.0


class MockWebSocketStream(WebSocketStream):
    """Mock implementation for testing base class."""

    def get_ws_url(self, instruments):
        return "wss://test.example.com/ws"

    def parse_message(self, message):
        data = json.loads(message)
        return {'type': data.get('type'), 'data': data.get('data')}


class TestWebSocketStream:
    """Test WebSocketStream base class."""

    @pytest.fixture
    def stream(self):
        return MockWebSocketStream()

    @pytest.fixture
    def stream_with_config(self):
        config = StreamConfig(max_reconnects=3, reconnect_interval=0.1)
        return MockWebSocketStream(config=config)

    def test_initialization(self, stream):
        """Test stream initialization."""
        assert stream._running is False
        assert stream._reconnect_count == 0
        assert stream._websocket is None
        assert len(stream._callbacks['tick']) == 0
        assert len(stream._callbacks['trade']) == 0
        assert len(stream._callbacks['orderbook']) == 0
        assert len(stream._callbacks['error']) == 0

    def test_add_callback(self, stream):
        """Test adding callbacks."""
        callback = MagicMock()
        stream.add_callback('tick', callback)
        assert callback in stream._callbacks['tick']

    def test_add_callback_invalid_type(self, stream):
        """Test adding callback to invalid event type."""
        callback = MagicMock()
        stream.add_callback('invalid_type', callback)
        assert callback not in stream._callbacks.get('invalid_type', [])

    def test_remove_callback(self, stream):
        """Test removing callbacks."""
        callback = MagicMock()
        stream.add_callback('tick', callback)
        stream.remove_callback('tick', callback)
        assert callback not in stream._callbacks['tick']

    def test_remove_callback_not_exists(self, stream):
        """Test removing non-existent callback."""
        callback = MagicMock()
        # Should not raise error
        stream.remove_callback('tick', callback)

    @pytest.mark.asyncio
    async def test_emit_sync_callback(self, stream):
        """Test emitting to synchronous callback."""
        callback = MagicMock()
        stream.add_callback('tick', callback)

        test_data = {'price': 100.0}
        await stream._emit('tick', test_data)

        callback.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_emit_async_callback(self, stream):
        """Test emitting to asynchronous callback."""
        async_callback = AsyncMock()
        stream.add_callback('tick', async_callback)

        test_data = {'price': 100.0}
        await stream._emit('tick', test_data)

        async_callback.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_emit_error_handling(self, stream):
        """Test error handling in callback execution - errors are printed, not emitted."""
        error_callback = MagicMock(side_effect=ValueError("Test error"))

        stream.add_callback('tick', error_callback)

        test_data = {'price': 100.0}
        # Should not raise exception even if callback fails
        await stream._emit('tick', test_data)

        # Error callback was attempted
        error_callback.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_disconnect(self, stream):
        """Test disconnect method."""
        stream._running = True
        mock_ws = AsyncMock()
        stream._websocket = mock_ws

        await stream.disconnect()

        assert stream._running is False
        assert stream._websocket is None
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_no_websocket(self, stream):
        """Test disconnect when websocket is None."""
        stream._running = True
        stream._websocket = None

        # Should not raise error
        await stream.disconnect()
        assert stream._running is False

    @pytest.mark.asyncio
    async def test_context_manager(self, stream):
        """Test async context manager."""
        stream.disconnect = AsyncMock()

        async with stream as s:
            assert s is stream

        stream.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_message(self, stream):
        """Test message routing."""
        tick_callback = MagicMock()
        stream.add_callback('tick', tick_callback)

        parsed = {'type': 'tick', 'data': {'price': 100.0}}
        await stream._route_message(parsed)

        tick_callback.assert_called_once_with({'price': 100.0})

    @pytest.mark.asyncio
    async def test_route_message_unknown_type(self, stream):
        """Test routing unknown message type."""
        unknown_callback = MagicMock()
        stream.add_callback('unknown', unknown_callback)

        parsed = {'type': 'unknown_type', 'data': {}}
        await stream._route_message(parsed)

        # Callback should not be called for unregistered types
        unknown_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_messages(self, stream):
        """Test message handling loop."""
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = [
            json.dumps({'type': 'tick', 'data': {'price': 100.0}}),
            json.dumps({'type': 'trade', 'data': {'size': 1.0}}),
        ]

        tick_callback = MagicMock()
        trade_callback = MagicMock()
        stream.add_callback('tick', tick_callback)
        stream.add_callback('trade', trade_callback)

        # Run handle_messages for a short time
        stream._running = True

        async def stop_after_delay():
            await asyncio.sleep(0.1)
            stream._running = False

        # Cancel the task after messages are processed
        task = asyncio.create_task(stream._handle_messages(mock_websocket))
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass


class TestDeribitStream:
    """Test DeribitStream implementation."""

    @pytest.fixture
    def stream(self):
        return DeribitStream()

    def test_get_ws_url(self, stream):
        """Test WebSocket URL generation."""
        url = stream.get_ws_url(['BTC-PERPETUAL'])
        assert url == DeribitStream.WS_URL

    def test_parse_trade(self, stream):
        """Test parsing trade message."""
        message = json.dumps({
            'params': {
                'channel': 'trades.BTC-PERPETUAL.100ms',
                'data': {
                    'timestamp': 1640995200000,
                    'instrument_name': 'BTC-PERPETUAL',
                    'price': 50000.0,
                    'amount': 1.5,
                    'direction': 'buy',
                    'trade_id': '12345'
                }
            }
        })

        result = stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'trade'
        trade = result['data']
        assert isinstance(trade, Trade)
        assert trade.instrument == 'BTC-PERPETUAL'
        assert trade.price == 50000.0
        assert trade.size == 1.5
        assert trade.side == OrderSide.BUY
        assert trade.timestamp.tzinfo == timezone.utc

    def test_parse_tick(self, stream):
        """Test parsing tick message."""
        message = json.dumps({
            'params': {
                'channel': 'ticker.BTC-PERPETUAL.100ms',
                'data': {
                    'timestamp': 1640995200000,
                    'instrument_name': 'BTC-PERPETUAL',
                    'best_bid_price': 49999.0,
                    'best_ask_price': 50001.0,
                    'best_bid_amount': 2.0,
                    'best_ask_amount': 1.5
                }
            }
        })

        result = stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'tick'
        tick = result['data']
        assert isinstance(tick, Tick)
        assert tick.instrument == 'BTC-PERPETUAL'
        assert tick.bid == 49999.0
        assert tick.ask == 50001.0
        assert tick.timestamp.tzinfo == timezone.utc

    def test_parse_orderbook(self, stream):
        """Test parsing orderbook message."""
        message = json.dumps({
            'params': {
                'channel': 'book.BTC-PERPETUAL.none.10.100ms',
                'data': {
                    'instrument_name': 'BTC-PERPETUAL',
                    'bids': [[49999.0, 2.0], [49998.0, 3.0]],
                    'asks': [[50001.0, 1.5], [50002.0, 2.5]]
                }
            }
        })

        result = stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'orderbook'
        ob = result.get('order_book') or result.get('data')
        assert isinstance(ob, OrderBook)
        assert ob.instrument == 'BTC-PERPETUAL'
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2

    def test_parse_invalid_json(self, stream):
        """Test parsing invalid JSON."""
        message = "invalid json"
        result = stream.parse_message(message)
        assert result is None

    def test_parse_no_params(self, stream):
        """Test parsing message without params."""
        message = json.dumps({'result': 'success'})
        result = stream.parse_message(message)
        assert result is None


class TestOKXStream:
    """Test OKXStream implementation (coin-margined options only)."""

    @pytest.fixture
    def okx_stream(self):
        return OKXStream()

    def test_get_ws_url(self, okx_stream):
        """Test WebSocket URL."""
        url = okx_stream.get_ws_url(['BTC-USD-240628-50000-C'])
        assert url == "wss://ws.okx.com:8443/ws/v5/public"

    def test_parse_trade(self, okx_stream):
        """Test parsing trade message from OKX."""
        message = json.dumps({
            'arg': {'channel': 'trades', 'instId': 'BTC-USD-240628-50000-C'},
            'data': [{
                'ts': '1640995200000',
                'instId': 'BTC-USD-240628-50000-C',
                'px': '50000.00',
                'sz': '1.500',
                'side': 'buy',
                'tradeId': '123456'
            }]
        })

        result = okx_stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'trade'
        trade = result['data']
        assert isinstance(trade, Trade)
        assert trade.instrument == 'BTC-USD-240628-50000-C'
        assert trade.price == 50000.0
        assert trade.side == OrderSide.BUY
        assert trade.timestamp.tzinfo == timezone.utc

    def test_parse_trade_sell_side(self, okx_stream):
        """Test parsing sell trade from OKX."""
        message = json.dumps({
            'arg': {'channel': 'trades', 'instId': 'BTC-USD-240628-50000-C'},
            'data': [{
                'ts': '1640995200000',
                'instId': 'BTC-USD-240628-50000-C',
                'px': '50000.00',
                'sz': '1.500',
                'side': 'sell',
                'tradeId': '123456'
            }]
        })

        result = okx_stream.parse_message(message)
        trade = result['data']
        assert trade.side == OrderSide.SELL

    def test_parse_tick(self, okx_stream):
        """Test parsing tick message from OKX."""
        message = json.dumps({
            'arg': {'channel': 'tickers', 'instId': 'BTC-USD-240628-50000-C'},
            'data': [{
                'ts': '1640995200000',
                'instId': 'BTC-USD-240628-50000-C',
                'bidPx': '49999.00',
                'askPx': '50001.00',
                'bidSz': '2.000',
                'askSz': '1.500'
            }]
        })

        result = okx_stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'tick'
        tick = result['data']
        assert isinstance(tick, Tick)
        assert tick.instrument == 'BTC-USD-240628-50000-C'
        assert tick.bid == 49999.0
        assert tick.ask == 50001.0
        assert tick.timestamp.tzinfo == timezone.utc

    def test_parse_orderbook(self, okx_stream):
        """Test parsing order book message from OKX."""
        message = json.dumps({
            'arg': {'channel': 'books', 'instId': 'BTC-USD-240628-50000-C'},
            'data': [{
                'instId': 'BTC-USD-240628-50000-C',
                'bids': [['49999.00', '2.000'], ['49998.00', '1.500']],
                'asks': [['50001.00', '1.500'], ['50002.00', '2.000']]
            }]
        })

        result = okx_stream.parse_message(message)

        assert result is not None
        assert result['type'] == 'orderbook'
        ob = result['data']
        assert isinstance(ob, OrderBook)
        assert ob.instrument == 'BTC-USD-240628-50000-C'
        assert ob.bids[0].price == 49999.0
        assert ob.asks[0].price == 50001.0

    def test_parse_event_message(self, okx_stream):
        """Test parsing event message (subscribe response)."""
        message = json.dumps({
            'event': 'subscribe',
            'arg': {'channel': 'trades', 'instId': 'BTC-USD-240628-50000-C'}
        })

        result = okx_stream.parse_message(message)
        assert result is None

    def test_parse_invalid_json(self, okx_stream):
        """Test parsing invalid JSON."""
        message = "invalid json"
        result = okx_stream.parse_message(message)
        assert result is None

    def test_parse_empty_data(self, okx_stream):
        """Test parsing message with empty data."""
        message = json.dumps({
            'arg': {'channel': 'trades', 'instId': 'BTC-USD-240628-50000-C'},
            'data': []
        })

        result = okx_stream.parse_message(message)
        assert result is None


class TestMultiExchangeStream:
    """Test MultiExchangeStream aggregation."""

    @pytest.fixture
    def multi_stream(self):
        return MultiExchangeStream()

    @pytest.fixture
    def mock_deribit_stream(self):
        stream = MagicMock(spec=DeribitStream)
        stream.add_callback = MagicMock()
        return stream

    @pytest.fixture
    def mock_okx_stream(self):
        stream = MagicMock(spec=OKXStream)
        stream.add_callback = MagicMock()
        return stream

    def test_add_exchange(self, multi_stream, mock_deribit_stream):
        """Test adding exchange stream."""
        multi_stream.add_exchange('deribit', mock_deribit_stream)

        assert 'deribit' in multi_stream.streams
        assert mock_deribit_stream.add_callback.call_count == 3  # tick, trade, orderbook

    def test_add_callback(self, multi_stream):
        """Test adding callback."""
        callback = MagicMock()
        multi_stream.add_callback('tick', callback)

        assert callback in multi_stream._callbacks['tick']

    @pytest.mark.asyncio
    async def test_connect_all(self, multi_stream, mock_deribit_stream, mock_okx_stream):
        """Test connecting to all exchanges."""
        mock_deribit_stream.connect = AsyncMock()
        mock_okx_stream.connect = AsyncMock()

        multi_stream.add_exchange('deribit', mock_deribit_stream)
        multi_stream.add_exchange('okx', mock_okx_stream)

        await multi_stream.connect_all({
            'deribit': ['BTC-PERPETUAL'],
            'okx': ['BTC-USD-240628-50000-C']
        })

        mock_deribit_stream.connect.assert_called_once_with(['BTC-PERPETUAL'])
        mock_okx_stream.connect.assert_called_once_with(['BTC-USD-240628-50000-C'])

    @pytest.mark.asyncio
    async def test_connect_all_raises_on_exchange_failure(
        self, multi_stream, mock_deribit_stream, mock_okx_stream
    ):
        """Test connect_all surfaces stream startup failures."""
        mock_deribit_stream.connect = AsyncMock(side_effect=RuntimeError("deribit down"))
        mock_okx_stream.connect = AsyncMock()

        multi_stream.add_exchange('deribit', mock_deribit_stream)
        multi_stream.add_exchange('okx', mock_okx_stream)

        with pytest.raises(RuntimeError, match="deribit"):
            await multi_stream.connect_all({
                'deribit': ['BTC-PERPETUAL'],
                'okx': ['BTC-USD-240628-50000-C']
            })

    @pytest.mark.asyncio
    async def test_forward_logs_async_callback_errors(self, multi_stream, caplog):
        """Test async callback exceptions are logged and cleaned up."""
        async def broken_callback(data, exchange):
            raise RuntimeError("boom")

        multi_stream.add_callback('tick', broken_callback)

        with caplog.at_level("ERROR"):
            multi_stream._forward('tick', {'price': 1.0}, 'deribit')
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert "Async tick callback failed for deribit: boom" in caplog.text
        assert not multi_stream._callback_tasks

    @pytest.mark.asyncio
    async def test_disconnect_all(self, multi_stream, mock_deribit_stream, mock_okx_stream):
        """Test disconnecting from all exchanges."""
        mock_deribit_stream.disconnect = AsyncMock()
        mock_okx_stream.disconnect = AsyncMock()

        multi_stream.add_exchange('deribit', mock_deribit_stream)
        multi_stream.add_exchange('okx', mock_okx_stream)

        await multi_stream.disconnect_all()

        mock_deribit_stream.disconnect.assert_called_once()
        mock_okx_stream.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_all_cancels_pending_callback_tasks(self, multi_stream):
        """Test disconnect cleans up background callback tasks."""
        pending = asyncio.create_task(asyncio.sleep(10))
        multi_stream._callback_tasks.add(pending)

        await multi_stream.disconnect_all()

        assert pending.cancelled()
        assert not multi_stream._callback_tasks

    @pytest.mark.asyncio
    async def test_context_manager(self, multi_stream, mock_deribit_stream):
        """Test async context manager."""
        mock_deribit_stream.disconnect = AsyncMock()
        multi_stream.add_exchange('deribit', mock_deribit_stream)

        async with multi_stream as ms:
            assert ms is multi_stream

        mock_deribit_stream.disconnect.assert_called_once()


class TestIntegration:
    """Integration tests with mocked websockets."""

    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_connection_lifecycle(self, mock_connect):
        """Test full connection lifecycle."""
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        mock_websocket.__aenter__ = AsyncMock(return_value=mock_websocket)
        mock_websocket.__aexit__ = AsyncMock(return_value=False)

        # Simulate messages then connection close
        mock_websocket.__aiter__.return_value = [
            json.dumps({'type': 'tick', 'data': {'price': 100.0}}),
        ]
        mock_websocket.close = AsyncMock()

        stream = MockWebSocketStream()

        # Add callback
        received_data = []
        async def on_tick(data):
            received_data.append(data)
            stream._running = False  # Stop after first message

        stream.add_callback('tick', on_tick)

        # Run connection
        stream._running = True
        await stream.connect(['TEST'])

        # Verify connection was made
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnection_logic(self):
        """Test reconnection on connection failure."""
        stream = MockWebSocketStream(
            config=StreamConfig(max_reconnects=2, reconnect_interval=0.01)
        )

        with patch('websockets.connect', side_effect=ConnectionError("Test error")):
            stream._running = True

            # Run for short time then stop
            async def stop_stream():
                await asyncio.sleep(0.05)
                stream._running = False

            asyncio.create_task(stop_stream())
            await stream.connect(['TEST'])

            # Should have attempted reconnection
            assert stream._reconnect_count > 0
