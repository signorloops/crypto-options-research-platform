"""Tests for Redis cache implementation.

These tests require a running Redis instance.
Skip if Redis is not available.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.redis_cache import GreeksCacheManager, RedisCache


@pytest.fixture
def mock_redis():
    """Create a mock Redis connection."""
    mock_pool = AsyncMock()
    mock_pool.ping = AsyncMock(return_value=True)
    mock_pool.close = AsyncMock()
    mock_pool.get = AsyncMock(return_value=None)
    mock_pool.setex = AsyncMock()
    mock_pool.set = AsyncMock()
    mock_pool.delete = AsyncMock()
    mock_pool.exists = AsyncMock(return_value=0)
    mock_pool.mget = AsyncMock(return_value=[])
    pipeline_mock = MagicMock()
    pipeline_mock.setex = MagicMock(return_value=pipeline_mock)
    pipeline_mock.execute = AsyncMock(return_value=True)
    mock_pool.pipeline = MagicMock(return_value=pipeline_mock)
    mock_pool.info = AsyncMock(return_value={
        'used_memory_human': '1M',
        'connected_clients': 1,
        'keyspace_hits': 100,
        'keyspace_misses': 10
    })
    mock_pool.dbsize = AsyncMock(return_value=0)
    mock_pool.ttl = AsyncMock(return_value=-1)
    mock_pool.publish = AsyncMock()
    mock_pool.pubsub = MagicMock(return_value=AsyncMock())
    mock_pool.scan_iter = AsyncMock(return_value=[])
    mock_pool.incr = AsyncMock(return_value=1)

    return mock_pool


@pytest.fixture
def redis_cache(mock_redis):
    """Create a RedisCache with mocked connection."""
    cache = RedisCache()
    cache._pool = mock_redis
    return cache


class TestRedisCacheConnection:
    """Test Redis connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis):
        """Test successful connection."""
        with patch('data.redis_cache.redis.Redis', return_value=mock_redis):
            cache = RedisCache()
            await cache.connect()
            assert cache._pool is not None
            mock_redis.ping.assert_called_once()
            await cache.disconnect()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        with patch('data.redis_cache.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value.ping = AsyncMock(side_effect=RedisConnectionError('Connection refused'))
            cache = RedisCache()
            with pytest.raises(RedisConnectionError, match='Connection refused'):
                await cache.connect()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnect when pool is None."""
        cache = RedisCache()
        # Should not raise
        await cache.disconnect()

    @pytest.mark.asyncio
    async def test_already_connected_raises(self, mock_redis):
        """Test connecting when already connected raises error."""
        with patch('data.redis_cache.redis.Redis', return_value=mock_redis):
            cache = RedisCache()
            await cache.connect()
            with pytest.raises(RuntimeError, match="already connected"):
                await cache.connect()


class TestRedisCache:
    """Test Redis cache functionality."""

    @pytest.mark.asyncio
    async def test_set_and_get_greeks(self, redis_cache, mock_redis):
        """Test setting and getting Greeks data."""
        greeks = {
            'delta': 0.5,
            'gamma': 0.01,
            'theta': -0.1,
            'vega': 0.2,
            'rho': 0.05
        }

        mock_redis.get = AsyncMock(return_value=json.dumps(greeks))

        await redis_cache.set_greeks('BTC-27DEC24-80000-C', greeks)
        result = await redis_cache.get_greeks('BTC-27DEC24-80000-C')

        assert result == greeks
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_greeks_cache_miss(self, redis_cache, mock_redis):
        """Test cache miss returns None."""
        mock_redis.get = AsyncMock(return_value=None)

        result = await redis_cache.get_greeks('NONEXISTENT')

        assert result is None

    @pytest.mark.asyncio
    async def test_set_greeks_batch(self, redis_cache, mock_redis):
        """Test batch setting Greeks."""
        greeks_data = {
            'BTC-27DEC24-80000-C': {'delta': 0.5, 'gamma': 0.01},
            'BTC-27DEC24-80000-P': {'delta': -0.5, 'gamma': 0.01},
        }
        pipe_mock = MagicMock()
        pipe_mock.setex = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=True)
        mock_redis.pipeline = MagicMock(return_value=pipe_mock)

        await redis_cache.set_greeks_batch(greeks_data)

        mock_redis.pipeline.assert_called_once()
        assert pipe_mock.setex.call_count == 2
        pipe_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_greeks_batch(self, redis_cache, mock_redis):
        """Test batch getting Greeks."""
        instruments = ['BTC-27DEC24-80000-C', 'BTC-27DEC24-80000-P']
        greeks_list = [
            json.dumps({'delta': 0.5, 'gamma': 0.01}),
            json.dumps({'delta': -0.5, 'gamma': 0.01}),
        ]
        mock_redis.mget = AsyncMock(return_value=greeks_list)

        result = await redis_cache.get_greeks_batch(instruments)

        mock_redis.mget.assert_called_once_with(['greeks:BTC-27DEC24-80000-C', 'greeks:BTC-27DEC24-80000-P'])
        assert result['BTC-27DEC24-80000-C']['delta'] == 0.5
        assert result['BTC-27DEC24-80000-P']['delta'] == -0.5

    @pytest.mark.asyncio
    async def test_set_and_get_iv(self, redis_cache, mock_redis):
        """Test IV caching."""
        mock_redis.get = AsyncMock(return_value='0.65')

        await redis_cache.set_iv('BTC-27DEC24-80000-C', 0.65)
        result = await redis_cache.get_iv('BTC-27DEC24-80000-C')

        assert result == 0.65

    @pytest.mark.asyncio
    async def test_get_iv_invalid_value(self, redis_cache, mock_redis):
        """Test IV caching with invalid value."""
        mock_redis.get = AsyncMock(return_value='invalid')

        result = await redis_cache.get_iv('BTC-27DEC24-80000-C')

        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_orderbook(self, redis_cache, mock_redis):
        """Test orderbook caching."""
        orderbook = {
            'bids': [[50000, 1.5], [49999, 2.0]],
            'asks': [[50001, 1.0], [50002, 2.5]],
            'timestamp': '2024-01-01T00:00:00Z'
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(orderbook))

        await redis_cache.set_orderbook('BTC-PERPETUAL', orderbook)
        result = await redis_cache.get_orderbook('BTC-PERPETUAL')

        assert result == orderbook
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_and_get_ticker(self, redis_cache, mock_redis):
        """Test ticker data caching."""
        ticker = {'price': 50000, 'volume': 1000}
        mock_redis.get = AsyncMock(return_value=json.dumps(ticker))

        await redis_cache.set_ticker('BTC-PERPETUAL', ticker)
        result = await redis_cache.get_ticker('BTC-PERPETUAL')

        assert result == ticker

    @pytest.mark.asyncio
    async def test_iv_term_structure(self, redis_cache, mock_redis):
        """Test IV term structure caching."""
        term_structure = [
            {'expiry': '2024-12-27', 'atm_iv': 0.65},
            {'expiry': '2025-01-31', 'atm_iv': 0.70}
        ]
        mock_redis.get = AsyncMock(return_value=json.dumps({
            'timestamp': '2024-01-01T00:00:00',
            'data': term_structure
        }))

        await redis_cache.set_iv_term_structure('BTC-USD', term_structure)
        result = await redis_cache.get_iv_term_structure('BTC-USD')

        assert result['data'] == term_structure

    @pytest.mark.asyncio
    async def test_generic_set_get(self, redis_cache, mock_redis):
        """Test generic set/get methods."""
        data = {'key': 'value', 'number': 42}
        mock_redis.get = AsyncMock(return_value=json.dumps(data))

        await redis_cache.set('mykey', data, ttl_seconds=60)
        result = await redis_cache.get('mykey')

        assert result == data

    @pytest.mark.asyncio
    async def test_exists(self, redis_cache, mock_redis):
        """Test key existence check."""
        mock_redis.exists = AsyncMock(return_value=1)

        result = await redis_cache.exists('mykey')

        assert result is True

    @pytest.mark.asyncio
    async def test_delete(self, redis_cache, mock_redis):
        """Test key deletion."""
        await redis_cache.delete('mykey')

        mock_redis.delete.assert_called_once_with('mykey')

    @pytest.mark.asyncio
    async def test_get_ttl(self, redis_cache, mock_redis):
        """Test TTL retrieval."""
        mock_redis.ttl = AsyncMock(return_value=45)

        result = await redis_cache.get_ttl('mykey')

        assert result == 45

    @pytest.mark.asyncio
    async def test_health_check(self, redis_cache, mock_redis):
        """Test health check."""
        result = await redis_cache.health_check()

        assert result is True
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        cache = RedisCache()
        result = await cache.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_failure(self, redis_cache, mock_redis):
        """Test health check failure."""
        mock_redis.ping = AsyncMock(side_effect=Exception('Connection refused'))

        result = await redis_cache.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_publish(self, redis_cache, mock_redis):
        """Test message publishing."""
        message = {'trade': 'data'}

        await redis_cache.publish('trades:deribit', message)

        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe(self, redis_cache, mock_redis):
        """Test pub/sub subscription."""
        pubsub_mock = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=pubsub_mock)

        async with redis_cache.subscribe('trades:deribit', 'trades:okx') as pubsub:
            mock_redis.pubsub.assert_called_once()
            pubsub_mock.subscribe.assert_called_once_with('trades:deribit', 'trades:okx')
            assert pubsub == pubsub_mock

        pubsub_mock.unsubscribe.assert_called_once()
        pubsub_mock.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_not_limited(self, redis_cache, mock_redis):
        """Test rate limiting - not limited."""
        # Lua script returns 1 for allowed
        mock_redis.eval = AsyncMock(return_value=1)

        result = await redis_cache.is_rate_limited('api:key', max_requests=10, window_seconds=60)

        assert result is False
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_limited(self, redis_cache, mock_redis):
        """Test rate limiting - limited."""
        # Lua script returns 0 for rate limited
        mock_redis.eval = AsyncMock(return_value=0)

        result = await redis_cache.is_rate_limited('api:key', max_requests=10, window_seconds=60)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, redis_cache, mock_redis):
        """Test getting cache statistics."""
        stats = await redis_cache.get_stats()

        assert 'used_memory_human' in stats
        assert 'connected_clients' in stats
        assert 'hit_rate' in stats
        assert 0 <= stats['hit_rate'] <= 1

    @pytest.mark.asyncio
    async def test_clear_pattern(self, redis_cache, mock_redis):
        """Test clearing keys by pattern."""
        async def mock_scan_iter(match):
            yield 'greeks:BTC-1'
            yield 'greeks:BTC-2'

        mock_redis.scan_iter = mock_scan_iter

        count = await redis_cache.clear_pattern('greeks:*')

        assert count == 2
        mock_redis.delete.assert_called_once_with('greeks:BTC-1', 'greeks:BTC-2')

    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        """Test that operations raise when not connected."""
        cache = RedisCache()

        with pytest.raises(RuntimeError, match="not connected"):
            await cache.get_greeks('BTC-27DEC24-80000-C')


class TestGreeksCacheManager:
    """Test Greeks cache manager."""

    @pytest.mark.asyncio
    async def test_get_greeks_cache_hit(self, mock_redis):
        """Test getting Greeks with cache hit."""
        greeks = {'delta': 0.5, 'gamma': 0.01}
        mock_redis.get = AsyncMock(return_value=json.dumps(greeks))
        mock_redis.ttl = AsyncMock(return_value=25)  # Above threshold

        redis_cache = RedisCache()
        redis_cache._pool = mock_redis

        manager = GreeksCacheManager(redis_cache, refresh_threshold_seconds=20.0)

        fetch_func = AsyncMock()
        result = await manager.get_greeks_with_refresh('BTC-27DEC24-80000-C', fetch_func)

        assert result == greeks
        fetch_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_greeks_cache_miss(self, mock_redis):
        """Test getting Greeks with cache miss."""
        mock_redis.get = AsyncMock(return_value=None)

        redis_cache = RedisCache()
        redis_cache._pool = mock_redis

        manager = GreeksCacheManager(redis_cache)

        fresh_greeks = {'delta': 0.5, 'gamma': 0.01}
        fetch_func = AsyncMock(return_value=fresh_greeks)

        result = await manager.get_greeks_with_refresh('BTC-27DEC24-80000-C', fetch_func)

        assert result == fresh_greeks
        fetch_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_greeks_ttl_refresh(self, mock_redis):
        """Test Greeks refresh when TTL is below threshold."""
        greeks = {'delta': 0.5, 'gamma': 0.01}
        mock_redis.get = AsyncMock(return_value=json.dumps(greeks))
        mock_redis.ttl = AsyncMock(return_value=10)  # Below 20s threshold

        redis_cache = RedisCache()
        redis_cache._pool = mock_redis

        manager = GreeksCacheManager(redis_cache, refresh_threshold_seconds=20.0)

        fetch_func = AsyncMock(return_value={'delta': 0.6, 'gamma': 0.02})

        with patch('asyncio.create_task') as mock_create_task:
            def _create_task(coro):
                # Close coroutine in test to avoid "never awaited" warnings.
                coro.close()
                task = MagicMock()
                task.add_done_callback = MagicMock()
                return task

            mock_create_task.side_effect = _create_task
            result = await manager.get_greeks_with_refresh('BTC-27DEC24-80000-C', fetch_func)

        assert result == greeks  # Returns cached value
        fetch_func.assert_not_called()  # Doesn't wait for fetch
        mock_create_task.assert_called_once()  # But schedules background refresh


class TestRedisCacheContextManager:
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_redis):
        """Test async context manager."""
        with patch('data.redis_cache.redis.Redis', return_value=mock_redis):
            async with RedisCache() as cache:
                assert cache._pool is not None

            # After exit, should be disconnected
            mock_redis.close.assert_called_once()


class TestCreateRedisCache:
    """Test factory function."""

    def test_create_from_env(self, monkeypatch):
        """Test creating cache from environment variables."""
        monkeypatch.setenv('REDIS_HOST', 'redis.example.com')
        monkeypatch.setenv('REDIS_PORT', '6380')
        monkeypatch.setenv('REDIS_DB', '1')

        from data.redis_cache import create_redis_cache
        cache = create_redis_cache()

        assert cache.host == 'redis.example.com'
        assert cache.port == 6380
        assert cache.db == 1
