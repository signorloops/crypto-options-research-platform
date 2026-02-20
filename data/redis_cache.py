"""
Redis cache layer for real-time data and fast lookups.

Features:
- Greeks caching with TTL
- IV term structure caching
- Pub/Sub for real-time updates
- Connection pooling
- Async support
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


class RedisCache:
    """
    Redis cache manager for real-time market data.

    Key patterns:
    - greeks:{instrument} -> Greeks data (30s TTL)
    - iv:{instrument} -> Implied volatility (30s TTL)
    - iv_term:{underlying} -> IV term structure (5min TTL)
    - orderbook:{instrument} -> OrderBook snapshot (1s TTL)
    - ticker:{instrument} -> Ticker data (5s TTL)
    - config:{key} -> Configuration (no TTL)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.decode_responses = decode_responses
        self._pool: Optional[redis.asyncio.Redis] = None

    def _check_connected(self) -> None:
        """Check if Redis is connected."""
        if self._pool is None:
            raise RuntimeError("Redis not connected. Call connect() first")

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        if self._pool is not None:
            raise RuntimeError("Redis already connected")

        try:
            self._pool = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
                socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
                socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
                retry_on_timeout=True
            )
            # Test connection
            await self._pool.ping()
            logger.info(
                "Redis connected",
                extra=log_extra(host=self.host, port=self.port, db=self.db)
            )
        except RedisConnectionError as e:
            logger.error("Redis connection failed", extra=log_extra(error=str(e)))
            self._pool = None
            raise
        except Exception as e:
            logger.error("Unexpected error connecting to Redis", extra=log_extra(error=str(e)))
            self._pool = None
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Redis disconnected")

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if self._pool is None:
            return False
        try:
            await self._pool.ping()
            return True
        except Exception:
            return False

    # Greeks caching

    async def set_greeks(
        self,
        instrument: str,
        greeks: Dict[str, float],
        ttl_seconds: int = 30
    ) -> None:
        """
        Cache Greeks data.

        Args:
            instrument: Option instrument name
            greeks: Dict with delta, gamma, theta, vega, rho
            ttl_seconds: Time to live (default 30s for real-time data)
        """
        self._check_connected()
        key = f"greeks:{instrument}"
        value = json.dumps(greeks)
        await self._pool.setex(key, ttl_seconds, value)

    async def get_greeks(self, instrument: str) -> Optional[Dict[str, float]]:
        """Get cached Greeks data."""
        self._check_connected()
        key = f"greeks:{instrument}"
        value = await self._pool.get(key)
        if value:
            return json.loads(value)
        return None

    async def set_greeks_batch(
        self,
        greeks_data: Dict[str, Dict[str, float]],
        ttl_seconds: int = 30
    ) -> None:
        """Batch cache multiple Greeks."""
        self._check_connected()
        pipe = self._pool.pipeline()
        for instrument, greeks in greeks_data.items():
            key = f"greeks:{instrument}"
            pipe.setex(key, ttl_seconds, json.dumps(greeks))
        await pipe.execute()

    async def get_greeks_batch(
        self,
        instruments: List[str]
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """Batch get Greeks."""
        self._check_connected()
        keys = [f"greeks:{inst}" for inst in instruments]
        values = await self._pool.mget(keys)
        return {
            inst: json.loads(val) if val else None
            for inst, val in zip(instruments, values)
        }

    # IV caching

    async def set_iv(
        self,
        instrument: str,
        iv: float,
        ttl_seconds: int = 30
    ) -> None:
        """Cache implied volatility."""
        self._check_connected()
        key = f"iv:{instrument}"
        await self._pool.setex(key, ttl_seconds, str(iv))

    async def get_iv(self, instrument: str) -> Optional[float]:
        """Get cached IV."""
        self._check_connected()
        key = f"iv:{instrument}"
        value = await self._pool.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning("Invalid IV value in cache", extra=log_extra(instrument=instrument, value=str(value)[:50]))
            return None

    async def set_iv_term_structure(
        self,
        underlying: str,
        term_structure: List[Dict[str, Any]],
        ttl_seconds: int = 300  # 5 minutes
    ) -> None:
        """
        Cache IV term structure.

        Args:
            underlying: e.g., "BTC-USD"
            term_structure: List of {expiry, days_to_expiry, atm_iv, strike}
            ttl_seconds: Longer TTL as this changes slowly
        """
        self._check_connected()
        key = f"iv_term:{underlying}"
        value = json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": term_structure
        })
        await self._pool.setex(key, ttl_seconds, value)

    async def get_iv_term_structure(
        self,
        underlying: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached IV term structure."""
        self._check_connected()
        key = f"iv_term:{underlying}"
        value = await self._pool.get(key)
        if value:
            return json.loads(value)
        return None

    # OrderBook caching

    async def set_orderbook(
        self,
        instrument: str,
        orderbook: Dict[str, Any],
        ttl_seconds: int = 1  # Very short TTL for orderbook
    ) -> None:
        """Cache orderbook snapshot."""
        self._check_connected()
        key = f"orderbook:{instrument}"
        value = json.dumps(orderbook)
        await self._pool.setex(key, ttl_seconds, value)

    async def get_orderbook(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached orderbook."""
        self._check_connected()
        key = f"orderbook:{instrument}"
        value = await self._pool.get(key)
        if value:
            return json.loads(value)
        return None

    # Ticker caching

    async def set_ticker(
        self,
        instrument: str,
        ticker: Dict[str, Any],
        ttl_seconds: int = 5
    ) -> None:
        """Cache ticker data."""
        self._check_connected()
        key = f"ticker:{instrument}"
        value = json.dumps(ticker)
        await self._pool.setex(key, ttl_seconds, value)

    async def get_ticker(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached ticker."""
        self._check_connected()
        key = f"ticker:{instrument}"
        value = await self._pool.get(key)
        if value:
            return json.loads(value)
        return None

    # Generic caching

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Generic cache set."""
        self._check_connected()
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if ttl_seconds:
            await self._pool.setex(key, ttl_seconds, serialized)
        else:
            await self._pool.set(key, serialized)

    async def get(self, key: str) -> Optional[Any]:
        """Generic cache get."""
        self._check_connected()
        value = await self._pool.get(key)
        if value and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def delete(self, key: str) -> None:
        """Delete a key."""
        self._check_connected()
        await self._pool.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        self._check_connected()
        return await self._pool.exists(key) > 0

    # Pub/Sub for real-time updates

    async def publish(self, channel: str, message: Any) -> None:
        """Publish message to channel."""
        self._check_connected()
        serialized = json.dumps(message) if not isinstance(message, str) else message
        await self._pool.publish(channel, serialized)
        logger.debug("Published message", extra=log_extra(channel=channel))

    @asynccontextmanager
    async def subscribe(self, *channels: str):
        """
        Subscribe to channels with automatic resource cleanup.

        Yields:
            PubSub object for listening

        Example:
            async with cache.subscribe("trades:deribit", "trades:okx") as pubsub:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        print(f"Received: {data}")
        """
        self._check_connected()
        pubsub = self._pool.pubsub()
        try:
            await pubsub.subscribe(*channels)
            logger.info("Subscribed to channels", extra=log_extra(channels=list(channels)))
            yield pubsub
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()
            logger.info("Unsubscribed from channels", extra=log_extra(channels=list(channels)))

    # Rate limiting

    # Lua script for atomic rate limiting
    # Returns: 1 if allowed, 0 if rate limited
    _RATE_LIMIT_SCRIPT = """
    local key = KEYS[1]
    local max_requests = tonumber(ARGV[1])
    local window_seconds = tonumber(ARGV[2])

    local current = redis.call('GET', key)
    if current == false then
        -- Key doesn't exist, create it
        redis.call('SET', key, 1, 'EX', window_seconds)
        return 1
    end

    local count = tonumber(current)
    if count >= max_requests then
        -- Refresh expiry even when rate limited for sliding window
        redis.call('EXPIRE', key, window_seconds)
        return 0
    end

    -- INCR and refresh expiry for sliding window behavior
    redis.call('INCR', key)
    redis.call('EXPIRE', key, window_seconds)
    return 1
    """

    async def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """
        Check if rate limit is exceeded using fixed window with atomic Lua script.

        Uses atomic Lua script to avoid race conditions between INCR and EXPIRE.

        Args:
            key: Rate limit key (e.g., "api:deribit:user123")
            max_requests: Maximum requests allowed
            window_seconds: Time window

        Returns:
            True if rate limited, False otherwise
        """
        self._check_connected()

        # Use Lua script for atomic operation
        result = await self._pool.eval(
            self._RATE_LIMIT_SCRIPT,
            1,  # numkeys
            key,
            str(max_requests),
            str(window_seconds)
        )
        return result == 0

    # Cache statistics

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        self._check_connected()
        info = await self._pool.info()
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "used_memory_human": info.get("used_memory_human", "N/A"),
            "connected_clients": info.get("connected_clients", 0),
            "total_keys": await self._pool.dbsize(),
            "hit_rate": hits / total if total > 0 else 0.0
        }

    async def clear_pattern(self, pattern: str, batch_size: int = 1000) -> int:
        """
        Clear all keys matching pattern.

        Uses batch deletion to avoid blocking Redis with large key sets.

        Args:
            pattern: Redis key pattern (e.g., "greeks:*")
            batch_size: Number of keys to delete per batch

        Returns:
            Number of keys deleted
        """
        self._check_connected()
        total_deleted = 0
        batch = []

        async def delete_batch(keys: list) -> int:
            if keys:
                await self._pool.delete(*keys)
                return len(keys)
            return 0

        async for key in self._pool.scan_iter(match=pattern):
            batch.append(key)
            if len(batch) >= batch_size:
                total_deleted += await delete_batch(batch)
                batch = []

        total_deleted += await delete_batch(batch)

        logger.info("Cleared cache pattern", extra=log_extra(pattern=pattern, count=total_deleted))
        return total_deleted

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key in seconds. Returns -1 if no TTL, -2 if not exists."""
        self._check_connected()
        return await self._pool.ttl(key)

    # Context managers

    async def __aenter__(self) -> 'RedisCache':
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


class GreeksCacheManager:
    """
    High-level manager for Greeks caching with automatic refresh.

    Uses singleflight pattern to prevent cache stampede - concurrent requests
    for the same missing key will share the same fetch operation.
    """

    def __init__(
        self,
        redis_cache: RedisCache,
        refresh_threshold_seconds: float = 20.0
    ):
        self.redis = redis_cache
        self.refresh_threshold = refresh_threshold_seconds
        self._refresh_tasks: set = set()
        # Singleflight locks to prevent cache stampede
        self._fetch_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects _fetch_locks
        # Cache for in-flight fetch results
        self._fetch_cache: Dict[str, Any] = {}
        self._lock_last_used: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal state for testing."""
        self._fetch_cache.clear()
        self._refresh_tasks.clear()
        self._fetch_locks.clear()
        self._lock_last_used.clear()

    async def _get_fetch_lock(self, instrument: str) -> asyncio.Lock:
        """Get or create a lock for the given instrument."""
        # Cleanup old locks periodically if we have many locks
        # Lower threshold for more aggressive cleanup
        if len(self._fetch_locks) > 100:
            await self._cleanup_old_locks()

        if instrument not in self._fetch_locks:
            async with self._locks_lock:
                # Double-check after acquiring lock
                if instrument not in self._fetch_locks:
                    self._fetch_locks[instrument] = asyncio.Lock()
                    self._lock_last_used[instrument] = asyncio.get_event_loop().time()
        else:
            # Update last used time
            self._lock_last_used[instrument] = asyncio.get_event_loop().time()
        return self._fetch_locks[instrument]

    async def _cleanup_old_locks(self) -> None:
        """Clean up locks that haven't been used for a while."""
        async with self._locks_lock:
            cutoff = asyncio.get_event_loop().time() - 600  # 10 minutes
            to_remove = [
                k for k, v in self._lock_last_used.items()
                if v < cutoff and not self._fetch_locks[k].locked()
            ]
            for k in to_remove:
                del self._fetch_locks[k]
                del self._lock_last_used[k]

    async def get_greeks_with_refresh(
        self,
        instrument: str,
        fetch_func: Callable[[str], Any]
    ) -> Dict[str, float]:
        """
        Get Greeks with automatic fetch on cache miss or stale data.

        Uses singleflight pattern to prevent cache stampede - if multiple
        concurrent requests ask for the same missing key, only one fetch
        operation will be executed and the result shared.

        Args:
            instrument: Option instrument
            fetch_func: Async function to fetch fresh Greeks

        Returns:
            Greeks data
        """
        # Try cache first
        greeks = await self.redis.get_greeks(instrument)

        if greeks is not None:
            # Cache hit - check if needs refresh
            ttl = await self.redis.get_ttl(f"greeks:{instrument}")
            if ttl < self.refresh_threshold:
                # Refresh in background
                logger.debug(
                    "Greeks stale, refreshing",
                    extra=log_extra(instrument=instrument, ttl=ttl)
                )
                task = asyncio.create_task(self._refresh_greeks(instrument, fetch_func))
                self._refresh_tasks.add(task)
                task.add_done_callback(lambda t: self._refresh_tasks.discard(t))
            return greeks

        # Cache miss - use singleflight to prevent stampede
        lock = await self._get_fetch_lock(instrument)

        async with lock:
            # Double-check after acquiring lock
            greeks = await self.redis.get_greeks(instrument)
            if greeks is not None:
                return greeks

            # Check if another coroutine already fetched it
            if instrument in self._fetch_cache:
                return self._fetch_cache[instrument]

            # Fetch and cache
            logger.debug("Greeks cache miss", extra=log_extra(instrument=instrument))
            try:
                greeks = await fetch_func(instrument)
                if greeks:
                    await self.redis.set_greeks(instrument, greeks)
                    # Store in fetch cache temporarily
                    self._fetch_cache[instrument] = greeks
                    # Clean up fetch cache after a short delay
                    asyncio.get_event_loop().call_later(
                        5.0, self._fetch_cache.pop, instrument, None
                    )
            except Exception as e:
                logger.error(
                    "Failed to fetch Greeks",
                    extra=log_extra(instrument=instrument, error=str(e))
                )
                raise

        return greeks

    async def _refresh_greeks(
        self,
        instrument: str,
        fetch_func: Callable[[str], Any]
    ) -> None:
        """Background refresh of Greeks."""
        try:
            greeks = await fetch_func(instrument)
            if greeks:
                await self.redis.set_greeks(instrument, greeks)
        except Exception as e:
            logger.warning(
                "Greeks refresh failed",
                extra=log_extra(instrument=instrument, error=str(e))
            )


# Convenience function for creating cache from config
def create_redis_cache(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None
) -> RedisCache:
    """
    Create RedisCache from environment or defaults.

    Environment variables:
    - REDIS_HOST (default: localhost)
    - REDIS_PORT (default: 6379)
    - REDIS_DB (default: 0)
    """
    import os

    return RedisCache(
        host=host or os.getenv("REDIS_HOST", "localhost"),
        port=port or int(os.getenv("REDIS_PORT", "6379")),
        db=db or int(os.getenv("REDIS_DB", "0"))
    )
