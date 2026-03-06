"""
Integrated data manager combining Parquet cache, DuckDB analytics, and Redis real-time cache.

This module provides a unified interface for:
- File-based Parquet caching (historical data)
- DuckDB analytics (SQL queries on cached data)
- Redis real-time cache (Greeks, IV, orderbook with TTL)
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from redis.exceptions import RedisError

from data.cache import DataCache, DataManager
from data.cache_policy import (
    GREEKS_TTL_SECONDS,
    IV_TERM_TTL_SECONDS,
    IV_TTL_SECONDS,
    ORDERBOOK_TTL_SECONDS,
    TICKER_TTL_SECONDS,
    invalidation_patterns,
    realtime_ttls,
)
from data.duckdb_cache import DUCKDB_OPERATION_EXCEPTIONS, DuckDBCache
from data.redis_cache import GreeksCacheManager, RedisCache
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)

REDIS_CONNECT_EXCEPTIONS = (RedisError, OSError, ValueError, TypeError, RuntimeError)


class _IntegratedRealtimeCacheMixin:
    """Shared real-time Redis cache convenience wrappers."""

    async def cache_greeks(
        self, instrument: str, greeks: Dict[str, float], ttl_seconds: int = GREEKS_TTL_SECONDS
    ) -> None:
        """Cache Greeks data."""
        redis = self.redis
        if redis:
            await redis.set_greeks(instrument, greeks, ttl_seconds)

    async def get_iv(self, instrument: str) -> Optional[float]:
        """Get cached implied volatility."""
        redis = self.redis
        if not redis:
            return None
        return await redis.get_iv(instrument)

    async def cache_iv(self, instrument: str, iv: float, ttl_seconds: int = IV_TTL_SECONDS) -> None:
        """Cache implied volatility."""
        redis = self.redis
        if redis:
            await redis.set_iv(instrument, iv, ttl_seconds)

    async def get_iv_term_structure(self, underlying: str) -> Optional[Dict[str, Any]]:
        """Get cached IV term structure."""
        redis = self.redis
        if not redis:
            return None
        return await redis.get_iv_term_structure(underlying)

    async def cache_iv_term_structure(
        self,
        underlying: str,
        term_structure: List[Dict[str, Any]],
        ttl_seconds: int = IV_TERM_TTL_SECONDS,
    ) -> None:
        """Cache IV term structure."""
        redis = self.redis
        if redis:
            await redis.set_iv_term_structure(underlying, term_structure, ttl_seconds)

    async def get_orderbook(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached orderbook snapshot."""
        redis = self.redis
        if not redis:
            return None
        return await redis.get_orderbook(instrument)

    async def cache_orderbook(
        self, instrument: str, orderbook: Dict[str, Any], ttl_seconds: int = ORDERBOOK_TTL_SECONDS
    ) -> None:
        """Cache orderbook snapshot."""
        redis = self.redis
        if redis:
            await redis.set_orderbook(instrument, orderbook, ttl_seconds)

    async def get_ticker(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached ticker data."""
        redis = self.redis
        if not redis:
            return None
        return await redis.get_ticker(instrument)

    async def cache_ticker(
        self, instrument: str, ticker: Dict[str, Any], ttl_seconds: int = TICKER_TTL_SECONDS
    ) -> None:
        """Cache ticker data."""
        redis = self.redis
        if redis:
            await redis.set_ticker(instrument, ticker, ttl_seconds)

    async def publish(self, channel: str, message: Any) -> None:
        """Publish message to Redis channel."""
        redis = self.redis
        if redis:
            await redis.publish(channel, message)

    async def subscribe(self, *channels: str):
        """Subscribe to Redis channels."""
        redis = self.redis
        if not redis:
            raise RuntimeError("Redis not initialized")
        return await redis.subscribe(*channels)

    async def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if rate limited."""
        redis = self.redis
        if not redis:
            return False  # No Redis = no rate limiting
        return await redis.is_rate_limited(key, max_requests, window_seconds)

    async def invalidate_realtime_cache(
        self,
        instrument: Optional[str] = None,
        underlying: Optional[str] = None,
    ) -> int:
        """Invalidate Redis real-time cache by policy patterns."""
        redis = self.redis
        if not redis:
            return 0

        deleted = 0
        for pattern in invalidation_patterns(instrument=instrument, underlying=underlying):
            deleted += await redis.clear_pattern(pattern)
        return deleted


class IntegratedDataManager(_IntegratedRealtimeCacheMixin):
    """
    High-level data manager integrating all cache layers.

    Architecture:
    - Parquet (DataCache): Persistent storage for historical data
    - DuckDB (DuckDBCache): Analytical queries on Parquet files
    - Redis (RedisCache): Real-time cache with TTL for Greeks/IV/OrderBook

    Usage:
        async with IntegratedDataManager() as manager:
            # Get historical data (uses Parquet cache)
            df = await manager.get_historical_data(...)

            # Query with SQL (uses DuckDB)
            result = manager.query_duckdb("SELECT * FROM trades WHERE price > 50000")

            # Get real-time Greeks (uses Redis)
            greeks = await manager.get_greeks("BTC-27DEC24-80000-C")
    """

    def __init__(
        self,
        parquet_cache: Optional[DataCache] = None,
        duckdb_path: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_redis: bool = True,
        enable_duckdb: bool = True,
    ):
        """Initialize integrated data manager."""
        self.parquet_manager = DataManager(parquet_cache or DataCache())
        self.enable_duckdb = enable_duckdb
        self.duckdb_path = duckdb_path
        self.duckdb: Optional[DuckDBCache] = None
        self.enable_redis = enable_redis
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis: Optional[RedisCache] = None
        self.greeks_manager: Optional[GreeksCacheManager] = None
        self._state_lock: Optional[asyncio.Lock] = None
        logger.info(
            "IntegratedDataManager initialized",
            extra=log_extra(
                duckdb_enabled=enable_duckdb,
                redis_enabled=enable_redis,
                duckdb_path=duckdb_path or "memory",
                redis_host=redis_host if enable_redis else None,
            ),
        )

    def _get_state_lock(self) -> asyncio.Lock:
        """Create the async state lock lazily inside an active event loop."""
        if self._state_lock is None:
            self._state_lock = asyncio.Lock()
        return self._state_lock

    def _reset_redis_runtime(self) -> None:
        """Clear redis-derived runtime state after disconnect or failed init."""
        self.redis = None
        self.greeks_manager = None

    def _close_duckdb_runtime(self) -> None:
        """Close DuckDB backend if present and clear runtime state."""
        duckdb = self.duckdb
        if duckdb is not None:
            duckdb.close()
            self.duckdb = None

    async def connect(self) -> None:
        """Connect to Redis if enabled."""
        async with self._get_state_lock():
            if self.enable_redis:
                try:
                    self.redis = RedisCache(host=self.redis_host, port=self.redis_port)
                    await self.redis.connect()
                    self.greeks_manager = GreeksCacheManager(self.redis)
                    logger.info("Redis connected successfully")
                except REDIS_CONNECT_EXCEPTIONS as e:
                    logger.warning(f"Failed to connect to Redis: {e}")
                    self._reset_redis_runtime()

            if self.enable_duckdb:
                try:
                    self.duckdb = DuckDBCache(self.duckdb_path)
                    logger.info("DuckDB initialized successfully")
                except DUCKDB_OPERATION_EXCEPTIONS as e:
                    logger.warning(f"Failed to initialize DuckDB: {e}")
                    self.duckdb = None

    async def disconnect(self) -> None:
        """Disconnect from all services."""
        async with self._get_state_lock():
            if self.redis:
                await self.redis.disconnect()
                self._reset_redis_runtime()
                logger.info("Redis disconnected")

            if self.duckdb:
                self._close_duckdb_runtime()
                logger.info("DuckDB closed")

    async def __aenter__(self) -> "IntegratedDataManager":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # ==================== Parquet Cache Methods ====================

    async def get_historical_data(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        start: datetime,
        end: datetime,
        downloader: Optional[Callable] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical data with automatic caching.

        Args:
            exchange: Exchange name (e.g., 'deribit', 'okx')
            data_type: Data type (e.g., 'trades', 'ticks', 'orderbook')
            instrument: Instrument name
            start: Start datetime
            end: End datetime
            downloader: Async function to download missing data
            use_cache: Whether to use cache

        Returns:
            DataFrame with requested data
        """
        return await self.parquet_manager.get_data(
            exchange=exchange,
            data_type=data_type,
            instrument=instrument,
            start=start,
            end=end,
            downloader=downloader,
            use_cache=use_cache,
        )

    # ==================== DuckDB Analytics Methods ====================

    def _require_duckdb(self) -> DuckDBCache:
        """Return initialized DuckDB backend or raise a consistent error."""
        duckdb = self.duckdb
        if duckdb is None:
            raise RuntimeError("DuckDB not initialized")
        return duckdb

    def _build_exchange_cache_target(
        self,
        *,
        exchange: str,
        data_type: str,
        instrument: str,
    ) -> tuple[str, str]:
        """Build sanitized parquet glob and DuckDB table name for cached exchange data."""
        safe_instrument = instrument.replace("/", "_").replace("-", "_")
        cache_base = Path(self.parquet_manager.cache.raw_dir)
        pattern = str(cache_base / exchange / data_type / safe_instrument / "**/*.parquet")
        table_name = f"{exchange}_{data_type}_{safe_instrument}"
        return pattern, table_name

    def load_parquet_to_duckdb(
        self, pattern: str, table_name: str, columns: Optional[List[str]] = None
    ) -> int:
        """
        Load Parquet files into DuckDB for querying.

        Args:
            pattern: Glob pattern for Parquet files
            table_name: Name for the table/view
            columns: Optional column subset

        Returns:
            Number of rows loaded
        """
        return self._require_duckdb().load_parquet(pattern, table_name, columns)

    def load_exchange_data_to_duckdb(self, exchange: str, data_type: str, instrument: str) -> str:
        """
        Load cached exchange data into DuckDB.

        Args:
            exchange: Exchange name
            data_type: Data type
            instrument: Instrument

        Returns:
            Table/view name created
        """
        pattern, table_name = self._build_exchange_cache_target(
            exchange=exchange,
            data_type=data_type,
            instrument=instrument,
        )
        self._require_duckdb().load_parquet(pattern, table_name)
        return table_name

    def query_duckdb(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query on DuckDB.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            Query results as DataFrame
        """
        return self._require_duckdb().query(sql, params)

    def create_tick_view(self, exchange: str, instrument: str) -> str:
        """Create optimized tick view for analysis."""
        return self._require_duckdb().create_tick_view(exchange, instrument)

    def create_trade_view(self, exchange: str, instrument: str) -> str:
        """Create optimized trade view for analysis."""
        return self._require_duckdb().create_trade_view(exchange, instrument)

    def resample_to_ohlcv(
        self,
        table_name: str,
        timeframe: str = "1H",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Resample tick data to OHLCV format.

        Args:
            table_name: Source table name
            timeframe: Resample frequency ('1min', '5min', '1H', '1D')
            start: Optional start filter
            end: Optional end filter

        Returns:
            OHLCV DataFrame
        """
        return self._require_duckdb().resample_ohlcv(table_name, timeframe, start, end)

    # ==================== Redis Real-time Methods ====================

    async def get_greeks(
        self, instrument: str, fetch_func: Optional[Callable] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks with automatic cache refresh.

        Args:
            instrument: Option instrument name
            fetch_func: Optional function to fetch fresh Greeks

        Returns:
            Greeks data or None
        """
        greeks_manager = self.greeks_manager
        redis = self.redis

        if not greeks_manager:
            if fetch_func:
                return await fetch_func(instrument)
            return None

        if fetch_func:
            return await greeks_manager.get_greeks_with_refresh(instrument, fetch_func)
        if redis is None:
            return None
        return await redis.get_greeks(instrument)

    # ==================== Utility Methods ====================

    def get_cache_status(self) -> Dict[str, Any]:
        """Get status of all cache layers."""
        return {
            "parquet": self.parquet_manager.cache.get_cache_info(),
            "policy": {"ttl_seconds": realtime_ttls()},
            "duckdb": {
                "enabled": self.enable_duckdb,
                "initialized": self.duckdb is not None,
                "path": self.duckdb_path or "memory",
            },
            "redis": {
                "enabled": self.enable_redis,
                "connected": self.redis is not None,
                "host": self.redis_host if self.enable_redis else None,
                "port": self.redis_port if self.enable_redis else None,
            },
        }

    async def get_redis_stats(self) -> Optional[Dict[str, Any]]:
        """Get Redis statistics if connected."""
        redis = self.redis
        if not redis:
            return None
        return await redis.get_stats()

    def close(self) -> None:
        """Close all connections (sync version for cleanup)."""
        self._close_duckdb_runtime()


# Convenience function for creating default manager
def create_integrated_manager(
    duckdb_path: Optional[str] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    enable_redis: bool = True,
    enable_duckdb: bool = True,
) -> IntegratedDataManager:
    """
    Create IntegratedDataManager from environment or defaults.

    Environment variables:
    - REDIS_HOST (default: localhost)
    - REDIS_PORT (default: 6379)
    - DUCKDB_PATH (default: None, uses in-memory)
    """
    import os

    return IntegratedDataManager(
        duckdb_path=duckdb_path or os.getenv("DUCKDB_PATH"),
        redis_host=redis_host or os.getenv("REDIS_HOST", "localhost"),
        redis_port=redis_port or int(os.getenv("REDIS_PORT", "6379")),
        enable_redis=enable_redis,
        enable_duckdb=enable_duckdb,
    )
