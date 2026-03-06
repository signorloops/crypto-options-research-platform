"""
Tests for integrated data manager wrappers and path behavior.
"""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from data.cache_policy import realtime_ttls
from data.cache import DataCache
from data.integrated_manager import IntegratedDataManager, create_integrated_manager


@pytest.mark.asyncio
async def test_get_historical_data_delegates_to_parquet_manager():
    """Historical data API should delegate to underlying DataManager."""
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    expected = pd.DataFrame({"price": [1.0, 2.0]})
    manager.parquet_manager.get_data = AsyncMock(return_value=expected)

    result = await manager.get_historical_data(
        exchange="okx",
        data_type="tick",
        instrument="BTC-USD",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        downloader=None,
        use_cache=True,
    )

    assert result.equals(expected)
    manager.parquet_manager.get_data.assert_awaited_once_with(
        exchange="okx",
        data_type="tick",
        instrument="BTC-USD",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        downloader=None,
        use_cache=True,
    )


def test_build_historical_request_returns_parquet_manager_kwargs():
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    request = manager._build_historical_request(
        exchange="okx",
        data_type="tick",
        instrument="BTC-USD",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        downloader=None,
        use_cache=False,
    )

    assert request == {
        "exchange": "okx",
        "data_type": "tick",
        "instrument": "BTC-USD",
        "start": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-01-02"),
        "downloader": None,
        "use_cache": False,
    }


def test_create_integrated_manager_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("DUCKDB_PATH", "/tmp/demo.duckdb")
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")

    manager = create_integrated_manager(enable_duckdb=False)

    assert manager.duckdb_path == "/tmp/demo.duckdb"
    assert manager.redis_host == "redis.internal"
    assert manager.redis_port == 6380
    assert manager.enable_duckdb is False


def test_create_integrated_manager_prefers_explicit_values_over_env(monkeypatch):
    monkeypatch.setenv("DUCKDB_PATH", "/tmp/from-env.duckdb")
    monkeypatch.setenv("REDIS_HOST", "redis.env")
    monkeypatch.setenv("REDIS_PORT", "6380")

    manager = create_integrated_manager(
        duckdb_path="/tmp/from-arg.duckdb",
        redis_host="redis.arg",
        redis_port=6390,
        enable_redis=False,
    )

    assert manager.duckdb_path == "/tmp/from-arg.duckdb"
    assert manager.redis_host == "redis.arg"
    assert manager.redis_port == 6390
    assert manager.enable_redis is False


def test_resolve_manager_config_uses_explicit_then_env_then_defaults():
    import data.integrated_manager as module

    config = module._resolve_manager_config(
        duckdb_path=None,
        redis_host=None,
        redis_port=None,
        env={
            "DUCKDB_PATH": "/tmp/env.duckdb",
            "REDIS_HOST": "redis.env",
            "REDIS_PORT": "6380",
        },
    )
    explicit = module._resolve_manager_config(
        duckdb_path="/tmp/arg.duckdb",
        redis_host="redis.arg",
        redis_port=6390,
        env={},
    )

    assert config == {
        "duckdb_path": "/tmp/env.duckdb",
        "redis_host": "redis.env",
        "redis_port": 6380,
    }
    assert explicit == {
        "duckdb_path": "/tmp/arg.duckdb",
        "redis_host": "redis.arg",
        "redis_port": 6390,
    }


@pytest.mark.asyncio
async def test_resolve_greeks_request_prefers_refresh_then_redis_then_fetch():
    import data.integrated_manager as module

    greeks_manager = MagicMock()
    greeks_manager.get_greeks_with_refresh = AsyncMock(return_value={"delta": 0.2})
    redis = MagicMock()
    redis.get_greeks = AsyncMock(return_value={"delta": 0.1})
    fetch_func = AsyncMock(return_value={"delta": 0.3})

    refreshed = await module._resolve_greeks_request(
        instrument="BTC",
        greeks_manager=greeks_manager,
        redis=redis,
        fetch_func=fetch_func,
    )
    cached = await module._resolve_greeks_request(
        instrument="BTC",
        greeks_manager=greeks_manager,
        redis=redis,
        fetch_func=None,
    )
    fetched = await module._resolve_greeks_request(
        instrument="BTC",
        greeks_manager=None,
        redis=None,
        fetch_func=fetch_func,
    )
    missing = await module._resolve_greeks_request(
        instrument="BTC",
        greeks_manager=None,
        redis=None,
        fetch_func=None,
    )

    assert refreshed == {"delta": 0.2}
    assert cached == {"delta": 0.1}
    assert fetched == {"delta": 0.3}
    assert missing is None


def test_load_exchange_data_to_duckdb_uses_sanitized_cache_path(tmp_path):
    """DuckDB loader should follow DataCache sanitization and base dir layout."""
    cache = DataCache(base_dir=tmp_path / "cache")
    manager = IntegratedDataManager(
        parquet_cache=cache,
        enable_duckdb=False,
        enable_redis=False,
    )
    manager.duckdb = MagicMock()

    table_name = manager.load_exchange_data_to_duckdb(
        exchange="okx",
        data_type="tick",
        instrument="BTC/USD-PERP",
    )

    assert table_name == "okx_tick_BTC_USD_PERP"
    called_pattern = manager.duckdb.load_parquet.call_args[0][0]
    assert "BTC_USD_PERP" in called_pattern
    assert str(cache.raw_dir) in called_pattern


def test_build_exchange_cache_target_returns_pattern_and_table_name(tmp_path):
    cache = DataCache(base_dir=tmp_path / "cache")
    manager = IntegratedDataManager(
        parquet_cache=cache,
        enable_duckdb=False,
        enable_redis=False,
    )

    pattern, table_name = manager._build_exchange_cache_target(
        exchange="okx",
        data_type="tick",
        instrument="BTC/USD-PERP",
    )

    assert table_name == "okx_tick_BTC_USD_PERP"
    assert "BTC_USD_PERP" in pattern
    assert str(cache.raw_dir) in pattern


@pytest.mark.asyncio
async def test_connect_gracefully_handles_backend_init_failures(monkeypatch):
    """Manager should degrade cleanly when optional backends fail to initialize."""
    import data.integrated_manager as module

    class BrokenRedis:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("redis down")

    class BrokenDuckDB:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("duckdb down")

    monkeypatch.setattr(module, "RedisCache", BrokenRedis)
    monkeypatch.setattr(module, "DuckDBCache", BrokenDuckDB)

    manager = IntegratedDataManager(enable_duckdb=True, enable_redis=True)
    await manager.connect()

    assert manager.redis is None
    assert manager.greeks_manager is None
    assert manager.duckdb is None


@pytest.mark.asyncio
async def test_connect_skips_disabled_backends(monkeypatch):
    """Disabled optional backends should not be constructed at all."""
    import data.integrated_manager as module

    class UnexpectedRedis:
        def __init__(self, *args, **kwargs):
            raise AssertionError("redis should stay disabled")

    class UnexpectedDuckDB:
        def __init__(self, *args, **kwargs):
            raise AssertionError("duckdb should stay disabled")

    monkeypatch.setattr(module, "RedisCache", UnexpectedRedis)
    monkeypatch.setattr(module, "DuckDBCache", UnexpectedDuckDB)

    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    await manager.connect()

    assert manager.redis is None
    assert manager.greeks_manager is None
    assert manager.duckdb is None


@pytest.mark.asyncio
async def test_invalidate_realtime_cache_uses_policy_patterns():
    """Invalidation should clear deterministic Redis key patterns."""
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    manager.redis = MagicMock()
    manager.redis.clear_pattern = AsyncMock(side_effect=[2, 1, 1, 1, 0])

    deleted = await manager.invalidate_realtime_cache(instrument="BTC-TEST", underlying="BTC")

    assert deleted == 5
    patterns = [call.args[0] for call in manager.redis.clear_pattern.await_args_list]
    assert patterns == [
        "greeks:BTC-TEST",
        "iv:BTC-TEST",
        "orderbook:BTC-TEST",
        "ticker:BTC-TEST",
        "iv_term:BTC",
    ]


def test_get_cache_status_includes_policy_ttls():
    """Cache status should surface policy TTLs for runtime introspection."""
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    status = manager.get_cache_status()

    assert "policy" in status
    assert status["policy"]["ttl_seconds"] == realtime_ttls()


def test_close_clears_duckdb_runtime():
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    duckdb = MagicMock()
    manager.duckdb = duckdb

    manager.close()

    duckdb.close.assert_called_once()
    assert manager.duckdb is None


def test_duckdb_wrappers_delegate_when_initialized():
    """DuckDB wrappers should call through to initialized backend."""
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    backend = MagicMock()
    backend.load_parquet.return_value = 12
    backend.query.return_value = pd.DataFrame({"x": [1]})
    backend.create_tick_view.return_value = "tick_view"
    backend.create_trade_view.return_value = "trade_view"
    backend.resample_ohlcv.return_value = pd.DataFrame({"close": [1.0]})
    manager.duckdb = backend

    assert manager.load_parquet_to_duckdb("*.parquet", "t") == 12
    assert manager.query_duckdb("select 1").iloc[0]["x"] == 1
    assert manager.create_tick_view("okx", "BTC-USD") == "tick_view"
    assert manager.create_trade_view("okx", "BTC-USD") == "trade_view"
    assert not manager.resample_to_ohlcv("t").empty


def test_require_duckdb_returns_backend_or_raises():
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)

    with pytest.raises(RuntimeError, match="DuckDB not initialized"):
        manager._require_duckdb()

    backend = MagicMock()
    manager.duckdb = backend
    assert manager._require_duckdb() is backend


def test_init_does_not_require_current_event_loop(monkeypatch):
    """Manager construction must not eagerly create asyncio primitives."""
    import data.integrated_manager as module

    def _unexpected_lock():
        raise RuntimeError("no current event loop")

    monkeypatch.setattr(module.asyncio, "Lock", _unexpected_lock)

    manager = module.IntegratedDataManager(enable_duckdb=False, enable_redis=False)

    assert manager is not None


@pytest.mark.asyncio
async def test_redis_wrappers_delegate_when_initialized():
    """Redis wrappers should delegate to backend methods when connected."""
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    redis = MagicMock()
    redis.get_greeks = AsyncMock(return_value={"delta": 0.1})
    redis.set_greeks = AsyncMock()
    redis.get_iv = AsyncMock(return_value=0.6)
    redis.set_iv = AsyncMock()
    redis.get_iv_term_structure = AsyncMock(return_value={"term": []})
    redis.set_iv_term_structure = AsyncMock()
    redis.get_orderbook = AsyncMock(return_value={"best_bid": 1})
    redis.set_orderbook = AsyncMock()
    redis.get_ticker = AsyncMock(return_value={"last": 10})
    redis.set_ticker = AsyncMock()
    redis.publish = AsyncMock()
    redis.subscribe = AsyncMock(return_value="subscribed")
    redis.is_rate_limited = AsyncMock(return_value=True)
    redis.get_stats = AsyncMock(return_value={"used_memory": 1})
    manager.redis = redis
    manager.greeks_manager = object()

    assert await manager.get_greeks("BTC") == {"delta": 0.1}
    await manager.cache_greeks("BTC", {"delta": 0.1})
    assert await manager.get_iv("BTC") == 0.6
    await manager.cache_iv("BTC", 0.6)
    assert await manager.get_iv_term_structure("BTC") == {"term": []}
    await manager.cache_iv_term_structure("BTC", [{"k": 1}])
    assert await manager.get_orderbook("BTC") == {"best_bid": 1}
    await manager.cache_orderbook("BTC", {"best_bid": 1})
    assert await manager.get_ticker("BTC") == {"last": 10}
    await manager.cache_ticker("BTC", {"last": 10})
    await manager.publish("topic", {"x": 1})
    assert await manager.subscribe("topic") == "subscribed"
    assert await manager.is_rate_limited("k", 1, 1) is True
    assert await manager.get_redis_stats() == {"used_memory": 1}


@pytest.mark.asyncio
async def test_connect_and_disconnect_success(monkeypatch):
    """Connect/disconnect should initialize and cleanup both optional backends."""
    import data.integrated_manager as module

    class GoodRedis:
        def __init__(self, *args, **kwargs):
            self.connected = False

        async def connect(self):
            self.connected = True

        async def disconnect(self):
            self.connected = False

    class GoodDuckDB:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisCache", GoodRedis)
    monkeypatch.setattr(module, "DuckDBCache", GoodDuckDB)

    manager = IntegratedDataManager(enable_duckdb=True, enable_redis=True)
    await manager.connect()
    assert manager.redis is not None
    assert manager.greeks_manager is not None
    assert manager.duckdb is not None

    await manager.disconnect()
    assert manager.redis is None
    assert manager.greeks_manager is None
    assert manager.duckdb is None


@pytest.mark.asyncio
async def test_disconnect_cleans_up_partial_backend_state():
    manager = IntegratedDataManager(enable_duckdb=False, enable_redis=False)
    redis = MagicMock()
    redis.disconnect = AsyncMock()
    manager.redis = redis
    manager.greeks_manager = object()

    await manager.disconnect()

    redis.disconnect.assert_awaited_once()
    assert manager.redis is None
    assert manager.greeks_manager is None

    duckdb = MagicMock()
    manager.duckdb = duckdb

    await manager.disconnect()

    duckdb.close.assert_called_once()
    assert manager.duckdb is None
