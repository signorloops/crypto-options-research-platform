# Cache Strategy and Consistency

This document defines cache TTLs, invalidation behavior, and consistency guarantees for CORP's three-layer cache stack.

## Cache Layers

- Parquet (`data/cache.py`): persistent historical snapshots by date partition.
- DuckDB (`data/duckdb_cache.py`): analytical SQL layer over Parquet datasets.
- Redis (`data/redis_cache.py`): low-latency real-time values.

## TTL Policy (Redis)

TTL values are centralized in `data/cache_policy.py` and can be overridden with environment variables.

| Key family | Default TTL | Env override |
|---|---:|---|
| `greeks:{instrument}` | 30s | `CACHE_TTL_GREEKS_SECONDS` |
| `iv:{instrument}` | 30s | `CACHE_TTL_IV_SECONDS` |
| `iv_term:{underlying}` | 300s | `CACHE_TTL_IV_TERM_SECONDS` |
| `orderbook:{instrument}` | 1s | `CACHE_TTL_ORDERBOOK_SECONDS` |
| `ticker:{instrument}` | 5s | `CACHE_TTL_TICKER_SECONDS` |

## Invalidation Rules

Use `IntegratedDataManager.invalidate_realtime_cache(...)` for deterministic invalidation.

- Instrument update: invalidates `greeks`, `iv`, `orderbook`, `ticker` for that instrument.
- Underlying IV surface update: invalidates `iv_term:{underlying}`.
- Full refresh: call without arguments to invalidate all real-time key families.

## Consistency Model

- Reads are eventually consistent across Parquet/DuckDB/Redis.
- Redis is source of truth for short-lived real-time values.
- Parquet is source of truth for historical replay and auditability.
- DuckDB should be refreshed from Parquet after write-heavy backfills.

## Stampede and Concurrency Safety

- `GreeksCacheManager` uses per-instrument singleflight locks to avoid duplicate refresh bursts.
- `IntegratedDataManager` uses `asyncio.Lock` for connection lifecycle state transitions.

## Operational Recommendations

- Trigger targeted invalidation after exchange reconnect and schema changes.
- Alert when Redis reconnect loops exceed threshold.
- Keep Parquet compaction as a scheduled maintenance task.
