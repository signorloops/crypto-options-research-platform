"""
DuckDB embedded database for local analytics.

Provides SQL querying capabilities over Parquet files with excellent performance.
Ideal for research and backtesting analysis.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import duckdb
import pandas as pd

from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


def _sanitize_identifier(name: str) -> str:
    """Validate and sanitize SQL identifier (table/column name)."""
    if not name or not isinstance(name, str):
        raise ValueError("Identifier must be a non-empty string")
    # Allow alphanumeric, underscore, and quotes
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return f'"{name}"'


def _validate_timeframe(timeframe: str) -> str:
    """Validate timeframe format for resampling.

    Accepts formats like: 1s, 1m, 5min, 15min, 1H, 4H, 1D
    Converts to DuckDB INTERVAL format.
    """
    # Map common formats to DuckDB interval format
    format_map = {
        '1min': '1 minute', '5min': '5 minutes', '15min': '15 minutes',
        '30min': '30 minutes', '1H': '1 hour', '2H': '2 hours',
        '4H': '4 hours', '6H': '6 hours', '12H': '12 hours',
        '1D': '1 day', '1W': '1 week', '1M': '1 month',
        '1s': '1 second', '5s': '5 seconds', '10s': '10 seconds',
        '1m': '1 minute', '5m': '5 minutes'
    }

    if timeframe in format_map:
        return format_map[timeframe]

    # Also accept DuckDB native format directly
    valid_pattern = r'^\d+\s+(second|minute|hour|day|week|month)s?$'
    if re.match(valid_pattern, timeframe, re.IGNORECASE):
        return timeframe

    raise ValueError(f"Invalid timeframe: {timeframe}. Use format like '1min', '5min', '1H', '1D'")


class DuckDBCache:
    """
    DuckDB-based cache for local market data analytics.

    Features:
    - SQL queries over Parquet files
    - Automatic view creation for easy access
    - Time-series optimized queries
    - Join support for multi-table analysis
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize DuckDB cache.

        Args:
            db_path: Path to DuckDB database file. If None, use in-memory.
            read_only: Open in read-only mode (for concurrent access)
            cache_dir: Base directory for cache files. Defaults to CORP_CACHE_DIR env var or 'data/cache'.
        """
        self.con = None
        # Set cache directory from parameter, env var, or default
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            import os
            self.cache_dir = Path(os.getenv("CORP_CACHE_DIR", "data/cache"))

        try:
            if db_path is None:
                # In-memory database
                self.con = duckdb.connect(":memory:")
                logger.info("Initialized in-memory DuckDB")
            else:
                db_path = Path(db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.con = duckdb.connect(str(db_path), read_only=read_only)
                logger.info("Initialized DuckDB", extra=log_extra(path=str(db_path)))

            # Configure for optimal performance
            self._configure()
        except Exception:
            self.close()
            raise

    def _configure(self) -> None:
        """Configure DuckDB for optimal analytics performance."""
        import os

        # Enable parallel CSV reading
        self.con.execute("SET enable_progress_bar = false")
        # Memory limit from environment variable or default 2GB
        memory_limit_mb = int(os.getenv("DUCKDB_MEMORY_LIMIT_MB", "2048"))
        self.con.execute(f"SET memory_limit = '{memory_limit_mb}MB'")
        # Threads for parallel processing
        self.con.execute("SET threads = 4")

    def load_parquet(
        self,
        pattern: str,
        table_name: str,
        columns: Optional[List[str]] = None
    ) -> int:
        """
        Load Parquet files into a table.

        Args:
            pattern: Glob pattern for Parquet files (e.g., "data/cache/**/*.parquet")
            table_name: Name for the table/view
            columns: Optional column subset to load

        Returns:
            Number of rows loaded
        """
        try:
            # Validate table name to prevent SQL injection
            safe_table = _sanitize_identifier(table_name)

            if columns:
                # Validate column names
                safe_cols = ", ".join(_sanitize_identifier(col) for col in columns)
                query = f"""
                    CREATE OR REPLACE VIEW {safe_table} AS
                    SELECT {safe_cols} FROM read_parquet(?, hive_partitioning=1)
                """
            else:
                query = f"""
                    CREATE OR REPLACE VIEW {safe_table} AS
                    SELECT * FROM read_parquet(?, hive_partitioning=1)
                """

            self.con.execute(query, [pattern])

            # Get row count using safe table name
            count = self.con.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[0]
            logger.info(
                "Loaded Parquet into DuckDB",
                extra=log_extra(table=table_name, rows=count, pattern=pattern)
            )
            return count

        except Exception as e:
            logger.error(
                "Failed to load Parquet",
                extra=log_extra(pattern=pattern, error=str(e))
            )
            return 0

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        temporary: bool = False
    ) -> None:
        """
        Load a pandas DataFrame into DuckDB.

        Args:
            df: DataFrame to load
            table_name: Name for the table
            temporary: If True, create temporary table (session only)
        """
        try:
            safe_table = _sanitize_identifier(table_name)
            temp_flag = "TEMPORARY" if temporary else ""
            self.con.execute(f"CREATE OR REPLACE {temp_flag} TABLE {safe_table} AS SELECT * FROM df")
            logger.info(
                "Loaded DataFrame into DuckDB",
                extra=log_extra(table=table_name, rows=len(df))
            )
        except Exception as e:
            logger.error(
                "Failed to load DataFrame",
                extra=log_extra(table=table_name, error=str(e))
            )
            raise

    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            sql: SQL query string
            params: Optional query parameters (for parameterized queries)

        Returns:
            Query results as DataFrame
        """
        try:
            if params:
                result = self.con.execute(sql, params).fetchdf()
            else:
                result = self.con.execute(sql).fetchdf()
            return result
        except Exception as e:
            logger.error("Query failed", extra=log_extra(sql=sql[:100], error=str(e)))
            raise

    def query_scalar(self, sql: str) -> Any:
        """Execute query returning a single scalar value."""
        try:
            result = self.con.execute(sql).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error("Query scalar failed", extra=log_extra(sql=sql[:100], error=str(e)))
            raise

    def create_tick_view(
        self,
        exchange: str,
        instrument: str,
        view_name: Optional[str] = None
    ) -> str:
        """
        Create a view for tick data with time-based indexing.

        Args:
            exchange: Exchange name
            instrument: Instrument symbol
            view_name: Optional custom view name

        Returns:
            Name of created view
        """
        if view_name is None:
            safe_instrument = instrument.replace("-", "_").replace("/", "_")
            view_name = f"tick_{exchange}_{safe_instrument}"

        # Validate view name
        safe_view = _sanitize_identifier(view_name)

        # Use configurable cache directory instead of hardcoded path
        pattern = str(self.cache_dir / f"raw/{exchange}/tick/{instrument}/**/*.parquet")

        query = f"""
            CREATE OR REPLACE VIEW {safe_view} AS
            SELECT
                timestamp,
                instrument,
                bid,
                ask,
                bid_size,
                ask_size,
                (bid + ask) / 2 as mid,
                CASE WHEN (ask + bid) / 2 > 0 THEN (ask - bid) / ((ask + bid) / 2) ELSE NULL END as spread_pct,
                timestamp::DATE as date,
                EXTRACT(HOUR FROM timestamp) as hour
            FROM read_parquet(?, hive_partitioning=1)
            WHERE bid > 0 AND ask > 0
            ORDER BY timestamp
        """

        self.con.execute(query, [pattern])
        logger.info("Created tick view", extra=log_extra(view=view_name))
        return view_name

    def create_trade_view(
        self,
        exchange: str,
        instrument: str,
        view_name: Optional[str] = None
    ) -> str:
        """Create a view for trade data."""
        if view_name is None:
            safe_instrument = instrument.replace("-", "_").replace("/", "_")
            view_name = f"trade_{exchange}_{safe_instrument}"

        safe_view = _sanitize_identifier(view_name)
        # Use configurable cache directory instead of hardcoded path
        pattern = str(self.cache_dir / f"raw/{exchange}/trades/{instrument}/**/*.parquet")

        query = f"""
            CREATE OR REPLACE VIEW {safe_view} AS
            SELECT
                timestamp,
                instrument,
                price,
                size,
                side,
                price * size as notional,
                timestamp::DATE as date,
                EXTRACT(HOUR FROM timestamp) as hour
            FROM read_parquet(?, hive_partitioning=1)
            ORDER BY timestamp
        """

        self.con.execute(query, [pattern])
        logger.info("Created trade view", extra=log_extra(view=view_name))
        return view_name

    def get_price_stats(
        self,
        table_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get price statistics for a table.

        Args:
            table_name: Table/view name
            start: Optional start time
            end: Optional end time

        Returns:
            Dictionary with statistics
        """
        safe_table = _sanitize_identifier(table_name)
        where_clause = ""
        params = {}

        if start and end:
            where_clause = "WHERE timestamp >= $start AND timestamp <= $end"
            params = {"start": start, "end": end}
        elif start:
            where_clause = "WHERE timestamp >= $start"
            params = {"start": start}
        elif end:
            where_clause = "WHERE timestamp <= $end"
            params = {"end": end}

        query = f"""
            SELECT
                COUNT(*) as count,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                MIN(bid) as min_bid,
                MAX(ask) as max_ask,
                AVG((ask - bid) / ((ask + bid) / 2)) as avg_spread_pct,
                AVG(bid_size + ask_size) as avg_liquidity
            FROM {safe_table}
            {where_clause}
        """

        result = self.query(query, params if params else None)
        return result.iloc[0].to_dict()

    def resample_ohlcv(
        self,
        table_name: str,
        timeframe: str = "1H",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Resample tick/trade data to OHLCV format.

        Args:
            table_name: Source table with price data
            timeframe: Resample frequency ('1min', '5min', '15min', '1H', '4H', '1D')
            start: Optional start filter
            end: Optional end filter

        Returns:
            OHLCV DataFrame
        """
        safe_table = _sanitize_identifier(table_name)
        where_clause = ""
        params = {}

        if start and end:
            where_clause = "WHERE timestamp >= $start AND timestamp <= $end"
            params = {"start": start, "end": end}
        elif start:
            where_clause = "WHERE timestamp >= $start"
            params = {"start": start}
        elif end:
            where_clause = "WHERE timestamp <= $end"
            params = {"end": end}

        # Validate timeframe to prevent SQL injection
        safe_timeframe = _validate_timeframe(timeframe)

        # DuckDB time_bucket signature: time_bucket(interval, timestamp)
        # Using window functions for OHLC to ensure correctness
        query = f"""
            WITH bucketed AS (
                SELECT
                    time_bucket(INTERVAL '{safe_timeframe}', timestamp) as bucket,
                    timestamp,
                    mid,
                    ROW_NUMBER() OVER (PARTITION BY time_bucket(INTERVAL '{safe_timeframe}', timestamp) ORDER BY timestamp) as rn_asc,
                    ROW_NUMBER() OVER (PARTITION BY time_bucket(INTERVAL '{safe_timeframe}', timestamp) ORDER BY timestamp DESC) as rn_desc
                FROM {safe_table}
                {where_clause}
            )
            SELECT
                bucket as timestamp,
                MAX(CASE WHEN rn_asc = 1 THEN mid END) as open,
                MAX(mid) as high,
                MIN(mid) as low,
                MAX(CASE WHEN rn_desc = 1 THEN mid END) as close,
                COUNT(*) as tick_count
            FROM bucketed
            GROUP BY bucket
            ORDER BY bucket
        """

        return self.query(query, params if params else None)

    def export_to_parquet(
        self,
        query: str,
        output_path: str
    ) -> None:
        """
        Export query results to Parquet file.

        Args:
            query: SQL query (should be validated before calling)
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Note: query is not parameterized as it's a full query string
        # Caller should validate the query source
        export_query = f"""
            COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        self.con.execute(export_query, [str(output_path)])
        logger.info("Exported to Parquet", extra=log_extra(path=str(output_path)))

    def close(self) -> None:
        """Close DuckDB connection."""
        if self.con is not None:
            try:
                self.con.close()
                logger.info("DuckDB connection closed")
            except Exception as e:
                logger.warning("Error closing DuckDB connection", extra=log_extra(error=str(e)))
            finally:
                self.con = None

    def __enter__(self) -> 'DuckDBCache':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AnalyticsQueries:
    """
    Pre-built analytics queries for common use cases.
    """

    @staticmethod
    def spread_analysis(table_name: str) -> str:
        """Query for bid-ask spread analysis."""
        return f"""
            SELECT
                date,
                AVG(spread_pct) as avg_spread,
                MAX(spread_pct) as max_spread,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY spread_pct) as p95_spread
            FROM {table_name}
            GROUP BY date
            ORDER BY date
        """

    @staticmethod
    def liquidity_profile(table_name: str) -> str:
        """Query for liquidity analysis by hour."""
        return f"""
            SELECT
                hour,
                AVG(bid_size + ask_size) as avg_liquidity,
                COUNT(*) as tick_count
            FROM {table_name}
            GROUP BY hour
            ORDER BY hour
        """

    @staticmethod
    def trade_intensity(table_name: str, window_minutes: int = 5) -> str:
        """Query for trade intensity analysis."""
        return f"""
            SELECT
                TIME_BUCKET(INTERVAL '{window_minutes} minutes', timestamp) as window,
                COUNT(*) as trade_count,
                SUM(size) as total_volume,
                AVG(price) as vwap
            FROM {table_name}
            GROUP BY window
            ORDER BY window
        """

    @staticmethod
    def volatility_estimate(table_name: str, lookback_hours: int = 24) -> str:
        """Estimate realized volatility from tick data."""
        return f"""
            WITH returns AS (
                SELECT
                    timestamp,
                    mid,
                    LN(mid / LAG(mid) OVER (ORDER BY timestamp)) as log_return
                FROM {table_name}
                WHERE timestamp > NOW() - INTERVAL '{lookback_hours} hours'
            )
            SELECT
                STDDEV(log_return) * SQRT(365 * 24 * 60) as annualized_vol
            FROM returns
            WHERE log_return IS NOT NULL
        """
