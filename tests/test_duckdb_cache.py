"""Tests for DuckDB cache implementation."""
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.duckdb_cache import AnalyticsQueries, DuckDBCache, _sanitize_identifier


@pytest.fixture
def duckdb_cache():
    """Create a DuckDB cache with automatic cleanup."""
    cache = DuckDBCache()
    yield cache
    cache.close()


@pytest.fixture
def sample_tick_data():
    """Create sample tick data."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='min'),
        'bid': [100.0 + i * 0.01 for i in range(100)],
        'ask': [101.0 + i * 0.01 for i in range(100)],
        'bid_size': [1.0] * 100,
        'ask_size': [1.0] * 100
    })


class TestDuckDBCache:
    """Test DuckDB cache functionality."""

    def test_in_memory_connection(self, duckdb_cache):
        """Test in-memory database connection."""
        result = duckdb_cache.query_scalar("SELECT 1 + 1")
        assert result == 2

    def test_file_based_connection(self):
        """Test file-based database connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = DuckDBCache(str(db_path))
            try:
                result = cache.query_scalar("SELECT 42")
                assert result == 42
            finally:
                cache.close()
            assert db_path.exists()

    def test_load_dataframe(self, duckdb_cache, sample_tick_data):
        """Test loading DataFrame into DuckDB."""
        duckdb_cache.load_dataframe(sample_tick_data, 'test_table')
        result = duckdb_cache.query('SELECT COUNT(*) as cnt FROM test_table')

        assert len(result) == 1
        assert result.iloc[0]['cnt'] == 100

    def test_load_dataframe_invalid_table_name(self, duckdb_cache):
        """Test that invalid table names raise error."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            duckdb_cache.load_dataframe(df, 'table with spaces')

    def test_query_with_params(self, duckdb_cache):
        """Test parameterized queries."""
        df = pd.DataFrame({
            'instrument': ['BTC', 'ETH', 'BTC', 'ETH'],
            'price': [50000, 3000, 51000, 3100]
        })

        duckdb_cache.load_dataframe(df, 'trades')
        result = duckdb_cache.query(
            'SELECT * FROM trades WHERE instrument = $instrument',
            {'instrument': 'BTC'}
        )

        assert len(result) == 2
        assert all(result['instrument'] == 'BTC')

    def test_query_scalar(self, duckdb_cache):
        """Test scalar query."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        duckdb_cache.load_dataframe(df, 'numbers')

        result = duckdb_cache.query_scalar('SELECT SUM(value) FROM numbers')
        assert result == 15

    def test_invalid_query_raises(self, duckdb_cache):
        """Test that invalid queries raise exceptions."""
        with pytest.raises(Exception):
            duckdb_cache.query("SELECT * FROM nonexistent_table")

    def test_query_scalar_error(self, duckdb_cache):
        """Test scalar query error handling."""
        with pytest.raises(Exception):
            duckdb_cache.query_scalar("SELECT * FROM nonexistent_table")


class TestSanitizeIdentifier:
    """Test SQL identifier sanitization."""

    def test_valid_identifier(self):
        """Test valid identifiers pass."""
        assert _sanitize_identifier('valid_table') == '"valid_table"'
        assert _sanitize_identifier('_underscore') == '"_underscore"'
        assert _sanitize_identifier('table123') == '"table123"'

    def test_invalid_identifier(self):
        """Test invalid identifiers raise error."""
        with pytest.raises(ValueError):
            _sanitize_identifier('table with spaces')

        with pytest.raises(ValueError):
            _sanitize_identifier('table; DROP TABLE users')

        with pytest.raises(ValueError, match="non-empty string"):
            _sanitize_identifier('')

        with pytest.raises(ValueError):
            _sanitize_identifier(None)


class TestAnalyticsQueries:
    """Test pre-built analytics queries."""

    def test_spread_analysis_query(self):
        """Test spread analysis query generation."""
        query = AnalyticsQueries.spread_analysis('tick_data')

        assert 'spread_pct' in query
        assert 'AVG' in query
        assert 'PERCENTILE_CONT' in query
        assert 'tick_data' in query

    def test_liquidity_profile_query(self):
        """Test liquidity profile query generation."""
        query = AnalyticsQueries.liquidity_profile('tick_data')

        assert 'hour' in query
        assert 'liquidity' in query
        assert 'GROUP BY hour' in query

    def test_trade_intensity_query(self):
        """Test trade intensity query generation."""
        query = AnalyticsQueries.trade_intensity('trades', window_minutes=5)

        assert '5 minutes' in query
        assert 'TIME_BUCKET' in query
        assert 'vwap' in query

    def test_volatility_estimate_query(self):
        """Test volatility estimate query generation."""
        query = AnalyticsQueries.volatility_estimate('tick_data', lookback_hours=24)

        assert 'log_return' in query
        assert 'STDDEV' in query
        assert 'annualized_vol' in query


class TestDuckDBContextManager:
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test context manager properly closes connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with DuckDBCache(str(db_path)) as cache:
                cache.query_scalar("SELECT 1")
                # Connection should be active here

            # After exit, connection should be closed
            assert db_path.exists()


class TestPriceStats:
    """Test price statistics functionality."""

    def test_get_price_stats(self, duckdb_cache, sample_tick_data):
        """Test getting price statistics."""
        duckdb_cache.load_dataframe(sample_tick_data, 'ticks')
        stats = duckdb_cache.get_price_stats('ticks')

        assert 'count' in stats
        assert 'min_bid' in stats
        assert 'max_ask' in stats
        assert stats['count'] == 100

    def test_get_price_stats_with_time_range(self, duckdb_cache, sample_tick_data):
        """Test price stats with time filtering."""
        duckdb_cache.load_dataframe(sample_tick_data, 'ticks')
        start = datetime(2024, 1, 1, 0, 30)
        end = datetime(2024, 1, 1, 0, 45)
        stats = duckdb_cache.get_price_stats('ticks', start=start, end=end)

        assert stats['count'] == 16  # 30-45 inclusive


class TestResampleOHLCV:
    """Test OHLCV resampling."""

    def test_resample_ohlcv(self, duckdb_cache):
        """Test OHLCV resampling."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=60, freq='min'),
            'mid': [100 + i * 0.1 for i in range(60)],
        })
        duckdb_cache.load_dataframe(df, 'ticks')

        ohlcv = duckdb_cache.resample_ohlcv('ticks', timeframe='5min')

        assert len(ohlcv) == 12  # 60 minutes / 5 = 12 bars
        assert 'open' in ohlcv.columns
        assert 'high' in ohlcv.columns
        assert 'low' in ohlcv.columns
        assert 'close' in ohlcv.columns


class TestResampleWithTimeRange:
    """Test resampling with time range filters."""

    def test_resample_with_start_end(self, duckdb_cache):
        """Test resampling with start and end filters."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 00:00', periods=120, freq='min'),
            'mid': [100 + i * 0.1 for i in range(120)],
        })
        duckdb_cache.load_dataframe(df, 'ticks')

        start = datetime(2024, 1, 1, 0, 30)
        end = datetime(2024, 1, 1, 0, 45)
        ohlcv = duckdb_cache.resample_ohlcv('ticks', timeframe='1min', start=start, end=end)

        assert len(ohlcv) == 16  # 30-45 inclusive
