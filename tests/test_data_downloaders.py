"""
Tests for data downloaders.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data.cache import DataCache
from data.downloaders.deribit import DeribitClient


class TestDeribitClient:
    """Test Deribit API client."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection(self):
        """Test basic connection."""
        client = DeribitClient()
        await client.connect()
        assert client._session is not None
        await client.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_instruments(self):
        """Test fetching instruments."""
        client = DeribitClient()
        async with client:
            instruments = await client.get_instruments(currency="BTC", instrument_type="option")
            assert len(instruments) > 0
            assert all(inst.underlying == "BTC" for inst in instruments)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_order_book(self):
        """Test fetching order book."""
        client = DeribitClient()
        async with client:
            # Use a perpetual contract
            ob = await client.get_order_book("BTC-PERPETUAL", depth=5)
            assert ob.best_bid > 0
            assert ob.best_ask > ob.best_bid
            assert len(ob.bids) <= 5
            assert len(ob.asks) <= 5


class TestDataCache:
    """Test data caching functionality."""

    def test_cache_creation(self, tmp_path):
        """Test cache initialization."""
        cache = DataCache(base_dir=str(tmp_path))
        assert cache.base_dir.exists()
        assert cache.raw_dir.exists()
        assert cache.processed_dir.exists()

    def test_put_and_get(self, tmp_path):
        """Test storing and retrieving data."""
        cache = DataCache(base_dir=str(tmp_path))

        # Create sample data (24 hours - single day)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'price': range(24)
        })

        date = datetime(2024, 1, 1)

        # Store data
        cache.put(df, "test_exchange", "test_type", "TEST", date)

        # Retrieve data
        retrieved = cache.get("test_exchange", "test_type", "TEST", date, date)

        assert retrieved is not None
        assert len(retrieved) == len(df)

    def test_exists_check(self, tmp_path):
        """Test cache existence check."""
        cache = DataCache(base_dir=str(tmp_path))

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'price': range(24)
        })

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1)

        # Before storing
        assert not cache.exists("test", "type", "SYM", start, end)

        # Store
        cache.put(df, "test", "type", "SYM", start)

        # After storing
        assert cache.exists("test", "type", "SYM", start, end)


class TestDataValidation:
    """Test data validation."""

    def test_valid_ohlcv(self):
        """Test validation of valid OHLCV data."""
        from data.validation import DataValidator

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'open': [100 + i for i in range(100)],
            'high': [102 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [101 + i for i in range(100)],
            'volume': [1000] * 100
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert result.is_valid
        assert len(result.issues) == 0

    def test_invalid_ohlcv_logic(self):
        """Test detection of invalid OHLC relationships."""
        from data.validation import DataValidator

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [110] * 10,  # Low > High - invalid!
            'close': [103] * 10,
            'volume': [1000] * 10
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any('invalid OHLC' in issue for issue in result.issues)

    def test_negative_prices(self):
        """Test detection of negative prices."""
        from data.validation import DataValidator

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'open': [100] * 9 + [-10],  # Negative price
            'high': [105] * 9 + [0],
            'low': [99] * 9 + [-20],
            'close': [103] * 9 + [-5],
            'volume': [1000] * 10
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any('negative' in issue.lower() for issue in result.issues)
