"""
Tests for data cache module.
"""
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.cache import DataCache, DataManager


class TestDataCache:
    """Test DataCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory using pytest's tmp_path fixture."""
        return str(tmp_path / "cache")

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache directory creation."""
        cache = DataCache(base_dir=temp_cache_dir)

        assert cache.base_dir.exists()
        assert cache.raw_dir.exists()
        assert cache.processed_dir.exists()

    def test_default_cache_path_stays_within_project(self):
        """Default cache location should be rooted under the current project."""
        cache = DataCache()
        project_root = Path(__file__).resolve().parent.parent
        assert str(cache.base_dir).startswith(str(project_root))

    def test_put_and_get_raw(self, temp_cache_dir):
        """Test storing and retrieving raw data."""
        cache = DataCache(base_dir=temp_cache_dir)

        # Create sample data for single date
        date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            'timestamp': pd.date_range(date, periods=24, freq='h'),
            'price': np.random.uniform(50000, 51000, 24)
        })

        # Store data
        cache.put(df, 'deribit', 'tick', 'BTC-PERPETUAL', date)

        # Retrieve data
        start = date
        end = date + timedelta(hours=23)
        retrieved = cache.get('deribit', 'tick', 'BTC-PERPETUAL', start, end)

        assert retrieved is not None
        assert len(retrieved) == len(df)
        assert list(retrieved.columns) == list(df.columns)

    def test_get_missing_data(self, temp_cache_dir):
        """Test retrieving non-existent data."""
        cache = DataCache(base_dir=temp_cache_dir)

        result = cache.get(
            'deribit', 'tick', 'NONEXISTENT',
            datetime(2024, 1, 1),
            datetime(2024, 1, 2)
        )

        assert result is None

    def test_cache_exists(self, temp_cache_dir):
        """Test cache existence check."""
        cache = DataCache(base_dir=temp_cache_dir)

        # Initially should not exist
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1, 23)
        assert not cache.exists('deribit', 'tick', 'BTC-PERPETUAL', start, end)

        # Store data
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, i) for i in range(5)],
            'price': [50000] * 5
        })
        cache.put(df, 'deribit', 'tick', 'BTC-PERPETUAL', start)

        # Should now exist
        assert cache.exists('deribit', 'tick', 'BTC-PERPETUAL', start, end)

    def test_get_date_range(self, temp_cache_dir):
        """Test retrieving data across multiple dates."""
        cache = DataCache(base_dir=temp_cache_dir)

        # Store data for multiple days using put_range
        dfs = []
        for day in range(3):
            date = datetime(2024, 1, 1 + day)
            df = pd.DataFrame({
                'timestamp': pd.date_range(date, periods=24, freq='h'),
                'price': np.random.uniform(50000, 51000, 24)
            })
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        cache.put_range(combined, 'deribit', 'tick', 'BTC-PERPETUAL')

        # Retrieve across all days
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3, 23)
        retrieved = cache.get('deribit', 'tick', 'BTC-PERPETUAL', start, end)

        assert retrieved is not None
        assert len(retrieved) == 72  # 3 days * 24 hours

    def test_instrument_name_sanitization(self, temp_cache_dir):
        """Test that special characters in instrument names are sanitized."""
        cache = DataCache(base_dir=temp_cache_dir)

        # Instrument with special characters
        date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 12)],
            'price': [50000]
        })

        # Should not raise error
        cache.put(df, 'deribit', 'tick', 'BTC/USD-PERP', date)
        retrieved = cache.get('deribit', 'tick', 'BTC/USD-PERP', date, date)

        assert retrieved is not None


class TestDataCacheEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_empty_dataframe(self, temp_cache_dir):
        """Test handling of empty DataFrame."""
        cache = DataCache(base_dir=temp_cache_dir)

        empty_df = pd.DataFrame({'timestamp': [], 'price': []})
        date = datetime(2024, 1, 1)

        # Should handle gracefully
        cache.put(empty_df, 'deribit', 'tick', 'BTC-PERPETUAL', date)

    def test_single_row(self, temp_cache_dir):
        """Test caching single row."""
        cache = DataCache(base_dir=temp_cache_dir)

        date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 12)],
            'price': [50000]
        })

        cache.put(df, 'deribit', 'tick', 'BTC-PERPETUAL', date)
        retrieved = cache.get('deribit', 'tick', 'BTC-PERPETUAL', date, date)

        assert retrieved is not None
        assert len(retrieved) == 1


class TestDataManager:
    """Tests for DataManager behavior and fallback logic."""

    @pytest.mark.asyncio
    async def test_fallback_to_downloader_when_cache_read_returns_none(self):
        """If cache metadata says exists but read fails, manager should download."""
        cache = MagicMock()
        cache.exists.return_value = True
        cache.get.return_value = None

        downloaded = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "price": [50000.0, 50010.0, 50020.0],
            }
        )

        async def downloader(instrument, start, end):
            return downloaded

        manager = DataManager(cache=cache)
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 2, 0, 0)

        result = await manager.get_data(
            exchange="deribit",
            data_type="tick",
            instrument="BTC-PERPETUAL",
            start=start,
            end=end,
            downloader=downloader,
            use_cache=True,
        )

        assert result is not None
        assert len(result) == 3
        cache.put_range.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_raises_on_invalid_time_range(self):
        """Invalid time range should raise instead of silently misbehaving."""
        manager = DataManager(DataCache())

        start = datetime(2024, 1, 2, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 0, 0)

        async def downloader(instrument, start, end):
            return pd.DataFrame()

        with pytest.raises(ValueError, match="start must be <= end"):
            await manager.get_data(
                exchange="deribit",
                data_type="tick",
                instrument="BTC-PERPETUAL",
                start=start,
                end=end,
                downloader=downloader,
                use_cache=True,
            )
