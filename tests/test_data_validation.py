"""
Tests for data validation module.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from data.validation import DataCleaner, DataValidator, ValidationResult


class TestDataValidator:
    """Test DataValidator class."""

    def test_valid_ohlcv(self):
        """Test validation of valid OHLCV data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'open': np.random.uniform(50000, 51000, 10),
            'high': np.random.uniform(51000, 52000, 10),
            'low': np.random.uniform(49000, 50000, 10),
            'close': np.random.uniform(50000, 51000, 10),
            'volume': np.random.uniform(1, 100, 10)
        })
        # Ensure OHLC logic
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert result.is_valid
        assert len(result.issues) == 0
        assert result.stats is not None
        assert result.stats['total_rows'] == 10

    def test_missing_columns(self):
        """Test detection of missing columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'close': [50000] * 5
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("Missing columns" in issue for issue in result.issues)

    def test_duplicate_timestamps(self):
        """Test detection and removal of duplicate timestamps."""
        timestamps = [datetime(2024, 1, 1, 12)] * 3 + [datetime(2024, 1, 1, 13)] * 2
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [50000] * 5,
            'high': [51000] * 5,
            'low': [49000] * 5,
            'close': [50500] * 5,
            'volume': [10] * 5
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        # Should have issues but still be valid after cleaning
        assert any("duplicate" in issue.lower() for issue in result.issues)
        assert result.stats['total_rows'] == 2  # Duplicates removed

    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'open': [50000, -100, 50000, 50000, 50000],
            'high': [51000, 51000, 51000, 51000, 51000],
            'low': [49000, -200, 49000, 49000, 49000],
            'close': [50500, -150, 50500, 50500, 50500],
            'volume': [10, 10, 10, 10, 10]
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("negative" in issue.lower() for issue in result.issues)

    def test_missing_values(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'open': [50000, np.nan, 50000, 50000, 50000],
            'high': [51000] * 5,
            'low': [49000] * 5,
            'close': [50500] * 5,
            'volume': [10, 0, 10, 10, 10]  # Zero volume
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        # Should have issues for null values
        assert len(result.issues) > 0
        # Stats should show reduced rows after cleaning
        assert result.stats['total_rows'] < 5

    def test_ohlc_logic_violation(self):
        """Test detection of OHLC logic violations."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'open': [50000] * 5,
            'high': [48000] * 5,  # High < Open (violation)
            'low': [49000] * 5,
            'close': [50500] * 5,
            'volume': [10] * 5
        })

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("invalid" in issue.lower() for issue in result.issues)


class TestDataCleaner:
    """Test DataCleaner class."""

    def test_cleaner_initialization(self):
        """Test DataCleaner creation."""
        cleaner = DataCleaner()
        assert cleaner is not None

    def test_cleaner_remove_outliers(self):
        """Test outlier removal."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='h'),
            'price': [50000] * 19 + [100000]  # One outlier
        })

        cleaner = DataCleaner()
        cleaned = cleaner.remove_outliers(df, column='price', method='zscore', threshold=3)

        assert len(cleaned) <= len(df)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creation of ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            issues=[],
            cleaned_data=None
        )

        assert result.is_valid
        assert result.issues == []

    def test_validation_result_with_data(self):
        """Test ValidationResult with stats."""
        result = ValidationResult(
            is_valid=False,
            issues=["Test issue"],
            stats={'rows_processed': 3}
        )

        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.stats['rows_processed'] == 3
