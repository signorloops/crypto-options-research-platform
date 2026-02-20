"""
Data validation and cleaning utilities for market data.
Ensures data quality before backtesting or analysis.
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    cleaned_data: Optional[pd.DataFrame] = None
    stats: Dict = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = {}


class DataValidator:
    """
    Validates market data for quality issues.
    """

    def __init__(self):
        self.issues: List[str] = []

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> ValidationResult:
        """
        Validate OHLCV data.

        Checks for:
        - Required columns
        - Missing values
        - OHLC logic (low <= open, close <= high)
        - Negative prices
        - Zero volumes
        - Duplicate timestamps
        - Out-of-order timestamps
        """
        if required_columns is None:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        self.issues = []
        df = df.copy()

        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.issues.append(f"Missing columns: {missing_cols}")
            return ValidationResult(is_valid=False, issues=self.issues)

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                self.issues.append(f"Cannot parse timestamp: {e}")
                return ValidationResult(is_valid=False, issues=self.issues)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Check for duplicates
        dups = df['timestamp'].duplicated().sum()
        if dups > 0:
            self.issues.append(f"Found {dups} duplicate timestamps")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # Check for missing values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            self.issues.append(f"Null values found: {cols_with_nulls}")
            df = df.dropna(subset=required_columns)

        # Check OHLC logic
        invalid_ohlc = (
            (df['low'] > df['high']) |
            (df['open'] > df['high']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] < df['low'])
        )
        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            self.issues.append(f"{n_invalid} rows with invalid OHLC logic")
            df = df[~invalid_ohlc]

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            neg_mask = df[col] <= 0
            if neg_mask.any():
                n_neg = neg_mask.sum()
                self.issues.append(f"{n_neg} rows with negative/zero {col}")
                df = df[~neg_mask]

        # Check for zero volume
        zero_vol = df['volume'] <= 0
        if zero_vol.any():
            n_zero = zero_vol.sum()
            self.issues.append(f"{n_zero} rows with zero/negative volume")

        # Calculate stats
        stats = {
            'total_rows': len(df),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'price_range': {
                'min': df['low'].min(),
                'max': df['high'].max()
            },
            'avg_volume': df['volume'].mean(),
            'gaps_found': self._detect_gaps(df)
        }

        is_valid = len(df) > 0 and len(self.issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues,
            cleaned_data=df if len(self.issues) > 0 else None,
            stats=stats
        )

    def validate_trades(
        self,
        df: pd.DataFrame,
        max_price_gap: float = 0.5  # 50% max price change between trades
    ) -> ValidationResult:
        """
        Validate trade data.

        Checks for:
        - Required columns
        - Missing values
        - Negative prices/sizes
        - Price jumps
        - Duplicate trade IDs
        - Timestamp ordering
        """
        self.issues = []
        df = df.copy()

        required_columns = ['timestamp', 'price', 'size', 'side']

        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.issues.append(f"Missing columns: {missing_cols}")
            return ValidationResult(is_valid=False, issues=self.issues)

        # Parse timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Check for duplicates
        if 'trade_id' in df.columns:
            dups = df['trade_id'].duplicated().sum()
            if dups > 0:
                self.issues.append(f"Found {dups} duplicate trade IDs")
                df = df.drop_duplicates(subset=['trade_id'], keep='first')

        # Check for negative values
        neg_prices = df['price'] <= 0
        if neg_prices.any():
            n_neg = neg_prices.sum()
            self.issues.append(f"{n_neg} trades with negative/zero price")
            df = df[~neg_prices]

        neg_sizes = df['size'] <= 0
        if neg_sizes.any():
            n_neg = neg_sizes.sum()
            self.issues.append(f"{n_neg} trades with negative/zero size")
            df = df[~neg_sizes]

        # Check for price jumps
        price_changes = df['price'].pct_change().abs()
        large_jumps = price_changes > max_price_gap
        if large_jumps.any():
            n_jumps = large_jumps.sum()
            self.issues.append(f"{n_jumps} trades with >{max_price_gap:.0%} price jump")

        # Check side values
        valid_sides = ['buy', 'sell', 'BUY', 'SELL']
        invalid_sides = ~df['side'].isin(valid_sides)
        if invalid_sides.any():
            n_invalid = invalid_sides.sum()
            self.issues.append(f"{n_invalid} trades with invalid side")

        # Stats
        stats = {
            'total_trades': len(df),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'price_range': {
                'min': df['price'].min(),
                'max': df['price'].max()
            },
            'total_volume': df['size'].sum(),
            'buy_sell_ratio': (df['side'].str.lower() == 'buy').mean()
        }

        is_valid = len(df) > 0 and len(self.issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues,
            cleaned_data=df if len(self.issues) > 0 else None,
            stats=stats
        )

    def _detect_gaps(self, df: pd.DataFrame, max_gap_minutes: int = 5) -> int:
        """Detect timestamp gaps in data."""
        if len(df) < 2:
            return 0

        # Calculate time differences
        time_diffs = df['timestamp'].diff().dropna()
        expected_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else timedelta(minutes=1)

        # Find gaps larger than expected
        gaps = time_diffs > expected_diff * 2
        return gaps.sum()


class DataCleaner:
    """
    Clean and preprocess market data.
    """

    @staticmethod
    def resample_ohlcv(
        df: pd.DataFrame,
        target_interval: str = '1h',
        price_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a different time interval.

        Args:
            df: DataFrame with OHLCV data
            target_interval: Target resampling interval (e.g., '1h', '15m', '1d')
            price_cols: List of price columns to resample

        Returns:
            Resampled DataFrame
        """
        if price_cols is None:
            price_cols = ['open', 'high', 'low', 'close']

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Only include columns that exist
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        resampled = df.resample(target_interval).agg(agg_rules)
        resampled = resampled.dropna()
        resampled = resampled.reset_index()

        return resampled

    @staticmethod
    def fill_gaps(
        df: pd.DataFrame,
        method: str = 'forward_fill',
        max_gap: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """
        Fill gaps in time series data.

        Args:
            df: DataFrame with timestamp column
            method: 'forward_fill', 'interpolate', or 'remove'
            max_gap: Maximum gap size to fill

        Returns:
            DataFrame with gaps handled
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Resample to regular intervals to expose gaps
        if len(df) > 1:
            # Infer frequency
            freq = pd.infer_freq(df.index)
            if freq:
                df = df.resample(freq).asfreq()

        if method == 'forward_fill':
            df = df.ffill()
        elif method == 'interpolate':
            df = df.interpolate(method='time')
        elif method == 'remove':
            df = df.dropna()

        df = df.reset_index()
        return df

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from a column.

        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr', 'zscore', or 'mad'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (df[column] >= lower) & (df[column] <= upper)

        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            mask = z_scores < threshold

        elif method == 'mad':
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))
            modified_z = 0.6745 * (df[column] - median) / mad
            mask = np.abs(modified_z) < threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        return df[mask].reset_index(drop=True)

    @staticmethod
    def align_timestamps(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        method: str = 'nearest'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two DataFrames to common timestamps.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            timestamp_col: Name of timestamp column
            method: 'nearest', 'ffill', or 'interpolate'

        Returns:
            Tuple of aligned DataFrames
        """
        df1 = df1.copy()
        df2 = df2.copy()

        df1[timestamp_col] = pd.to_datetime(df1[timestamp_col])
        df2[timestamp_col] = pd.to_datetime(df2[timestamp_col])

        # Find common time range
        start = max(df1[timestamp_col].min(), df2[timestamp_col].min())
        end = min(df1[timestamp_col].max(), df2[timestamp_col].max())

        # Filter to common range
        df1 = df1[(df1[timestamp_col] >= start) & (df1[timestamp_col] <= end)]
        df2 = df2[(df2[timestamp_col] >= start) & (df2[timestamp_col] <= end)]

        # Merge on nearest timestamp
        df1.set_index(timestamp_col, inplace=True)
        df2.set_index(timestamp_col, inplace=True)

        # Resample to common frequency
        freq = pd.infer_freq(df1.index) or '1min'
        common_index = pd.date_range(start=start, end=end, freq=freq)

        df1 = df1.reindex(common_index, method=method if method != 'nearest' else None)
        df2 = df2.reindex(common_index, method=method if method != 'nearest' else None)

        df1 = df1.reset_index().rename(columns={'index': timestamp_col})
        df2 = df2.reset_index().rename(columns={'index': timestamp_col})

        return df1, df2


class DataQualityReport:
    """
    Generate comprehensive data quality reports.
    """

    def __init__(self):
        self.validator = DataValidator()

    def generate_report(self, df: pd.DataFrame, data_type: str = 'ohlcv') -> Dict:
        """
        Generate comprehensive quality report.

        Args:
            df: DataFrame to analyze
            data_type: 'ohlcv', 'trades', or 'orderbook'

        Returns:
            Dictionary with quality metrics
        """
        if data_type == 'ohlcv':
            validation = self.validator.validate_ohlcv(df)
        elif data_type == 'trades':
            validation = self.validator.validate_trades(df)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        report = {
            'is_valid': validation.is_valid,
            'issues_found': len(validation.issues),
            'issues': validation.issues,
            'statistics': validation.stats,
            'recommendations': self._generate_recommendations(validation)
        }

        return report

    def _generate_recommendations(self, validation: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        for issue in validation.issues:
            if 'duplicate' in issue.lower():
                recommendations.append("Consider removing duplicate timestamps before analysis")
            elif 'negative' in issue.lower():
                recommendations.append("Review data source for price/size anomalies")
            elif 'gap' in issue.lower():
                recommendations.append("Consider interpolation for missing periods")
            elif 'outlier' in issue.lower() or 'jump' in issue.lower():
                recommendations.append("Review extreme price movements for data errors")

        if not validation.is_valid:
            recommendations.append("Data requires cleaning before use in backtesting")

        return recommendations
