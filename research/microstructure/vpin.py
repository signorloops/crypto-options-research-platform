"""
VPIN (Volume-Synchronized Probability of Informed Trading) calculator.
VPIN is a measure of order flow toxicity based on volume buckets rather than time.
Reference: Easley, Lopez de Prado, and O'Hara (2012)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VPINResult:
    """Result of VPIN calculation."""
    timestamps: np.ndarray
    vpin_values: np.ndarray
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray
    volume_buckets: np.ndarray

    def get_high_toxicity_periods(self, threshold: float = 0.4) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Identify periods of high toxicity."""
        high_toxicity = self.vpin_values > threshold
        periods = []
        in_period = False
        start = None

        for i, is_high in enumerate(high_toxicity):
            if is_high and not in_period:
                start = self.timestamps[i]
                in_period = True
            elif not is_high and in_period:
                periods.append((start, self.timestamps[i]))
                in_period = False

        if in_period:
            periods.append((start, self.timestamps[-1]))

        return periods


class VPINCalculator:
    """
    Calculate VPIN metric for detecting toxic order flow.

    VPIN measures the probability that informed traders are present in the market.
    High VPIN indicates:
    - Increased adverse selection risk for market makers
    - Higher probability of information-based trading
    - Potential need to widen spreads or reduce quoting
    """

    def __init__(self, volume_bucket_size: float = 100.0, num_buckets: int = 50):
        """
        Args:
            volume_bucket_size: Target volume per bucket (in base currency)
            num_buckets: Number of buckets for rolling VPIN calculation
        """
        self.volume_bucket_size = volume_bucket_size
        self.num_buckets = num_buckets

    def calculate(self, trades_df: pd.DataFrame) -> VPINResult:
        """
        Calculate VPIN from trade data.

        Args:
            trades_df: DataFrame with columns:
                - timestamp: datetime
                - price: float
                - size: float (base currency volume)
                - side: 'buy' or 'sell' (optional, will estimate if missing)

        Returns:
            VPINResult with timestamps and VPIN values
        """
        # Validate required columns
        required_cols = ['timestamp', 'price', 'size']
        missing = set(required_cols) - set(trades_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = trades_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Estimate trade direction if not provided (using tick rule)
        if 'side' not in df.columns or df['side'].isna().all():
            df['side'] = self._estimate_trade_direction(df)

        # Create volume buckets (vectorized using cumsum/searchsorted)
        bucket_timestamps, buy_arr, sell_arr, total_arr = self._create_volume_buckets(df)

        if len(total_arr) < self.num_buckets:
            # Not enough buckets, return neutral values
            volume = df['size'] * df['price']
            return VPINResult(
                timestamps=df['timestamp'].values,
                vpin_values=np.full(len(df), 0.5),
                buy_volumes=volume.where(df['side'] == 'buy', 0).values,
                sell_volumes=volume.where(df['side'] == 'sell', 0).values,
                volume_buckets=np.zeros(len(df))
            )

        # Vectorized VPIN computation using numpy arrays and cumsum
        n = len(total_arr)
        imbalance_arr = np.abs(buy_arr - sell_arr)

        # Rolling sums via cumsum difference (O(n) instead of O(n*w))
        cum_imbalance = np.concatenate(([0], np.cumsum(imbalance_arr)))
        cum_total = np.concatenate(([0], np.cumsum(total_arr)))

        w = self.num_buckets
        half_w = max(1, w // 2)
        vpin_values = np.full(n, 0.5)
        idx = np.arange(n)
        starts = np.maximum(0, idx - w + 1)
        window_lengths = idx - starts + 1
        valid = window_lengths >= half_w
        window_total = cum_total[idx + 1] - cum_total[starts]
        window_imbalance = cum_imbalance[idx + 1] - cum_imbalance[starts]
        safe_total = np.where(window_total > 0, window_total, 1.0)
        computed = np.where(window_total > 0, window_imbalance / (2.0 * safe_total), 0.0)
        vpin_values[valid] = computed[valid]

        return VPINResult(
            timestamps=bucket_timestamps,
            vpin_values=vpin_values,
            buy_volumes=buy_arr,
            sell_volumes=sell_arr,
            volume_buckets=np.arange(n)
        )

    def _estimate_trade_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate trade direction using Lee-Ready tick test.

        If trade price > last trade price: buy
        If trade price < last trade price: sell
        If equal, use previous direction (recursive)
        """
        price_diff = df['price'].diff()

        sides = pd.Series(np.nan, index=df.index, dtype='object')
        sides[price_diff > 0] = 'buy'
        sides[price_diff < 0] = 'sell'
        # Forward-fill stable-price trades; default first side to buy.
        return sides.ffill().fillna('buy')

    def _create_volume_buckets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create volume-synchronized buckets via cumsum/searchsorted."""
        if df.empty:
            empty = np.array([])
            return empty, empty, empty, empty

        prices = df['price'].to_numpy(dtype=float)
        sizes = df['size'].to_numpy(dtype=float)
        sides = df['side'].to_numpy()
        timestamps = df['timestamp'].to_numpy()
        volumes = np.maximum(sizes * prices, 0.0)  # Volume in quote currency
        if len(volumes) == 0:
            empty = np.array([])
            return empty, empty, empty, empty

        cum_total = np.cumsum(volumes)
        total_volume = float(cum_total[-1]) if len(cum_total) else 0.0
        if total_volume <= 0:
            empty = np.array([])
            return empty, empty, empty, empty

        bucket_size = float(self.volume_bucket_size)
        n_full = int(total_volume // bucket_size)
        residual = total_volume - n_full * bucket_size
        include_partial = residual >= bucket_size * 0.5
        n_buckets = n_full + (1 if include_partial else 0)
        if n_buckets == 0:
            empty = np.array([])
            return empty, empty, empty, empty

        is_buy = sides == 'buy'
        buy_trade_vol = np.where(is_buy, volumes, 0.0)
        sell_trade_vol = np.where(is_buy, 0.0, volumes)
        cum_buy = np.cumsum(buy_trade_vol)
        cum_sell = np.cumsum(sell_trade_vol)

        full_buy = np.array([], dtype=float)
        full_sell = np.array([], dtype=float)
        full_timestamps = np.array([], dtype=timestamps.dtype)
        full_totals = np.array([], dtype=float)
        cum_buy_at_boundaries = np.array([], dtype=float)
        cum_sell_at_boundaries = np.array([], dtype=float)

        if n_full > 0:
            boundaries = (np.arange(1, n_full + 1, dtype=float) * bucket_size)
            boundary_idx = np.searchsorted(cum_total, boundaries, side='left')
            boundary_idx = np.clip(boundary_idx, 0, len(cum_total) - 1)

            trade_starts = cum_total[boundary_idx] - volumes[boundary_idx]
            partial_to_boundary = boundaries - trade_starts

            buy_prefix = np.where(boundary_idx > 0, cum_buy[boundary_idx - 1], 0.0)
            sell_prefix = np.where(boundary_idx > 0, cum_sell[boundary_idx - 1], 0.0)
            buy_partial = np.where(is_buy[boundary_idx], partial_to_boundary, 0.0)
            sell_partial = np.where(is_buy[boundary_idx], 0.0, partial_to_boundary)

            cum_buy_at_boundaries = buy_prefix + buy_partial
            cum_sell_at_boundaries = sell_prefix + sell_partial

            full_buy = np.diff(np.concatenate(([0.0], cum_buy_at_boundaries)))
            full_sell = np.diff(np.concatenate(([0.0], cum_sell_at_boundaries)))
            full_totals = np.full(n_full, bucket_size, dtype=float)
            full_timestamps = timestamps[boundary_idx]

        if include_partial:
            prev_buy = float(cum_buy_at_boundaries[-1]) if n_full > 0 else 0.0
            prev_sell = float(cum_sell_at_boundaries[-1]) if n_full > 0 else 0.0
            partial_buy = float(cum_buy[-1] - prev_buy)
            partial_sell = float(cum_sell[-1] - prev_sell)

            bucket_timestamps = np.concatenate((full_timestamps, np.array([timestamps[-1]])))
            buy_buckets = np.concatenate((full_buy, np.array([partial_buy])))
            sell_buckets = np.concatenate((full_sell, np.array([partial_sell])))
            total_buckets = np.concatenate((full_totals, np.array([residual])))
        else:
            bucket_timestamps = full_timestamps
            buy_buckets = full_buy
            sell_buckets = full_sell
            total_buckets = full_totals

        return bucket_timestamps, buy_buckets, sell_buckets, total_buckets


class OrderFlowToxicityAnalyzer:
    """
    Comprehensive order flow toxicity analysis.
    """

    def __init__(self):
        self.vpin_calc = VPINCalculator()

    def analyze(self, trades_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform full toxicity analysis.

        Returns DataFrame with:
        - vpin: VPIN metric
        - trade_intensity: Trades per minute
        - volume_imbalance: Buy/Sell ratio
        - large_trade_ratio: Proportion of large trades
        - price_impact: Correlation between trade direction and subsequent returns
        """
        df = trades_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate VPIN
        vpin_result = self.vpin_calc.calculate(df)

        # Resample to 1-minute windows for analysis
        df['minute'] = df['timestamp'].dt.floor('1min')
        grouped = df.groupby('minute')

        # Precompute threshold once to avoid repeated O(n) quantile calls.
        large_threshold = float(df['size'].quantile(0.95))

        # Convert timestamps to int64 nanoseconds for fast nearest lookup.
        vpin_ts = np.array(vpin_result.timestamps, dtype='datetime64[ns]')
        vpin_vals = np.array(vpin_result.vpin_values, dtype=float)
        if len(vpin_ts) == 0:
            return pd.DataFrame()
        vpin_ts_i8 = vpin_ts.astype(np.int64)

        results = []
        for minute, group in grouped:
            size_series = group['size']
            total_volume = float(size_series.sum())
            buy_volume = float(size_series[group['side'] == 'buy'].sum())
            sell_volume = float(size_series[group['side'] == 'sell'].sum())

            minute_i8 = np.datetime64(minute, 'ns').astype(np.int64)
            insert_pos = int(np.searchsorted(vpin_ts_i8, minute_i8))
            if insert_pos == 0:
                vpin_idx = 0
            elif insert_pos >= len(vpin_ts_i8):
                vpin_idx = len(vpin_ts_i8) - 1
            else:
                left = insert_pos - 1
                right = insert_pos
                vpin_idx = left if abs(vpin_ts_i8[left] - minute_i8) <= abs(vpin_ts_i8[right] - minute_i8) else right
            vpin_val = float(vpin_vals[vpin_idx])

            # Large trades (>95th percentile)
            large_trades = int((size_series > large_threshold).sum())
            trade_count = len(group)
            volume_imbalance = (buy_volume - sell_volume) / (total_volume + 1e-8)

            result = {
                'timestamp': minute,
                'vpin': vpin_val,
                'trade_count': trade_count,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'volume_imbalance': volume_imbalance,
                'large_trade_count': large_trades,
                'large_trade_ratio': large_trades / trade_count if trade_count > 0 else 0,
                'avg_trade_size': float(size_series.mean()),
                'price_volatility': float(group['price'].std()) if trade_count > 1 else 0
            }

            results.append(result)

        return pd.DataFrame(results)

    def detect_anomalies(self, analysis_df: pd.DataFrame, zscore_threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalous periods in the analysis."""
        df = analysis_df.copy()

        # Calculate z-scores for key metrics
        for col in ['vpin', 'trade_count', 'volume_imbalance', 'large_trade_ratio']:
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-8)

        # Flag anomalies
        df['is_anomaly'] = (
            (df['vpin_zscore'] > zscore_threshold) |
            (df['trade_count_zscore'] > zscore_threshold) |
            (abs(df['volume_imbalance_zscore']) > zscore_threshold)
        )

        return df
