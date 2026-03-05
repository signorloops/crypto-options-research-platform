"""VPIN (volume-synchronized probability of informed trading) calculator."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _empty_bucket_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.array([])
    return empty, empty, empty, empty


def _full_bucket_components(
    *,
    cum_total: np.ndarray,
    volumes: np.ndarray,
    is_buy: np.ndarray,
    cum_buy: np.ndarray,
    cum_sell: np.ndarray,
    timestamps: np.ndarray,
    bucket_size: float,
    n_full: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_full <= 0:
        empty_float = np.array([], dtype=float)
        empty_ts = np.array([], dtype=timestamps.dtype)
        return empty_float, empty_float, empty_ts, empty_float, empty_float, empty_float

    boundaries = (np.arange(1, n_full + 1, dtype=float) * bucket_size)
    boundary_idx = np.searchsorted(cum_total, boundaries, side="left")
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
    return (
        full_buy,
        full_sell,
        full_timestamps,
        full_totals,
        cum_buy_at_boundaries,
        cum_sell_at_boundaries,
    )


def _append_partial_bucket(
    *,
    include_partial: bool,
    n_full: int,
    timestamps: np.ndarray,
    cum_buy: np.ndarray,
    cum_sell: np.ndarray,
    cum_buy_at_boundaries: np.ndarray,
    cum_sell_at_boundaries: np.ndarray,
    full_timestamps: np.ndarray,
    full_buy: np.ndarray,
    full_sell: np.ndarray,
    full_totals: np.ndarray,
    residual: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not include_partial:
        return full_timestamps, full_buy, full_sell, full_totals

    prev_buy = float(cum_buy_at_boundaries[-1]) if n_full > 0 else 0.0
    prev_sell = float(cum_sell_at_boundaries[-1]) if n_full > 0 else 0.0
    partial_buy = float(cum_buy[-1] - prev_buy)
    partial_sell = float(cum_sell[-1] - prev_sell)
    bucket_timestamps = np.concatenate((full_timestamps, np.array([timestamps[-1]])))
    buy_buckets = np.concatenate((full_buy, np.array([partial_buy])))
    sell_buckets = np.concatenate((full_sell, np.array([partial_sell])))
    total_buckets = np.concatenate((full_totals, np.array([residual])))
    return bucket_timestamps, buy_buckets, sell_buckets, total_buckets


def _volume_inputs_from_df(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None:
    if df.empty:
        return None
    prices = df["price"].to_numpy(dtype=float)
    sizes = df["size"].to_numpy(dtype=float)
    sides = df["side"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    volumes = np.maximum(sizes * prices, 0.0)
    if len(volumes) == 0:
        return None
    cum_total = np.cumsum(volumes)
    total_volume = float(cum_total[-1]) if len(cum_total) else 0.0
    if total_volume <= 0:
        return None
    return volumes, cum_total, total_volume, sides, timestamps


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


def _fallback_vpin_result(df: pd.DataFrame) -> VPINResult:
    volume = np.maximum(df["size"].to_numpy(dtype=float) * df["price"].to_numpy(dtype=float), 0.0)
    is_buy = df["side"].to_numpy() == "buy"
    buy_vol = np.where(is_buy, volume, 0.0)
    sell_vol = np.where(is_buy, 0.0, volume)
    total_vol = float(np.sum(volume))
    fallback_vpin = 0.0
    if total_vol > 0:
        flow_imbalance = np.abs(np.sum(buy_vol) - np.sum(sell_vol))
        fallback_vpin = float(np.clip(flow_imbalance / total_vol, 0.0, 1.0))
    return VPINResult(
        timestamps=df["timestamp"].values,
        vpin_values=np.full(len(df), fallback_vpin),
        buy_volumes=buy_vol,
        sell_volumes=sell_vol,
        volume_buckets=np.zeros(len(df)),
    )


def _rolling_vpin_values(
    *,
    buy_arr: np.ndarray,
    sell_arr: np.ndarray,
    total_arr: np.ndarray,
    num_buckets: int,
) -> np.ndarray:
    n = len(total_arr)
    imbalance_arr = np.abs(buy_arr - sell_arr)
    cum_imbalance = np.concatenate(([0], np.cumsum(imbalance_arr)))
    cum_total = np.concatenate(([0], np.cumsum(total_arr)))

    w = num_buckets
    half_w = max(1, w // 2)
    vpin_values = np.full(n, 0.5)
    idx = np.arange(n)
    starts = np.maximum(0, idx - w + 1)
    window_lengths = idx - starts + 1
    valid = window_lengths >= half_w
    window_total = cum_total[idx + 1] - cum_total[starts]
    window_imbalance = cum_imbalance[idx + 1] - cum_imbalance[starts]
    safe_total = np.where(window_total > 0, window_total, 1.0)
    computed = np.where(window_total > 0, window_imbalance / safe_total, 0.0)
    computed = np.clip(computed, 0.0, 1.0)
    vpin_values[valid] = computed[valid]
    return vpin_values


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
        """Calculate VPIN from trade data."""
        required_cols = ['timestamp', 'price', 'size']
        missing = set(required_cols) - set(trades_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = trades_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if 'side' not in df.columns or df['side'].isna().all():
            df['side'] = self._estimate_trade_direction(df)
        bucket_timestamps, buy_arr, sell_arr, total_arr = self._create_volume_buckets(df)
        if len(total_arr) < self.num_buckets:
            return _fallback_vpin_result(df)
        n = len(total_arr)
        vpin_values = _rolling_vpin_values(
            buy_arr=buy_arr,
            sell_arr=sell_arr,
            total_arr=total_arr,
            num_buckets=self.num_buckets,
        )
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
        inputs = _volume_inputs_from_df(df)
        if inputs is None:
            return _empty_bucket_arrays()
        volumes, cum_total, total_volume, sides, timestamps = inputs
        bucket_size = float(self.volume_bucket_size)
        n_full = int(total_volume // bucket_size); residual = total_volume - n_full * bucket_size
        n_buckets = n_full + (1 if (include_partial := residual >= bucket_size * 0.5) else 0)
        if n_buckets == 0:
            return _empty_bucket_arrays()
        is_buy = sides == 'buy'
        buy_trade_vol = np.where(is_buy, volumes, 0.0)
        sell_trade_vol = np.where(is_buy, 0.0, volumes)
        cum_buy = np.cumsum(buy_trade_vol)
        cum_sell = np.cumsum(sell_trade_vol)
        full_components = _full_bucket_components(
            cum_total=cum_total,
            volumes=volumes,
            is_buy=is_buy,
            cum_buy=cum_buy,
            cum_sell=cum_sell,
            timestamps=timestamps,
            bucket_size=bucket_size,
            n_full=n_full,
        )
        full_buy, full_sell, full_timestamps, full_totals, cum_buy_at_boundaries, cum_sell_at_boundaries = full_components
        return _append_partial_bucket(
            include_partial=include_partial,
            n_full=n_full,
            timestamps=timestamps,
            cum_buy=cum_buy,
            cum_sell=cum_sell,
            cum_buy_at_boundaries=cum_buy_at_boundaries,
            cum_sell_at_boundaries=cum_sell_at_boundaries,
            full_timestamps=full_timestamps,
            full_buy=full_buy,
            full_sell=full_sell,
            full_totals=full_totals,
            residual=residual,
        )


def _nearest_vpin_value(
    *,
    minute: pd.Timestamp,
    vpin_ts_i8: np.ndarray,
    vpin_vals: np.ndarray,
) -> float:
    minute_i8 = np.datetime64(minute, "ns").astype(np.int64)
    insert_pos = int(np.searchsorted(vpin_ts_i8, minute_i8))
    if insert_pos == 0:
        return float(vpin_vals[0])
    if insert_pos >= len(vpin_ts_i8):
        return float(vpin_vals[-1])
    left = insert_pos - 1
    right = insert_pos
    idx = left if abs(vpin_ts_i8[left] - minute_i8) <= abs(vpin_ts_i8[right] - minute_i8) else right
    return float(vpin_vals[idx])


def _minute_toxicity_row(
    *,
    minute: pd.Timestamp,
    group: pd.DataFrame,
    large_threshold: float,
    vpin_ts_i8: np.ndarray,
    vpin_vals: np.ndarray,
) -> dict:
    size_series = group["size"]
    total_volume = float(size_series.sum())
    buy_volume = float(size_series[group["side"] == "buy"].sum())
    sell_volume = float(size_series[group["side"] == "sell"].sum())
    large_trades = int((size_series > large_threshold).sum())
    trade_count = len(group)
    return {
        "timestamp": minute,
        "vpin": _nearest_vpin_value(minute=minute, vpin_ts_i8=vpin_ts_i8, vpin_vals=vpin_vals),
        "trade_count": trade_count,
        "total_volume": total_volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "volume_imbalance": (buy_volume - sell_volume) / (total_volume + 1e-8),
        "large_trade_count": large_trades,
        "large_trade_ratio": large_trades / trade_count if trade_count > 0 else 0,
        "avg_trade_size": float(size_series.mean()),
        "price_volatility": float(group["price"].std()) if trade_count > 1 else 0,
    }


class OrderFlowToxicityAnalyzer:
    """
    Comprehensive order flow toxicity analysis.
    """

    def __init__(self):
        self.vpin_calc = VPINCalculator()

    def analyze(self, trades_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Perform full toxicity analysis."""
        df = trades_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        vpin_result = self.vpin_calc.calculate(df)
        df['minute'] = df['timestamp'].dt.floor('1min')
        grouped = df.groupby('minute')
        large_threshold = float(df['size'].quantile(0.95))
        vpin_ts = np.array(vpin_result.timestamps, dtype='datetime64[ns]')
        vpin_vals = np.array(vpin_result.vpin_values, dtype=float)
        if len(vpin_ts) == 0:
            return pd.DataFrame()
        vpin_ts_i8 = vpin_ts.astype(np.int64)
        return pd.DataFrame(
            [
                _minute_toxicity_row(
                    minute=minute,
                    group=group,
                    large_threshold=large_threshold,
                    vpin_ts_i8=vpin_ts_i8,
                    vpin_vals=vpin_vals,
                )
                for minute, group in grouped
            ]
        )

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
