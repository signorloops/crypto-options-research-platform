"""
Jump risk premia signals from clustered return jumps.

This module provides a lightweight estimator that can be attached to
backtest pipelines as an optional alpha/risk control signal.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class JumpRiskPremiaSignal:
    """Container for jump premia signals."""

    positive_jump_premium: float
    negative_jump_premium: float
    net_jump_premium: float
    positive_jump_intensity: float
    negative_jump_intensity: float
    jump_cluster_imbalance: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to plain dictionary."""
        return {
            "positive_jump_premium": float(self.positive_jump_premium),
            "negative_jump_premium": float(self.negative_jump_premium),
            "net_jump_premium": float(self.net_jump_premium),
            "positive_jump_intensity": float(self.positive_jump_intensity),
            "negative_jump_intensity": float(self.negative_jump_intensity),
            "jump_cluster_imbalance": float(self.jump_cluster_imbalance),
        }


class JumpRiskPremiaEstimator:
    """
    Estimate positive/negative jump premia from return windows.

    A jump is detected by standardized return threshold exceedance.
    Premium is proportional to jump exceedance size, intensity, and clustering.
    """

    def __init__(
        self,
        window: int = 96,
        jump_zscore: float = 2.5,
        min_obs: int = 30,
    ):
        if window <= 2:
            raise ValueError("window must be > 2")
        if jump_zscore <= 0:
            raise ValueError("jump_zscore must be positive")
        if min_obs <= 5:
            raise ValueError("min_obs must be > 5")
        self.window = int(window)
        self.jump_zscore = float(jump_zscore)
        self.min_obs = int(min_obs)

    @staticmethod
    def _cluster_score(mask: np.ndarray) -> float:
        """Return clustering strength from consecutive jump runs."""
        if mask.size == 0:
            return 0.0

        run_lengths = []
        run = 0
        for flag in mask:
            if flag:
                run += 1
            elif run > 0:
                run_lengths.append(run)
                run = 0
        if run > 0:
            run_lengths.append(run)

        if not run_lengths:
            return 0.0
        mean_run = float(np.mean(run_lengths))
        max_run = float(np.max(run_lengths))
        # 0 for isolated jumps, increases with clustering persistence.
        return max(0.0, 0.5 * (mean_run - 1.0) + 0.5 * (max_run - 1.0))

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size > 0 else 0.0

    def estimate_from_returns(self, returns: np.ndarray) -> JumpRiskPremiaSignal:
        """Estimate jump premia from a return array."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < self.min_obs:
            return JumpRiskPremiaSignal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Focus on the most recent window.
        if arr.size > self.window:
            arr = arr[-self.window :]

        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        if sigma < 1e-10:
            return JumpRiskPremiaSignal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        upper = mu + self.jump_zscore * sigma
        lower = mu - self.jump_zscore * sigma

        pos_mask = arr > upper
        neg_mask = arr < lower

        pos_excess = arr[pos_mask] - upper
        neg_excess = lower - arr[neg_mask]

        pos_intensity = float(np.mean(pos_mask))
        neg_intensity = float(np.mean(neg_mask))

        pos_cluster = self._cluster_score(pos_mask)
        neg_cluster = self._cluster_score(neg_mask)

        pos_premium = pos_intensity * self._safe_mean(pos_excess) * (1.0 + pos_cluster)
        neg_premium = neg_intensity * self._safe_mean(neg_excess) * (1.0 + neg_cluster)
        net_premium = pos_premium - neg_premium

        imbalance_denom = pos_cluster + neg_cluster + 1e-12
        cluster_imbalance = (pos_cluster - neg_cluster) / imbalance_denom

        return JumpRiskPremiaSignal(
            positive_jump_premium=float(max(pos_premium, 0.0)),
            negative_jump_premium=float(max(neg_premium, 0.0)),
            net_jump_premium=float(net_premium),
            positive_jump_intensity=float(max(pos_intensity, 0.0)),
            negative_jump_intensity=float(max(neg_intensity, 0.0)),
            jump_cluster_imbalance=float(np.clip(cluster_imbalance, -1.0, 1.0)),
        )

    def estimate_series_from_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Estimate jump premia as a time series from return series."""
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        values = returns.astype(float).fillna(0.0).to_numpy()

        rows = []
        for idx in range(len(values)):
            start = max(0, idx + 1 - self.window)
            signal = self.estimate_from_returns(values[start : idx + 1])
            rows.append(signal.to_dict())

        return pd.DataFrame(rows, index=returns.index)

    def estimate_series_from_prices(self, prices: pd.Series) -> pd.DataFrame:
        """Estimate jump premia time series from price series."""
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        clean = prices.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        clean = clean.clip(lower=1e-12)

        log_ret = np.log(clean).diff().fillna(0.0)
        return self.estimate_series_from_returns(log_ret)

    def latest_from_prices(self, prices: pd.Series) -> Optional[JumpRiskPremiaSignal]:
        """Convenience API: estimate latest signal from price series."""
        if len(prices) < self.min_obs:
            return None
        series = self.estimate_series_from_prices(prices)
        if series.empty:
            return None
        last = series.iloc[-1]
        return JumpRiskPremiaSignal(
            positive_jump_premium=float(last["positive_jump_premium"]),
            negative_jump_premium=float(last["negative_jump_premium"]),
            net_jump_premium=float(last["net_jump_premium"]),
            positive_jump_intensity=float(last["positive_jump_intensity"]),
            negative_jump_intensity=float(last["negative_jump_intensity"]),
            jump_cluster_imbalance=float(last["jump_cluster_imbalance"]),
        )
