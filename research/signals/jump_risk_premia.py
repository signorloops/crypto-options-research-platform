"""
Jump risk premia signals from clustered return jumps.

This module provides a lightweight estimator that can be attached to
backtest pipelines as an optional alpha/risk control signal.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rolling_cluster_scores(masks: np.ndarray) -> np.ndarray:
    """Vectorized ``JumpRiskPremiaEstimator._cluster_score`` over mask rows.

    For each boolean row, score = max(0, 0.5*(mean_run - 1) + 0.5*(max_run - 1))
    where runs are maximal stretches of True. Trailing runs count too: the
    window rows are right-aligned and the left pad compares False, so a run
    that reaches the newest observation is still closed by construction.
    Rows with no True cell score 0, matching the scalar helper.

    Runs are measured via per-cell run-lengths (``run_len[:, j]`` = number of
    consecutive True cells ending at column j), built with one vectorized
    update per column — O(n * window) work in ``window`` numpy calls instead
    of a per-row Python loop. Exactly one cell per run has ``run_len == 1``
    (its first), so run counts and lengths fall out of the same array.
    """
    n_rows, width = masks.shape
    if n_rows == 0 or width == 0:
        return np.zeros(n_rows, dtype=float)

    run_len = np.zeros(masks.shape, dtype=np.int32)
    for j in range(width):
        if j == 0:
            run_len[:, 0] = masks[:, 0]
        else:
            run_len[:, j] = np.where(masks[:, j], run_len[:, j - 1] + 1, 0)

    any_run = masks.any(axis=1)
    max_run = run_len.max(axis=1).astype(float)
    n_true = masks.sum(axis=1).astype(float)
    n_runs = (run_len == 1).sum(axis=1).astype(float)  # one per run start

    mean_run = n_true / np.maximum(n_runs, 1.0)
    score = 0.5 * (mean_run - 1.0) + 0.5 * (max_run - 1.0)
    return np.where(any_run, np.maximum(score, 0.0), 0.0)


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
        arr = np.asarray(returns, dtype=float); arr = arr[np.isfinite(arr)]
        zero_signal = JumpRiskPremiaSignal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if arr.size < self.min_obs:
            return zero_signal
        arr = arr[-self.window :] if arr.size > self.window else arr
        mu = float(np.mean(arr)); sigma = float(np.std(arr))
        if sigma < 1e-10:
            return zero_signal
        upper = mu + self.jump_zscore * sigma; lower = mu - self.jump_zscore * sigma
        pos_mask = arr > upper; neg_mask = arr < lower
        pos_excess = arr[pos_mask] - upper
        neg_excess = lower - arr[neg_mask]
        pos_intensity = float(np.mean(pos_mask)); neg_intensity = float(np.mean(neg_mask))
        pos_cluster = self._cluster_score(pos_mask); neg_cluster = self._cluster_score(neg_mask)
        pos_premium = pos_intensity * self._safe_mean(pos_excess) * (1.0 + pos_cluster)
        neg_premium = neg_intensity * self._safe_mean(neg_excess) * (1.0 + neg_cluster)
        net_premium = pos_premium - neg_premium; imbalance_denom = pos_cluster + neg_cluster + 1e-12
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
        """Estimate jump premia as a time series from return series.

        Vectorized with numpy stride tricks and cumulative sums: rolling
        mean/std come from prefix sums (O(n)) while the jump masks, excess
        sums and run-length cluster scores are computed on an (n x window)
        sliding-window *view* of the returns. This replaces an O(n * window)
        pure-Python loop that re-ran the scalar estimator once per row.

        Output is row-for-row identical to calling ``estimate_from_returns``
        on each expanding window: rows whose window holds fewer than
        ``min_obs`` finite observations (or has ~zero variance) carry the
        all-zero signal, and non-finite entries are excluded from that row's
        statistics and masks, exactly as the scalar path filters them.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        values = returns.astype(float).fillna(0.0).to_numpy()
        n = values.size
        empty_cols = list(JumpRiskPremiaSignal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).to_dict())
        if n == 0:
            return pd.DataFrame({c: [] for c in empty_cols}, index=returns.index)

        finite = np.isfinite(values)
        vals = np.where(finite, values, 0.0)
        window = self.window

        # Right-aligned sliding windows: row i is values[max(0, i-window+1):i+1],
        # left-padded with NaN so pad positions never count as jumps.
        padded = np.concatenate([np.full(window, np.nan), np.where(finite, values, np.nan)])
        win = sliding_window_view(padded, window)[-n:]

        # Prefix-sum statistics over each row's *finite* entries.
        ends = np.arange(1, n + 1)
        starts = ends - np.minimum(ends, window)
        fcount = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
        cnt = fcount[ends] - fcount[starts]
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        csumsq = np.concatenate([[0.0], np.cumsum(vals * vals)])
        means = (csum[ends] - csum[starts]) / np.maximum(cnt, 1)
        second = (csumsq[ends] - csumsq[starts]) / np.maximum(cnt, 1)
        stds = np.sqrt(np.maximum(second - means * means, 0.0))

        upper = means + self.jump_zscore * stds
        lower = means - self.jump_zscore * stds

        pos_mask = win > upper[:, None]
        neg_mask = win < lower[:, None]
        pos_count = pos_mask.sum(axis=1).astype(float)
        neg_count = neg_mask.sum(axis=1).astype(float)
        safe_cnt = np.maximum(cnt, 1)
        pos_intensity = pos_count / safe_cnt
        neg_intensity = neg_count / safe_cnt

        pos_excess = np.where(pos_mask, win - upper[:, None], 0.0)
        neg_excess = np.where(neg_mask, lower[:, None] - win, 0.0)
        pos_excess_mean = np.where(
            pos_count > 0,
            pos_excess.sum(axis=1) / np.maximum(pos_count, 1.0),
            0.0,
        )
        neg_excess_mean = np.where(
            neg_count > 0,
            neg_excess.sum(axis=1) / np.maximum(neg_count, 1.0),
            0.0,
        )

        pos_cluster = _rolling_cluster_scores(pos_mask)
        neg_cluster = _rolling_cluster_scores(neg_mask)

        pos_premium = pos_intensity * pos_excess_mean * (1.0 + pos_cluster)
        neg_premium = neg_intensity * neg_excess_mean * (1.0 + neg_cluster)
        net_premium = pos_premium - neg_premium
        cluster_imbalance = (pos_cluster - neg_cluster) / (
            pos_cluster + neg_cluster + 1e-12
        )

        # Scalar path emits the zero signal when the window has too few
        # finite observations or is degenerate (sigma < 1e-10).
        inactive = (cnt < self.min_obs) | (stds < 1e-10)
        pos_premium[inactive] = 0.0
        neg_premium[inactive] = 0.0
        net_premium[inactive] = 0.0
        pos_intensity[inactive] = 0.0
        neg_intensity[inactive] = 0.0
        cluster_imbalance[inactive] = 0.0

        return pd.DataFrame(
            {
                "positive_jump_premium": np.maximum(pos_premium, 0.0),
                "negative_jump_premium": np.maximum(neg_premium, 0.0),
                "net_jump_premium": net_premium,
                "positive_jump_intensity": np.maximum(pos_intensity, 0.0),
                "negative_jump_intensity": np.maximum(neg_intensity, 0.0),
                "jump_cluster_imbalance": np.clip(cluster_imbalance, -1.0, 1.0),
            },
            index=returns.index,
        )

    def estimate_series_from_prices(self, prices: pd.Series) -> pd.DataFrame:
        """Estimate jump premia time series from price series."""
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        clean = prices.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        clean = clean.clip(lower=1e-12)

        log_ret = np.log(clean).diff().fillna(0.0)
        return self.estimate_series_from_returns(log_ret)

    def latest_from_prices(self, prices: pd.Series) -> Optional[JumpRiskPremiaSignal]:
        """Convenience API: estimate latest signal from price series.

        Only the tail that can influence the final row is processed — the
        rolling estimator's last output depends solely on the most recent
        ``window`` returns, i.e. the last ``window + 1`` prices. Slicing first
        (instead of computing the whole history to keep one row) makes this
        O(window) rather than O(n).
        """
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if len(prices) < self.min_obs:
            return None
        tail = prices.iloc[-(self.window + 1):] if len(prices) > self.window + 1 else prices
        series = self.estimate_series_from_prices(tail)
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
