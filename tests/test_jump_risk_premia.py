"""
Tests for jump risk premia signal estimator.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from research.signals.jump_risk_premia import JumpRiskPremiaEstimator


class TestJumpRiskPremiaEstimator:
    """Test jump premia estimation logic."""

    def test_positive_jump_cluster_generates_positive_net_premium(self):
        """When positive jump clusters dominate, net premium should be positive."""
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 0.004, size=240)
        base[80:86] += 0.035
        base[160:164] += 0.028

        est = JumpRiskPremiaEstimator(window=120, jump_zscore=2.0)
        signal = est.estimate_from_returns(base)

        assert signal.positive_jump_premium > 0.0
        assert signal.net_jump_premium > 0.0

    def test_negative_jump_cluster_generates_negative_net_premium(self):
        """When negative jump clusters dominate, net premium should be negative."""
        rng = np.random.default_rng(7)
        base = rng.normal(0.0, 0.004, size=240)
        base[70:76] -= 0.036
        base[145:150] -= 0.030

        est = JumpRiskPremiaEstimator(window=120, jump_zscore=2.0)
        signal = est.estimate_from_returns(base)

        assert signal.negative_jump_premium > 0.0
        assert signal.net_jump_premium < 0.0

    def test_price_series_outputs_required_columns(self):
        """Series estimation should output all jump premium columns."""
        rng = np.random.default_rng(123)
        idx = pd.date_range(datetime(2024, 1, 1), periods=180, freq="min")
        rets = rng.normal(0.0, 0.003, size=180)
        rets[50] += 0.03
        rets[120] -= 0.028
        prices = 50000 * np.exp(np.cumsum(rets))
        series = pd.Series(prices, index=idx)

        est = JumpRiskPremiaEstimator(window=30, jump_zscore=2.0)
        out = est.estimate_series_from_prices(series)

        assert "positive_jump_premium" in out.columns
        assert "negative_jump_premium" in out.columns
        assert "net_jump_premium" in out.columns
        assert "jump_cluster_imbalance" in out.columns
        assert np.isfinite(out["net_jump_premium"].iloc[-1])
