"""
Tests for jump risk premia signal estimator.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

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


class TestSeriesVectorizationParity:
    """The vectorized series estimator must match the scalar path exactly.

    ``estimate_series_from_returns`` used to re-run the scalar estimator once
    per row in a Python loop. It is now computed with cumulative sums and a
    sliding-window view; these tests pin the output to the scalar reference so
    the rewrite cannot change the numbers.
    """

    def _reference_series(self, est, values):
        # Mirror the original loop exactly: fillna(0.0) is applied to the
        # whole series BEFORE each window is sliced, so NaN positions enter
        # the window as 0.0 observations (they are not dropped).
        filled = np.where(np.isfinite(values), values, 0.0)
        rows = []
        for idx in range(len(values)):
            start = max(0, idx + 1 - est.window)
            rows.append(est.estimate_from_returns(filled[start : idx + 1]).to_dict())
        return rows

    def test_matches_scalar_loop_on_random_data(self):
        rng = np.random.default_rng(123)
        values = rng.normal(0.0, 0.004, size=200)
        values[40] += 0.03
        values[41] += 0.028  # clustered positive jumps
        values[120] -= 0.026
        values[150] = np.nan  # non-finite entry

        est = JumpRiskPremiaEstimator(window=30, jump_zscore=2.0, min_obs=12)
        out = est.estimate_series_from_returns(pd.Series(values))
        ref = self._reference_series(est, values)

        assert len(out) == len(values)
        for i, ref_row in enumerate(ref):
            for key, expected in ref_row.items():
                assert out.iloc[i][key] == pytest.approx(expected, rel=1e-9, abs=1e-12), (
                    f"row {i} column {key}"
                )

    def test_matches_scalar_loop_across_window_regimes(self):
        rng = np.random.default_rng(7)
        for window, min_obs in [(3, 6), (10, 6), (50, 20)]:
            values = rng.normal(0.0, 0.005, size=120)
            values[::17] -= 0.02
            est = JumpRiskPremiaEstimator(window=window, jump_zscore=1.8, min_obs=min_obs)
            out = est.estimate_series_from_returns(pd.Series(values))
            ref = self._reference_series(est, values)
            for i, ref_row in enumerate(ref):
                for key, expected in ref_row.items():
                    assert out.iloc[i][key] == pytest.approx(expected, rel=1e-9, abs=1e-12), (
                        f"window={window} row {i} column {key}"
                    )

    def test_empty_series_returns_empty_frame(self):
        est = JumpRiskPremiaEstimator(window=10, jump_zscore=2.0)
        out = est.estimate_series_from_returns(pd.Series(dtype=float))
        assert out.empty
        assert list(out.columns) == list(
            JumpRiskPremiaEstimator()
            .estimate_from_returns(np.zeros(2))
            .to_dict()
        )

    def test_cluster_runs_count_trailing_runs(self):
        # A jump run that reaches the newest observation must still be scored
        # (the scalar helper counts trailing runs too).
        values = np.concatenate([np.zeros(30), np.full(4, 0.05), np.zeros(2), np.full(3, -0.06)])
        est = JumpRiskPremiaEstimator(window=12, jump_zscore=1.5, min_obs=8)
        out = est.estimate_series_from_returns(pd.Series(values))
        ref = self._reference_series(est, values)
        for i, ref_row in enumerate(ref):
            for key, expected in ref_row.items():
                assert out.iloc[i][key] == pytest.approx(expected, rel=1e-9, abs=1e-12)


class TestLatestFromPricesSlicing:
    """latest_from_prices must only process the tail that affects the last row."""

    def _prices(self, n=2000, seed=5):
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0, 0.003, size=n)
        rets[::250] += 0.02
        return pd.Series(50000.0 * np.exp(np.cumsum(rets)))

    def test_matches_full_series_last_row(self):
        est = JumpRiskPremiaEstimator(window=48, jump_zscore=2.5, min_obs=20)
        prices = self._prices()
        full = est.estimate_series_from_prices(prices)
        latest = est.latest_from_prices(prices)

        assert latest is not None
        last = full.iloc[-1]
        for key, value in latest.to_dict().items():
            assert value == pytest.approx(float(last[key]), rel=1e-12, abs=1e-15), key

    def test_short_series_returns_none(self):
        est = JumpRiskPremiaEstimator(window=48, jump_zscore=2.5, min_obs=20)
        assert est.latest_from_prices(pd.Series([100.0, 101.0, 102.0])) is None

    def test_rejects_non_series_input(self):
        est = JumpRiskPremiaEstimator()
        with pytest.raises(TypeError):
            est.latest_from_prices([100.0, 101.0, 102.0])
