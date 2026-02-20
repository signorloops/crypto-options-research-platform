"""
Tests for volatility models (GARCH, HAR-RV, EWMA).
"""
import numpy as np

from research.volatility.models import (
    bipower_variation,
    egarch_volatility,
    estimate_garch_params,
    ewma_series,
    ewma_volatility,
    garch_volatility,
    gjr_garch_volatility,
    hamilton_filter_regime_switching,
    har_forecast,
    har_volatility,
    medrv_volatility,
    realized_kernel_volatility,
    rough_volatility_signature,
    two_scale_realized_volatility,
    volatility_regime_switching,
)


class TestEWMA:
    """Test EWMA volatility functions."""

    def test_ewma_volatility_basic(self):
        """Test basic EWMA calculation."""
        returns = np.random.normal(0, 0.02, 100)

        volatility = ewma_volatility(returns, lambda_param=0.94)

        assert isinstance(volatility, float)
        assert volatility > 0

    def test_ewma_series(self):
        """Test EWMA series calculation."""
        returns = np.random.normal(0, 0.02, 100)

        series = ewma_series(returns, lambda_param=0.94)

        assert isinstance(series, np.ndarray)
        assert len(series) == len(returns)
        assert all(v >= 0 for v in series)

    def test_ewma_different_lambda(self):
        """Test with different decay factors."""
        returns = np.random.normal(0, 0.02, 50)

        vol_fast = ewma_volatility(returns, lambda_param=0.9)
        vol_slow = ewma_volatility(returns, lambda_param=0.99)

        # Both should be positive
        assert vol_fast > 0
        assert vol_slow > 0


class TestGARCH:
    """Test GARCH volatility functions."""

    def test_garch_volatility_basic(self):
        """Test GARCH calculation with fixed parameters."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)

        # garch_volatility returns a float (current volatility estimate)
        volatility = garch_volatility(returns, omega=0.000001, alpha=0.1, beta=0.85)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_estimate_garch_params(self):
        """Test GARCH parameter estimation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 200)

        omega, alpha, beta = estimate_garch_params(returns)

        assert omega > 0
        assert alpha >= 0
        assert beta >= 0
        assert alpha + beta < 1.0  # Stationarity


class TestHAR:
    """Test HAR-RV (Heterogeneous Autoregressive) functions."""

    def test_har_volatility_basic(self):
        """Test HAR-RV calculation."""
        np.random.seed(42)
        rv_daily = np.abs(np.random.normal(0, 0.02, 100)) ** 2

        forecast = har_volatility(rv_daily, periods=(1, 5, 22))

        assert isinstance(forecast, float)
        assert forecast >= 0

    def test_har_forecast(self):
        """Test HAR-RV forecasting."""
        np.random.seed(42)
        rv_series = np.abs(np.random.normal(0, 0.02, 100)) ** 2

        forecast = har_forecast(rv_series, h=5)

        assert isinstance(forecast, float)
        assert forecast >= 0


class TestRoughVolatility:
    """Test rough volatility functions."""

    def test_rough_volatility_signature(self):
        """Test rough volatility signature calculation."""
        np.random.seed(42)
        log_prices = np.cumsum(np.random.normal(0, 0.001, 1000))

        signature = rough_volatility_signature(log_prices, sampling="daily")

        assert isinstance(signature, float)
        assert signature >= 0


class TestVolatilityRegime:
    """Test regime switching model."""

    def test_regime_switching(self):
        """Test volatility regime switching detection."""
        np.random.seed(42)
        # Create returns with different volatilities
        low_vol = np.random.normal(0, 0.01, 100)
        high_vol = np.random.normal(0, 0.05, 100)
        returns = np.concatenate([low_vol, high_vol])

        result = volatility_regime_switching(returns, n_states=2)

        # Result contains regime info (actual keys may vary)
        assert isinstance(result, dict)
        # Check that result has expected structure with high/low vol states
        assert "high_vol_state" in result or "low_vol_state" in result or "current_high_vol_probability" in result

    def test_hamilton_filter_regime_switching(self):
        """Hamilton filter variant should return transition matrix and probabilities."""
        np.random.seed(42)
        returns = np.concatenate([
            np.random.normal(0, 0.01, 120),
            np.random.normal(0, 0.04, 120),
        ])
        out = hamilton_filter_regime_switching(returns)
        assert "transition_matrix" in out
        assert "current_high_vol_probability" in out
        assert len(out["transition_matrix"]) == 2


class TestRobustVolatilityEstimators:
    """Test new robust volatility estimators."""

    def test_bipower_and_medrv(self):
        returns = np.random.normal(0, 0.01, 200)
        bv = bipower_variation(returns, annualize=False)
        med = medrv_volatility(returns, annualize=False)
        assert bv >= 0
        assert med >= 0

    def test_two_scale_and_kernel(self):
        returns = np.random.normal(0, 0.01, 300)
        ts = two_scale_realized_volatility(returns, annualize=False)
        rk = realized_kernel_volatility(returns, annualize=False)
        assert ts >= 0
        assert rk >= 0

    def test_egarch_and_gjr(self):
        returns = np.random.normal(0, 0.02, 300)
        e = egarch_volatility(returns, annualize=False)
        g = gjr_garch_volatility(returns, annualize=False)
        assert e >= 0
        assert g >= 0
