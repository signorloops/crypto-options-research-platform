"""
Tests for volatility module.
"""
import numpy as np
import pytest

from research.volatility import (
    VolatilitySurface,
    black_scholes_price,
    ewma_volatility,
    garch_volatility,
    garman_klass_volatility,
    har_volatility,
    implied_volatility,
    implied_volatility_jaeckel,
    implied_volatility_lbr,
    parkinson_volatility,
    realized_variance,
    realized_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
)


class TestHistoricalVolatility:
    """Test historical volatility estimators."""

    def test_realized_variance_basic(self):
        """Test realized variance calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        rv = realized_variance(returns)
        expected = 0.01**2 + 0.01**2 + 0.02**2 + 0.02**2
        assert np.isclose(rv, expected)

    def test_realized_volatility_annualized(self):
        """Test realized volatility with annualization."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        vol = realized_volatility(returns, annualize=True, periods=365)
        rv = realized_variance(returns)
        n = len(returns)
        expected = np.sqrt(rv / n * 365)
        assert np.isclose(vol, expected)

    def test_parkinson_volatility(self):
        """Test Parkinson volatility estimation."""
        # Generate synthetic OHLC with known volatility
        high = np.array([102.0, 103.0, 104.0, 103.0])
        low = np.array([98.0, 97.0, 98.0, 97.0])
        vol = parkinson_volatility(high, low, annualize=False)
        assert vol > 0
        assert vol < 1.0

    def test_parkinson_empty(self):
        """Test Parkinson with empty data."""
        vol = parkinson_volatility(np.array([]), np.array([]))
        assert vol == 0.0

    def test_garman_klass_volatility(self):
        """Test Garman-Klass volatility."""
        open_p = np.array([100.0, 101.0, 102.0, 101.0])
        high = np.array([102.0, 103.0, 104.0, 103.0])
        low = np.array([98.0, 99.0, 100.0, 99.0])
        close = np.array([101.0, 102.0, 103.0, 102.0])

        vol = garman_klass_volatility(open_p, high, low, close, annualize=False)
        assert vol > 0
        assert vol < 1.0

    def test_rogers_satchell_volatility(self):
        """Test Rogers-Satchell volatility (drift-independent)."""
        open_p = np.array([100.0, 105.0, 110.0, 115.0])  # Strong trend
        high = np.array([102.0, 107.0, 112.0, 117.0])
        low = np.array([98.0, 103.0, 108.0, 113.0])
        close = np.array([101.0, 106.0, 111.0, 116.0])

        vol = rogers_satchell_volatility(open_p, high, low, close, annualize=False)
        assert vol > 0

    def test_yang_zhang_volatility(self):
        """Test Yang-Zhang volatility (most efficient)."""
        open_p = np.array([100.0, 101.0, 102.0, 101.0])
        high = np.array([102.0, 103.0, 104.0, 103.0])
        low = np.array([98.0, 99.0, 100.0, 99.0])
        close = np.array([101.0, 102.0, 103.0, 102.0])

        vol_yz = yang_zhang_volatility(open_p, high, low, close, annualize=False)
        assert vol_yz > 0


class TestVolatilityModels:
    """Test volatility forecasting models."""

    def test_ewma_volatility(self):
        """Test EWMA volatility."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        vol = ewma_volatility(returns, lambda_param=0.94, annualize=False)
        assert vol > 0
        assert vol < 1.0

    def test_ewma_series(self):
        """Test EWMA volatility series."""
        returns = np.array([0.01, -0.02, 0.015, -0.01])
        vols = ewma_volatility(returns, annualize=False)
        assert isinstance(vols, float)

    def test_garch_volatility(self):
        """Test GARCH(1,1) volatility."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        vol = garch_volatility(
            returns, omega=1e-6, alpha=0.1, beta=0.85, annualize=False
        )
        assert vol > 0
        assert vol < 1.0

    def test_garch_nonstationary(self):
        """Test GARCH with non-stationary parameters raises error."""
        returns = np.random.normal(0, 0.02, 50)
        with pytest.raises(ValueError, match="Non-stationary"):
            garch_volatility(returns, alpha=0.6, beta=0.5)

    def test_har_volatility(self):
        """Test HAR-RV model."""
        # Generate synthetic daily RV data
        np.random.seed(42)
        rv_daily = np.abs(np.random.normal(0.0001, 0.001, 30)) ** 2
        rv_pred = har_volatility(rv_daily)
        assert rv_pred >= 0


class TestImpliedVolatility:
    """Test implied volatility calculations."""

    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing."""
        price = black_scholes_price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=True
        )
        assert price > 0
        # ATM call with these params should be around 10.45
        assert 8 < price < 15

    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing."""
        price = black_scholes_price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=False
        )
        assert price > 0
        # ATM put with these params should be around 5.57
        assert 3 < price < 10

    def test_implied_volatility_recovery(self):
        """Test that implied vol recovers input volatility."""
        S, K, T, r = 100, 100, 0.5, 0.05
        true_sigma = 0.25

        # Calculate option price
        price = black_scholes_price(S, K, T, r, true_sigma, is_call=True)

        # Recover implied vol
        iv = implied_volatility(price, S, K, T, r, is_call=True)

        assert np.isclose(iv, true_sigma, atol=1e-4)

    def test_implied_volatility_different_strikes(self):
        """Test implied vol for ITM and OTM options."""
        S, T, r = 100, 0.5, 0.05
        sigma = 0.2

        for K in [90, 100, 110]:  # ITM, ATM, OTM
            price = black_scholes_price(S, K, T, r, sigma, is_call=True)
            iv = implied_volatility(price, S, K, T, r, is_call=True)
            assert np.isclose(iv, sigma, rtol=0.01)

    def test_implied_volatility_methods(self):
        """Test different implied vol methods."""
        S, K, T, r = 100, 100, 0.5, 0.05
        sigma = 0.25
        price = black_scholes_price(S, K, T, r, sigma, is_call=True)

        iv_bisection = implied_volatility(price, S, K, T, r, method="bisection")
        iv_newton = implied_volatility(price, S, K, T, r, method="newton")
        iv_hybrid = implied_volatility(price, S, K, T, r, method="hybrid")

        assert np.isclose(iv_bisection, sigma, atol=1e-3)
        assert np.isclose(iv_newton, sigma, atol=1e-4)
        assert np.isclose(iv_hybrid, sigma, atol=1e-4)

    def test_implied_volatility_lbr(self):
        """LBR-style solver should recover volatility robustly."""
        S, K, T, r = 100, 95, 0.75, 0.03
        sigma = 0.32
        price = black_scholes_price(S, K, T, r, sigma, is_call=True)
        iv = implied_volatility_lbr(price, S, K, T, r, is_call=True)
        assert np.isclose(iv, sigma, atol=1e-4)

    def test_implied_volatility_jaeckel(self):
        """Jaeckel entrypoint should recover volatility (with fallback if dependency missing)."""
        S, K, T, r = 100, 105, 0.5, 0.01
        sigma = 0.28
        price = black_scholes_price(S, K, T, r, sigma, is_call=True)
        iv = implied_volatility_jaeckel(price, S, K, T, r, is_call=True)
        assert np.isclose(iv, sigma, atol=1e-4)


class TestVolatilitySurface:
    """Test volatility surface functionality."""

    def test_surface_creation(self):
        """Test building volatility surface."""
        surface = VolatilitySurface()

        strikes = [90, 95, 100, 105, 110]
        expiries = [0.25, 0.5, 0.75]

        S = 100
        r = 0.05
        true_vol = 0.2

        # Generate market prices
        market_prices = []
        for T in expiries:
            for K in strikes:
                price = black_scholes_price(S, K, T, r, true_vol, is_call=True)
                market_prices.append(price)

        # Repeat strikes and expiries to match
        all_strikes = strikes * len(expiries)
        all_expiries = [e for e in expiries for _ in strikes]

        surface.add_from_market_data(
            all_strikes, all_expiries, market_prices, S, r
        )

        assert len(surface.points) == len(all_strikes)

    def test_surface_interpolation(self):
        """Test surface interpolation."""
        surface = VolatilitySurface()
        from research.volatility.implied import VolatilityPoint

        # Add some points
        surface.add_point(VolatilityPoint(
            strike=90, expiry=0.5, volatility=0.22,
            underlying_price=100, is_call=True
        ))
        surface.add_point(VolatilityPoint(
            strike=110, expiry=0.5, volatility=0.18,
            underlying_price=100, is_call=True
        ))

        # Interpolate
        vol = surface.get_volatility(100, 0.5)
        assert 0.18 < vol < 0.22

    def test_surface_skew(self):
        """Test getting volatility skew."""
        surface = VolatilitySurface()
        from research.volatility.implied import VolatilityPoint

        for K in [90, 95, 100, 105, 110]:
            surface.add_point(VolatilityPoint(
                strike=K, expiry=0.5, volatility=0.2 + (K-100)*0.001,
                underlying_price=100, is_call=True
            ))

        skew = surface.get_skew(0.5)
        assert len(skew) == 5
        assert all(isinstance(x, tuple) and len(x) == 2 for x in skew)

    def test_surface_summary(self):
        """Test surface summary statistics."""
        surface = VolatilitySurface()
        from research.volatility.implied import VolatilityPoint

        for i, K in enumerate([90, 100, 110]):
            surface.add_point(VolatilityPoint(
                strike=K, expiry=0.5, volatility=0.18 + i*0.02,
                underlying_price=100, is_call=True
            ))

        summary = surface.summary()
        assert "min_vol" in summary
        assert "max_vol" in summary
        assert "mean_vol" in summary
        assert summary["min_vol"] == 0.18
        assert summary["max_vol"] == 0.22

    def test_surface_empty(self):
        """Test empty surface behavior."""
        surface = VolatilitySurface()

        # Should return default value
        vol = surface.get_volatility(100, 0.5)
        assert vol == 0.2

        summary = surface.summary()
        assert summary == {}

    def test_surface_svi_fit_and_no_arb_check(self):
        """SVI fit should run and no-arbitrage checker returns structured output."""
        surface = VolatilitySurface()
        S = 100.0
        r = 0.01
        expiries = [0.25, 0.5]
        strikes = [80, 90, 100, 110, 120]

        for T in expiries:
            for K in strikes:
                vol = 0.2 + 0.05 * abs(np.log(K / S))
                price = black_scholes_price(S, K, T, r, vol, is_call=True)
                surface.add_from_market_data([K], [T], [price], S, r)

        params = surface.fit_all_svi()
        assert len(params) >= 1

        validation = surface.validate_no_arbitrage()
        assert "butterfly" in validation
        assert "calendar" in validation
        assert "no_arbitrage" in validation

    def test_surface_ssvi_mode_is_distinct_and_bounded(self):
        """SSVI interpolation should run in its own path and keep vols bounded."""
        surface = VolatilitySurface()
        S = 100.0
        r = 0.01
        expiries = [0.1, 0.25, 0.5, 1.0]
        strikes = [80, 90, 100, 110, 120]

        for T in expiries:
            for K in strikes:
                k = np.log(K / S)
                vol = 0.18 + 0.06 * abs(k) + 0.05 * np.sqrt(T)
                price = black_scholes_price(S, K, T, r, vol, is_call=True)
                surface.add_from_market_data([K], [T], [price], S, r)

        vol_linear = surface.get_volatility(97.0, 0.37, method="linear")
        vol_ssvi = surface.get_volatility(97.0, 0.37, method="ssvi")

        assert abs(vol_ssvi - vol_linear) > 1e-6
        assert 0.01 <= vol_ssvi <= 2.0

    def test_surface_ssvi_total_variance_non_decreasing_across_maturity(self):
        """At fixed moneyness, SSVI total variance should be non-decreasing in maturity."""
        surface = VolatilitySurface()
        S = 100.0
        r = 0.01
        expiries = [0.12, 0.25, 0.5, 0.9]
        strikes = [85, 95, 100, 105, 115]

        for T in expiries:
            for K in strikes:
                k = np.log(K / S)
                vol = 0.20 + 0.04 * np.sqrt(T) + 0.05 * abs(k)
                price = black_scholes_price(S, K, T, r, vol, is_call=True)
                surface.add_from_market_data([K], [T], [price], S, r)

        moneyness = 1.05
        total_vars = []
        for T in expiries:
            v = surface.get_volatility(S * moneyness, T, method="ssvi")
            total_vars.append(v * v * T)

        for i in range(1, len(total_vars)):
            assert total_vars[i] + 1e-8 >= total_vars[i - 1]
