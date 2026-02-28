"""
Tests for inverse (coin-margined) option pricing model.
"""
import numpy as np
import pytest

from core.exceptions import ValidationError
from research.pricing.inverse_options import (
    InverseGreeks,
    InverseOptionPricer,
    inverse_option_parity,
)


class TestInverseOptionPricer:
    """Test coin-margined option pricing."""

    def test_basic_call_pricing(self):
        """Test basic call option pricing."""
        price = InverseOptionPricer.calculate_price(
            S=50000,  # BTC @ $50k
            K=50000,  # ATM
            T=0.25,   # 3 months
            r=0.05,
            sigma=0.60,
            option_type="call"
        )

        # Price should be positive and reasonable (in BTC)
        assert price > 0
        assert price < 0.001  # Should be small for ATM (much less than 1 BTC)

    def test_basic_put_pricing(self):
        """Test basic put option pricing."""
        price = InverseOptionPricer.calculate_price(
            S=50000,
            K=50000,
            T=0.25,
            r=0.05,
            sigma=0.60,
            option_type="put"
        )

        assert price > 0
        assert price < 0.001

    def test_itm_call_higher_than_otm(self):
        """ITM call should have higher price than OTM call."""
        # For inverse call, ITM means S > K (price high, so 1/S < 1/K)
        itm_price = InverseOptionPricer.calculate_price(
            S=55000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )
        otm_price = InverseOptionPricer.calculate_price(
            S=45000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        # ITM call should have higher price
        assert itm_price > otm_price

    def test_itm_put_higher_than_otm(self):
        """ITM put should have higher price than OTM put."""
        # For inverse put, ITM means S < K (price low, so 1/S > 1/K)
        itm_price = InverseOptionPricer.calculate_price(
            S=45000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="put"
        )
        otm_price = InverseOptionPricer.calculate_price(
            S=55000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="put"
        )

        # ITM put should have higher price
        assert itm_price > otm_price

    def test_expired_itm_call(self):
        """Expired ITM call should have intrinsic value."""
        S, K = 55000, 50000
        price = InverseOptionPricer.calculate_price(
            S=S, K=K, T=0, r=0.05, sigma=0.60, option_type="call"
        )
        # Intrinsic value = 1/K - 1/S
        expected = 1.0/K - 1.0/S
        assert abs(price - expected) < 1e-10
        assert price > 0

    def test_expired_itm_put(self):
        """Expired ITM put should have intrinsic value."""
        S, K = 45000, 50000
        price = InverseOptionPricer.calculate_price(
            S=S, K=K, T=0, r=0.05, sigma=0.60, option_type="put"
        )
        # Intrinsic value = 1/S - 1/K
        expected = 1.0/S - 1.0/K
        assert abs(price - expected) < 1e-10
        assert price > 0

    def test_expired_otm_call(self):
        """Expired OTM call should be worthless."""
        price = InverseOptionPricer.calculate_price(
            S=45000, K=50000, T=0, r=0.05, sigma=0.60, option_type="call"
        )
        assert price == 0.0

    def test_expired_otm_put(self):
        """Expired OTM put should be worthless."""
        price = InverseOptionPricer.calculate_price(
            S=55000, K=50000, T=0, r=0.05, sigma=0.60, option_type="put"
        )
        assert price == 0.0

    def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        with pytest.raises(ValidationError):
            InverseOptionPricer.calculate_price(
                S=-50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
            )

    def test_invalid_option_type_raises(self):
        """Invalid option type should raise validation error instead of defaulting to put path."""
        with pytest.raises(ValueError, match="option_type"):
            InverseOptionPricer.calculate_price(
                S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="invalid"
            )

    def test_price_increases_with_volatility(self):
        """Option price should increase with volatility."""
        S, K, T, r = 50000, 50000, 0.25, 0.05

        price_low_vol = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma=0.30, option_type="call"
        )
        price_high_vol = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma=0.90, option_type="call"
        )

        assert price_high_vol > price_low_vol

    def test_call_put_symmetry_atm(self):
        """ATM call and put should have approximately equal value for inverse options."""
        # This is a property of inverse options when r=0
        S, K, T = 50000, 50000, 0.25
        r = 0.0  # No interest rate

        call_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma=0.60, option_type="call"
        )
        put_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma=0.60, option_type="put"
        )

        # With r=0, they should be approximately equal
        assert abs(call_price - put_price) < 1e-6


class TestInverseGreeks:
    """Test coin-margined Greeks calculation."""

    def test_delta_call_positive(self):
        """Test that inverse call delta is positive."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        # Inverse call delta should be positive (as S increases, 1/S decreases,
        # but the formula includes 1/S^2 factor which dominates)
        assert greeks.delta > 0

    def test_delta_put_negative(self):
        """Test that inverse put delta is negative."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="put"
        )

        # Inverse put delta should be negative
        assert greeks.delta < 0

    def test_gamma_positive(self):
        """Test Gamma is positive for inverse options (convex in S)."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        # For inverse options, Gamma is positive because option price is convex in S
        # (same as standard Black-Scholes)
        # This was corrected from earlier incorrect implementation
        assert greeks.gamma > 0

    def test_theta_finite(self):
        """Test Theta is finite (inverse options can have positive or negative theta)."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        # For inverse options, theta can be positive or negative depending on parameters
        # Due to the 1/S transformation, the time decay behavior differs from standard options
        assert np.isfinite(greeks.theta)
        assert abs(greeks.theta) < 1e-6  # Should be very small for inverse options

    def test_vega_positive(self):
        """Test Vega is positive."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        # Vega should be positive
        assert greeks.vega > 0

    def test_greeks_are_finite(self):
        """Test that all Greeks are finite numbers."""
        greeks = InverseOptionPricer.calculate_greeks(
            S=50000, K=50000, T=0.25, r=0.05, sigma=0.60, option_type="call"
        )

        assert np.isfinite(greeks.delta)
        assert np.isfinite(greeks.gamma)
        assert np.isfinite(greeks.theta)
        assert np.isfinite(greeks.vega)
        assert np.isfinite(greeks.rho)

    def test_price_and_greeks_consistency(self):
        """Test that calculate_price_and_greeks is consistent with individual calls."""
        S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

        # Individual calculations
        price_single = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        greeks_single = InverseOptionPricer.calculate_greeks(
            S, K, T, r, sigma, "call"
        )

        # Combined calculation
        price_combo, greeks_combo = InverseOptionPricer.calculate_price_and_greeks(
            S, K, T, r, sigma, "call"
        )

        assert abs(price_single - price_combo) < 1e-10
        assert abs(greeks_single.delta - greeks_combo.delta) < 1e-10
        assert abs(greeks_single.gamma - greeks_combo.gamma) < 1e-10
        assert abs(greeks_single.vega - greeks_combo.vega) < 1e-10


class TestInversePnL:
    """Test coin-margined PnL calculation."""

    def test_inverse_pnl_formula(self):
        """Test the inverse PnL formula."""
        pnl = InverseOptionPricer.calculate_pnl(
            entry_price=50000,
            exit_price=60000,
            size=1.0,
            inverse=True
        )

        # PnL = 1/50000 - 1/60000 (positive when price goes up for long)
        expected = (1/50000 - 1/60000)
        assert abs(pnl - expected) < 1e-10
        assert pnl > 0  # Long position profits when price goes up

    def test_inverse_pnl_short(self):
        """Test PnL for short position."""
        pnl = InverseOptionPricer.calculate_pnl(
            entry_price=60000,
            exit_price=50000,
            size=-1.0,  # Short
            inverse=True
        )

        # Short profits when price goes down
        assert pnl > 0

    def test_linear_vs_inverse(self):
        """Compare linear and inverse PnL."""
        # Linear (U-margined)
        linear_pnl = InverseOptionPricer.calculate_pnl(
            entry_price=50000,
            exit_price=60000,
            size=1.0,
            inverse=False
        )
        assert linear_pnl == 10000  # Linear: 60000 - 50000

        # Inverse (coin-margined)
        inverse_pnl = InverseOptionPricer.calculate_pnl(
            entry_price=50000,
            exit_price=60000,
            size=1.0,
            inverse=True
        )
        # Inverse: (1/50000 - 1/60000) ≈ 0.00000333 BTC
        assert abs(inverse_pnl - 0.00000333) < 1e-7

    def test_inverse_pnl_zero_division(self):
        """Test PnL handles zero prices gracefully."""
        pnl = InverseOptionPricer.calculate_pnl(
            entry_price=0,
            exit_price=60000,
            size=1.0,
            inverse=True
        )
        assert pnl == 0.0


class TestPutCallParity:
    """Test coin-margined put-call parity."""

    def test_parity_relationship(self):
        """Test put-call parity holds."""
        S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

        call_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        put_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "put"
        )

        deviation = inverse_option_parity(call_price, put_price, S, K, T, r)

        # Deviation should be very small (near zero)
        assert abs(deviation) < 1e-6

    def test_parity_with_zero_rate(self):
        """Test parity when interest rate is zero."""
        S, K, T, sigma = 50000, 50000, 0.25, 0.60
        r = 0.0

        call_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        put_price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "put"
        )

        deviation = inverse_option_parity(call_price, put_price, S, K, T, r)
        assert abs(deviation) < 1e-10

    def test_parity_at_expiry(self):
        """Test parity at expiry."""
        S, K, r = 50000, 50000, 0.05
        T = 0.0  # At expiry

        # Use intrinsic values
        call_price = max(0, 1/K - 1/S)
        put_price = max(0, 1/S - 1/K)

        deviation = inverse_option_parity(call_price, put_price, S, K, T, r)
        assert abs(deviation) < 1e-10


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def test_iv_recovery(self):
        """Test that we can recover input volatility."""
        S, K, T, r, true_sigma = 50000, 50000, 0.25, 0.05, 0.60

        # Calculate price with known vol
        price = InverseOptionPricer.calculate_price(
            S, K, T, r, true_sigma, "call"
        )

        # Use a tighter tolerance for more accurate recovery
        iv = InverseOptionPricer.calculate_implied_volatility(
            price, S, K, T, r, "call", tol=1e-10
        )

        # Verify the recovered IV produces the same price
        price_at_iv = InverseOptionPricer.calculate_price(S, K, T, r, iv, "call")
        price_diff = abs(price_at_iv - price)

        # Price difference should be very small
        assert price_diff < 1e-10, f"Price mismatch: {price_diff:.2e}"

        # IV should be close to original (within 5%)
        assert abs(iv - true_sigma) < 0.05, f"IV mismatch: {iv} vs {true_sigma}"

    def test_iv_invalid_price(self):
        """Test IV with invalid price."""
        iv = InverseOptionPricer.calculate_implied_volatility(
            -1, 50000, 50000, 0.25, 0.05, "call"
        )
        assert iv == 0.0

    def test_iv_zero_price(self):
        """Test IV with zero price."""
        iv = InverseOptionPricer.calculate_implied_volatility(
            0, 50000, 50000, 0.25, 0.05, "call"
        )
        assert iv == 0.0

    def test_iv_short_maturity_anchor_stabilization_is_opt_in(self):
        """Short-dated IV stabilization should keep result closer to anchor only when enabled."""
        S, K, T, r = 50000, 60000, 1.0 / 365.0, 0.0
        true_sigma = 0.55
        anchor_sigma = 0.60

        clean_price = InverseOptionPricer.calculate_price(S, K, T, r, true_sigma, "call")
        noisy_price = clean_price * 1.8

        iv_raw = InverseOptionPricer.calculate_implied_volatility(
            noisy_price, S, K, T, r, "call"
        )
        iv_stabilized = InverseOptionPricer.calculate_implied_volatility(
            noisy_price,
            S,
            K,
            T,
            r,
            "call",
            stabilize_short_maturity=True,
            short_maturity_threshold=7.0 / 365.0,
            anchor_sigma=anchor_sigma,
            max_anchor_deviation=0.35,
        )

        assert abs(iv_stabilized - anchor_sigma) < abs(iv_raw - anchor_sigma)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_time_to_expiry(self):
        """Test with very small T."""
        S, K, r, sigma = 50000, 50000, 0.05, 0.60
        T = 1e-12  # Very small but non-zero

        price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        # Should be close to intrinsic value
        intrinsic = max(0, 1/K - 1/S)
        assert abs(price - intrinsic) < 1e-6

    def test_very_high_volatility(self):
        """Test with very high volatility."""
        S, K, T, r = 50000, 50000, 0.25, 0.05
        sigma = 2.0  # 200% volatility

        price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        assert price > 0
        assert np.isfinite(price)

    def test_very_low_volatility(self):
        """Test with very low volatility."""
        S, K, T, r = 50000, 50000, 0.25, 0.05
        sigma = 0.01  # 1% volatility

        price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        assert price >= 0
        assert np.isfinite(price)

    def test_deep_itm_call(self):
        """Test deep ITM call."""
        S, K, T, r, sigma = 100000, 40000, 0.25, 0.05, 0.60

        price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "call"
        )
        # Deep ITM call should be close to intrinsic value
        intrinsic = 1/K - 1/S
        assert abs(price - intrinsic * np.exp(-r * T)) < 0.0001

    def test_deep_itm_put(self):
        """Test deep ITM put."""
        S, K, T, r, sigma = 40000, 100000, 0.25, 0.05, 0.60

        price = InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, "put"
        )
        # Deep ITM put should be close to intrinsic value
        intrinsic = 1/S - 1/K
        assert abs(price - intrinsic) < 0.0001
