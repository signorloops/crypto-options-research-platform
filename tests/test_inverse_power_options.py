"""Tests for inverse-power option pricer."""

import numpy as np

from research.pricing.inverse_options import InverseOptionPricer
from research.pricing.inverse_power_options import InversePowerOptionPricer


def test_price_is_non_negative_and_deterministic():
    kwargs = dict(
        S=50000.0,
        K=50000.0,
        T=30.0 / 365.0,
        r=0.02,
        sigma=0.6,
        option_type="call",
        power=1.5,
        n_paths=20000,
        seed=42,
    )
    p1 = InversePowerOptionPricer.calculate_price(**kwargs)
    p2 = InversePowerOptionPricer.calculate_price(**kwargs)

    assert p1 >= 0.0
    assert np.isfinite(p1)
    assert np.isclose(p1, p2)


def test_intrinsic_value_at_expiry():
    call = InversePowerOptionPricer.calculate_price(
        S=60000.0,
        K=50000.0,
        T=0.0,
        r=0.02,
        sigma=0.5,
        option_type="call",
        power=1.0,
        n_paths=4096,
    )
    put = InversePowerOptionPricer.calculate_price(
        S=40000.0,
        K=50000.0,
        T=0.0,
        r=0.02,
        sigma=0.5,
        option_type="put",
        power=1.0,
        n_paths=4096,
    )

    assert np.isclose(call, max(1.0 / 50000.0 - 1.0 / 60000.0, 0.0))
    assert np.isclose(put, max(1.0 / 40000.0 - 1.0 / 50000.0, 0.0))


def test_power_one_mc_is_reasonably_close_to_closed_form_inverse_price():
    params = dict(
        S=50000.0,
        K=50000.0,
        T=45.0 / 365.0,
        r=0.01,
        sigma=0.55,
        option_type="call",
    )
    closed_form = InverseOptionPricer.calculate_price(**params)
    mc = InversePowerOptionPricer.calculate_price(
        **params,
        power=1.0,
        n_paths=120000,
        seed=7,
    )

    assert abs(mc - closed_form) < 3e-4


def test_price_and_greeks_outputs_finite_values():
    price, greeks = InversePowerOptionPricer.calculate_price_and_greeks(
        S=50000.0,
        K=52000.0,
        T=20.0 / 365.0,
        r=0.01,
        sigma=0.5,
        option_type="put",
        power=1.2,
        n_paths=20000,
        seed=11,
    )

    assert np.isfinite(price)
    assert np.isfinite(greeks.delta)
    assert np.isfinite(greeks.gamma)
    assert np.isfinite(greeks.theta)
    assert np.isfinite(greeks.vega)
    assert np.isfinite(greeks.rho)
