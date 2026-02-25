import math

import pytest

from research.pricing.inverse_power_options import InversePowerOptionPricer
from research.pricing.quanto_inverse_power import QuantoInversePowerOptionPricer


def test_quanto_inverse_power_matches_base_when_no_fx_vol_and_unit_fx():
    params = dict(
        S=50000.0,
        K=50000.0,
        T=0.2,
        r=0.02,
        sigma=0.6,
        option_type="call",
        power=1.3,
        n_paths=30000,
        seed=42,
    )
    base = InversePowerOptionPricer.calculate_price(**params)
    quanto = QuantoInversePowerOptionPricer.calculate_price(
        **params,
        fx_rate=1.0,
        sigma_fx=0.0,
        rho=0.4,
    )
    assert math.isclose(quanto, base, rel_tol=1e-9, abs_tol=1e-12)


def test_quanto_inverse_power_price_decreases_with_positive_correlation():
    params = dict(
        S=52000.0,
        K=50000.0,
        T=0.5,
        r=0.01,
        sigma=0.55,
        option_type="put",
        power=1.1,
        n_paths=40000,
        seed=7,
        fx_rate=1.2,
        sigma_fx=0.8,
    )
    neg_corr = QuantoInversePowerOptionPricer.calculate_price(**params, rho=-0.6)
    pos_corr = QuantoInversePowerOptionPricer.calculate_price(**params, rho=0.6)

    assert pos_corr < neg_corr


def test_quanto_inverse_power_greeks_include_fx_and_corr_sensitivity():
    _, greeks = QuantoInversePowerOptionPricer.calculate_price_and_greeks(
        S=55000.0,
        K=50000.0,
        T=0.35,
        r=0.02,
        sigma=0.5,
        option_type="call",
        fx_rate=1.3,
        sigma_fx=0.7,
        rho=0.3,
        power=1.2,
        n_paths=20000,
        seed=12,
    )

    assert greeks.fx_delta < 0.0
    assert greeks.corr_sensitivity < 0.0
    assert greeks.quanto_adjustment > 0.0


def test_quanto_inverse_power_input_validation():
    with pytest.raises(ValueError):
        QuantoInversePowerOptionPricer.calculate_price(
            S=50000.0,
            K=50000.0,
            T=0.2,
            r=0.01,
            sigma=0.5,
            option_type="call",
            fx_rate=0.0,
            sigma_fx=0.6,
            rho=0.2,
            power=1.0,
        )

    with pytest.raises(ValueError):
        QuantoInversePowerOptionPricer.calculate_price(
            S=50000.0,
            K=50000.0,
            T=0.2,
            r=0.01,
            sigma=0.5,
            option_type="call",
            fx_rate=1.2,
            sigma_fx=-0.1,
            rho=0.2,
            power=1.0,
        )

    with pytest.raises(ValueError):
        QuantoInversePowerOptionPricer.calculate_price(
            S=50000.0,
            K=50000.0,
            T=0.2,
            r=0.01,
            sigma=0.5,
            option_type="call",
            fx_rate=1.2,
            sigma_fx=0.6,
            rho=1.1,
            power=1.0,
        )
