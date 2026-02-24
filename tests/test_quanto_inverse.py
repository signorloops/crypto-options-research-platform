import math

import pytest

from research.hedging.quanto_inverse import QuantoInverseHedger
from research.pricing.inverse_options import InverseOptionPricer
from research.pricing.quanto_inverse import QuantoInverseOptionPricer


def test_quanto_price_matches_inverse_when_no_fx_vol_and_unit_fx():
    S, K, T, r, sigma = 50000.0, 50000.0, 0.25, 0.03, 0.65
    base = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")
    quanto = QuantoInverseOptionPricer.calculate_price(
        S, K, T, r, sigma, "call", fx_rate=1.0, sigma_fx=0.0, rho=0.5
    )
    assert math.isclose(quanto, base, rel_tol=1e-9, abs_tol=1e-12)


def test_quanto_price_decreases_with_positive_correlation():
    S, K, T, r, sigma = 50000.0, 50000.0, 0.5, 0.02, 0.6

    neg_corr = QuantoInverseOptionPricer.calculate_price(
        S, K, T, r, sigma, "put", fx_rate=1.2, sigma_fx=0.8, rho=-0.6
    )
    pos_corr = QuantoInverseOptionPricer.calculate_price(
        S, K, T, r, sigma, "put", fx_rate=1.2, sigma_fx=0.8, rho=0.6
    )

    assert pos_corr < neg_corr


def test_quanto_greeks_expose_fx_delta_and_corr_sensitivity():
    _, greeks = QuantoInverseOptionPricer.calculate_price_and_greeks(
        S=55000.0,
        K=50000.0,
        T=0.3,
        r=0.02,
        sigma=0.55,
        option_type="call",
        fx_rate=1.35,
        sigma_fx=0.7,
        rho=0.4,
    )

    assert greeks.fx_delta < 0.0
    assert greeks.corr_sensitivity < 0.0
    assert greeks.quanto_adjustment > 0.0


def test_quanto_hedger_builds_near_delta_neutral_plan():
    _, greeks = QuantoInverseOptionPricer.calculate_price_and_greeks(
        S=50000.0,
        K=52000.0,
        T=0.2,
        r=0.03,
        sigma=0.58,
        option_type="put",
        fx_rate=1.1,
        sigma_fx=0.65,
        rho=0.25,
    )

    plan = QuantoInverseHedger.build_hedge_plan(
        greeks=greeks,
        position_size=120.0,
        spot_price=50000.0,
        fx_rate=1.1,
        spot_lot_size=0.0,
        fx_lot_size=0.0,
    )

    assert abs(plan.residual_spot_delta) < 1e-10
    assert abs(plan.residual_fx_delta) < 1e-10
    assert plan.spot_notional_usd >= 0.0
    assert plan.fx_notional_settlement >= 0.0


def test_quanto_input_validation():
    with pytest.raises(ValueError):
        QuantoInverseOptionPricer.calculate_price(
            50000.0, 50000.0, 0.25, 0.03, 0.6, "call", fx_rate=0.0, sigma_fx=0.5, rho=0.2
        )

    with pytest.raises(ValueError):
        QuantoInverseOptionPricer.calculate_price(
            50000.0, 50000.0, 0.25, 0.03, 0.6, "call", fx_rate=1.2, sigma_fx=-0.1, rho=0.2
        )

    with pytest.raises(ValueError):
        QuantoInverseOptionPricer.calculate_price(
            50000.0, 50000.0, 0.25, 0.03, 0.6, "call", fx_rate=1.2, sigma_fx=0.5, rho=1.2
        )
