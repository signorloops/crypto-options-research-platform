"""Tests for inverse-power option pricer."""

import numpy as np

from research.pricing.inverse_options import InverseOptionPricer
from research.pricing.inverse_power_options import InversePowerOptionPricer, InversePowerQuote


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


def test_price_from_quote_matches_direct_call():
    quote = InversePowerQuote(
        spot=50000.0,
        strike=51000.0,
        maturity=30.0 / 365.0,
        rate=0.02,
        sigma=0.5,
        option_type="call",
        power=1.3,
    )
    from_quote = InversePowerOptionPricer.calculate_price_from_quote(
        quote,
        n_paths=20000,
        seed=17,
    )
    direct = InversePowerOptionPricer.calculate_price(
        S=quote.spot,
        K=quote.strike,
        T=quote.maturity,
        r=quote.rate,
        sigma=quote.sigma,
        option_type=quote.option_type,
        power=quote.power,
        n_paths=20000,
        seed=17,
    )
    assert np.isclose(from_quote, direct)


def test_theta_uses_forward_stencil_when_t_bump_collapses():
    """Theta near T=0 must use a forward (one-sided) difference.

    Previously the down-bump landed on the T=0 intrinsic branch while the
    denominator stayed 2*dt, so theta was biased. The dt bump floors at 1e-6,
    so any T <= 1e-6 collapses. Compare the short-maturity FD theta against a
    reference computed with a small symmetric stencil that stays in T > 0.
    """
    S, K, r, sigma, power = 50000.0, 52000.0, 0.02, 0.5, 1.2

    for T in (5e-7, 9e-7):
        normals = InversePowerOptionPricer._generate_normals(n_paths=20000, seed=3)

        def price_at(t: float) -> float:
            return InversePowerOptionPricer.calculate_price(
                S=S, K=K, T=t, r=r, sigma=sigma, option_type="put",
                power=power, seed=None, normals=normals,
            )

        dt = max(T * 1e-3, 1e-6)
        assert T - dt <= 0.0  # precondition: the down-bump collapses onto T=0

        base = price_at(T)
        bumped, greeks = InversePowerOptionPricer.calculate_price_and_greeks(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type="put",
            power=power, n_paths=20000, seed=3, bump_rel=1e-3,
        )
        assert np.isclose(bumped, base)

        # Reference: symmetric stencil with steps small enough to stay in T > 0.
        h = T / 10.0
        reference_theta = -((price_at(T + h) - price_at(T - h)) / (2.0 * h))
        forward_only = -(price_at(T + dt) - base) / dt

        assert np.isclose(greeks.theta, forward_only, rtol=1e-9, atol=0.0)
        # The old biased quotient (denominator 2*dt with the collapsed stencil)
        # would have differed; verify the fix is meaningfully closer to truth.
        biased = -(price_at(T + dt) - price_at(0.0)) / (2.0 * dt)
        assert abs(greeks.theta - reference_theta) < abs(biased - reference_theta)


def test_vega_uses_forward_stencil_when_sigma_bump_collapses():
    """Vega near sigma=0 must use a forward (one-sided) difference.

    With sigma below the dv bump floor (1e-6), the down-bump clamps to the
    sigma=0 branch while the denominator stays 2*dv, roughly halving the
    reported vega. Note the vega magnitudes here are ~1e-14, so all
    comparisons use atol=0 — np.isclose's default atol=1e-8 would trivially
    pass and hide the bug.
    """
    S, K, T, r, power = 50000.0, 52000.0, 30.0 / 365.0, 0.02, 1.2
    sigma = 5e-7  # below the dv floor, so sigma - dv collapses onto 0
    normals = InversePowerOptionPricer._generate_normals(n_paths=20000, seed=5)

    def price_at(vol: float) -> float:
        return InversePowerOptionPricer.calculate_price(
            S=S, K=K, T=T, r=r, sigma=vol, option_type="put",
            power=power, seed=None, normals=normals,
        )

    base = price_at(sigma)
    bumped, greeks = InversePowerOptionPricer.calculate_price_and_greeks(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type="put",
        power=power, n_paths=20000, seed=5, bump_rel=1e-3,
    )
    assert np.isclose(bumped, base)

    dv = max(sigma * 1e-3, 1e-6)
    assert sigma - dv <= 0.0  # precondition: down-bump collapses
    forward_only = (price_at(sigma + dv) - base) / dv
    biased = (price_at(sigma + dv) - price_at(0.0)) / (2.0 * dv)

    assert np.isclose(greeks.vega, forward_only, rtol=1e-6, atol=0.0)
    # The old symmetric quotient with the collapsed stencil differs materially.
    assert not np.isclose(greeks.vega, biased, rtol=0.1, atol=0.0)


def test_greek_unit_metadata_declared_and_units_are_annual_per_unit():
    """InversePowerGreeks reports ANNUAL theta and PER-UNIT-VOL vega.

    This is the opposite of ``InverseGreeks`` (daily / per-1%), which is why
    both classes expose THETA_UNIT/VEGA_UNIT classvars: the two modules share
    field names while reporting different units, and consumers must rescale
    before combining. Guarded here so the convention cannot silently drift.
    """
    from research.pricing.inverse_power_options import InversePowerGreeks

    assert InversePowerGreeks.THETA_UNIT == "annual"
    assert InversePowerGreeks.VEGA_UNIT == "per_unit_vol"
    assert InversePowerGreeks.RHO_UNIT == "per_unit_rate"

    _, greeks = InversePowerOptionPricer.calculate_price_and_greeks(
        S=50000.0,
        K=50000.0,
        T=30.0 / 365.0,
        r=0.02,
        sigma=0.6,
        option_type="call",
        power=1.0,
        n_paths=20000,
        seed=42,
    )

    # Annual theta: same order of magnitude as -dV/dT (not /365 like the
    # daily convention used by InverseOptionPricer).
    assert greeks.theta < 0.0
    assert abs(greeks.theta) > 1e-8

    # Per-unit-vol vega: bumping sigma by 0.01 must move the price by
    # approximately vega * 0.01 (a per-1% vega would equal that move exactly).
    dv = 0.01
    p_up = InversePowerOptionPricer.calculate_price(
        S=50000.0, K=50000.0, T=30.0 / 365.0, r=0.02, sigma=0.6 + dv,
        option_type="call", power=1.0, n_paths=20000, seed=42,
    )
    p_dn = InversePowerOptionPricer.calculate_price(
        S=50000.0, K=50000.0, T=30.0 / 365.0, r=0.02, sigma=0.6 - dv,
        option_type="call", power=1.0, n_paths=20000, seed=42,
    )
    per_unit_estimate = (p_up - p_dn) / (2.0 * dv)
    assert np.isclose(greeks.vega, per_unit_estimate, rtol=0.15)
