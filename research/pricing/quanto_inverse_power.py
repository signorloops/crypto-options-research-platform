"""Quanto-inverse-power option pricing utilities.

This module extends inverse-power option pricing with a lightweight quanto
adjustment for settlement-currency mismatch risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from research.pricing.inverse_power_options import InversePowerOptionPricer


@dataclass
class QuantoInversePowerGreeks:
    """Greeks for quanto-inverse-power options in settlement currency."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    fx_delta: float
    corr_sensitivity: float
    quanto_adjustment: float


class QuantoInversePowerOptionPricer:
    """Price and risk analytics for quanto-inverse-power options.

    Approximation:
    - Start from inverse-power option value in collateral currency.
    - Convert by FX and apply quanto factor exp(-rho*sigma_s*sigma_fx*T).
    """

    EPS = 1e-12

    @staticmethod
    def _validate_quanto_inputs(fx_rate: float, sigma_fx: float, rho: float) -> None:
        if not np.isfinite(fx_rate) or fx_rate <= 0:
            raise ValueError("fx_rate must be positive and finite")
        if not np.isfinite(sigma_fx) or sigma_fx < 0:
            raise ValueError("sigma_fx must be non-negative and finite")
        if not np.isfinite(rho) or rho < -1.0 or rho > 1.0:
            raise ValueError("rho must be in [-1, 1]")

    @staticmethod
    def _quanto_factor(
        T: float,
        fx_rate: float,
        sigma_spot: float,
        sigma_fx: float,
        rho: float,
    ) -> float:
        t_eff = max(float(T), 0.0)
        adjustment = float(np.exp(-rho * sigma_spot * sigma_fx * t_eff))
        return adjustment / fx_rate

    @staticmethod
    def calculate_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        fx_rate: float,
        sigma_fx: float,
        rho: float,
        power: float = 1.0,
        n_paths: int = 100_000,
        seed: int | None = 42,
    ) -> float:
        """Calculate quanto-inverse-power option price in settlement currency."""
        QuantoInversePowerOptionPricer._validate_quanto_inputs(fx_rate, sigma_fx, rho)

        base_price = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=seed,
        )
        factor = QuantoInversePowerOptionPricer._quanto_factor(
            T=T,
            fx_rate=fx_rate,
            sigma_spot=sigma,
            sigma_fx=sigma_fx,
            rho=rho,
        )
        return float(max(0.0, base_price * factor))

    @staticmethod
    def calculate_price_and_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        fx_rate: float,
        sigma_fx: float,
        rho: float,
        power: float = 1.0,
        n_paths: int = 100_000,
        seed: int | None = 42,
        bump_rel: float = 1e-3,
    ) -> Tuple[float, QuantoInversePowerGreeks]:
        """Calculate quanto-inverse-power price and Greeks."""
        QuantoInversePowerOptionPricer._validate_quanto_inputs(fx_rate, sigma_fx, rho)

        base_price, base_greeks = InversePowerOptionPricer.calculate_price_and_greeks(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=seed,
            bump_rel=bump_rel,
        )

        t_eff = max(float(T), 0.0)
        quanto_adjustment = float(np.exp(-rho * sigma * sigma_fx * t_eff))
        factor = quanto_adjustment / fx_rate

        price = float(max(0.0, base_price * factor))
        delta = float(base_greeks.delta * factor)
        gamma = float(base_greeks.gamma * factor)
        rho_rate = float(base_greeks.rho * factor)

        d_factor_dT = -rho * sigma * sigma_fx * factor
        theta = float(base_greeks.theta * factor + base_price * d_factor_dT)

        d_factor_d_sigma = -rho * sigma_fx * t_eff * factor
        vega = float(base_greeks.vega * factor + base_price * d_factor_d_sigma)

        fx_delta = float(-price / fx_rate)
        corr_sensitivity = float(-base_price * factor * sigma * sigma_fx * t_eff)

        greeks = QuantoInversePowerGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho_rate,
            fx_delta=fx_delta,
            corr_sensitivity=corr_sensitivity,
            quanto_adjustment=quanto_adjustment,
        )
        return price, greeks
