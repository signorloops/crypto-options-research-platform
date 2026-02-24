"""Quanto-inverse option pricing utilities.

This module extends inverse option pricing with a lightweight quanto adjustment
for settlement-currency mismatch risk.
"""
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from research.pricing.inverse_options import InverseOptionPricer


@dataclass
class QuantoInverseGreeks:
    """Greeks for a quanto-inverse option quoted in settlement currency."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    fx_delta: float
    corr_sensitivity: float
    quanto_adjustment: float


class QuantoInverseOptionPricer:
    """Price and risk analytics for quanto-inverse options.

    Pricing approximation:
    - Start from inverse option value in base collateral currency.
    - Convert into settlement currency by dividing by FX rate.
    - Apply a drift-style quanto adjustment factor exp(-rho*sigma_s*sigma_fx*T).
    """

    DAYS_PER_YEAR = 365.0

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
    ) -> float:
        """Calculate quanto-inverse option price in settlement currency."""
        QuantoInverseOptionPricer._validate_quanto_inputs(fx_rate, sigma_fx, rho)

        base_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
        factor = QuantoInverseOptionPricer._quanto_factor(T, fx_rate, sigma, sigma_fx, rho)
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
    ) -> Tuple[float, QuantoInverseGreeks]:
        """Calculate price and Greeks for quanto-inverse option."""
        QuantoInverseOptionPricer._validate_quanto_inputs(fx_rate, sigma_fx, rho)

        base_price, base_greeks = InverseOptionPricer.calculate_price_and_greeks(
            S, K, T, r, sigma, option_type
        )

        t_eff = max(float(T), 0.0)
        quanto_adjustment = float(np.exp(-rho * sigma * sigma_fx * t_eff))
        factor = quanto_adjustment / fx_rate

        price = float(max(0.0, base_price * factor))
        delta = float(base_greeks.delta * factor)
        gamma = float(base_greeks.gamma * factor)
        rho_rate = float(base_greeks.rho * factor)

        # Convert base daily theta into settlement currency and add quanto drift decay.
        d_factor_dT = -rho * sigma * sigma_fx * factor
        theta = float(base_greeks.theta * factor + (base_price * d_factor_dT) / QuantoInverseOptionPricer.DAYS_PER_YEAR)

        # Base vega is per 1% vol; apply chain adjustment for quanto factor sensitivity to sigma.
        d_factor_d_sigma = -rho * sigma_fx * t_eff * factor
        vega = float(base_greeks.vega * factor + base_price * d_factor_d_sigma * 0.01)

        fx_delta = float(-price / fx_rate)
        corr_sensitivity = float(-base_price * factor * sigma * sigma_fx * t_eff)

        greeks = QuantoInverseGreeks(
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

    @staticmethod
    def decompose_quanto_effect(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        fx_rate: float,
        sigma_fx: float,
        rho: float,
    ) -> Dict[str, float]:
        """Return base-vs-quanto decomposition for analysis and reporting."""
        base_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
        quanto_factor = QuantoInverseOptionPricer._quanto_factor(T, fx_rate, sigma, sigma_fx, rho)
        quanto_price = max(0.0, float(base_price * quanto_factor))
        converted_base = float(base_price / fx_rate)
        adjustment = float(quanto_price - converted_base)
        return {
            "base_price": float(base_price),
            "converted_base_price": converted_base,
            "quanto_factor": float(quanto_factor),
            "quanto_price": quanto_price,
            "quanto_adjustment": adjustment,
        }
