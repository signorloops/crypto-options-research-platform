"""
Experimental inverse-power option pricer.

This module provides a Monte Carlo baseline for inverse-power payoff family:
- Call payoff: max(0, K^{-p} - S_T^{-p})
- Put payoff:  max(0, S_T^{-p} - K^{-p})

When p=1, it degenerates to classic inverse-option payoff shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class InversePowerGreeks:
    """Finite-difference Greeks for inverse-power option."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class InversePowerOptionPricer:
    """Monte Carlo baseline pricer for inverse-power options."""

    EPS = 1e-12

    @staticmethod
    def _validate_inputs(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        power: float,
        n_paths: int,
    ) -> None:
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if T < 0:
            raise ValueError("T must be non-negative")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if power <= 0:
            raise ValueError("power must be positive")
        if n_paths < 1024:
            raise ValueError("n_paths must be >= 1024 for stable Monte Carlo")
        if not np.isfinite([S, K, T, r, sigma, power]).all():
            raise ValueError("all numeric inputs must be finite")

    @staticmethod
    def _generate_normals(n_paths: int, seed: int | None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        half = n_paths // 2
        z = rng.standard_normal(half)
        normals = np.concatenate([z, -z])
        if normals.size < n_paths:
            normals = np.concatenate([normals, rng.standard_normal(1)])
        return normals[:n_paths]

    @staticmethod
    def _terminal_prices(
        S: float,
        T: float,
        r: float,
        sigma: float,
        normals: np.ndarray,
    ) -> np.ndarray:
        if T < InversePowerOptionPricer.EPS:
            return np.full_like(normals, fill_value=S, dtype=float)
        drift = (r - 0.5 * sigma**2) * T
        diff = sigma * np.sqrt(T) * normals
        return S * np.exp(drift + diff)

    @staticmethod
    def _payoff(
        ST: np.ndarray,
        K: float,
        power: float,
        option_type: Literal["call", "put"],
    ) -> np.ndarray:
        inv_st_p = np.power(np.maximum(ST, InversePowerOptionPricer.EPS), -power)
        inv_k_p = K ** (-power)
        if option_type == "call":
            return np.maximum(inv_k_p - inv_st_p, 0.0)
        if option_type == "put":
            return np.maximum(inv_st_p - inv_k_p, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    @staticmethod
    def calculate_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        power: float = 1.0,
        n_paths: int = 100_000,
        seed: int | None = 42,
        normals: np.ndarray | None = None,
    ) -> float:
        """Price inverse-power option with Monte Carlo."""
        InversePowerOptionPricer._validate_inputs(S, K, T, r, sigma, power, n_paths)

        if T < InversePowerOptionPricer.EPS:
            intrinsic = InversePowerOptionPricer._payoff(
                np.array([S], dtype=float), K, power, option_type
            )
            return float(intrinsic[0])

        if normals is None:
            normals = InversePowerOptionPricer._generate_normals(n_paths, seed)
        st = InversePowerOptionPricer._terminal_prices(S, T, r, sigma, normals)
        payoff = InversePowerOptionPricer._payoff(st, K, power, option_type)
        return float(np.exp(-r * T) * np.mean(payoff))

    @staticmethod
    def calculate_price_and_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        power: float = 1.0,
        n_paths: int = 100_000,
        seed: int | None = 42,
        bump_rel: float = 1e-3,
    ) -> tuple[float, InversePowerGreeks]:
        """Return price and finite-difference Greeks using common random numbers."""
        InversePowerOptionPricer._validate_inputs(S, K, T, r, sigma, power, n_paths)

        normals = InversePowerOptionPricer._generate_normals(n_paths=n_paths, seed=seed)

        price = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )

        ds = max(S * bump_rel, 1e-6)
        dv = max(sigma * bump_rel, 1e-6)
        dr = max(abs(r) * bump_rel, 1e-6)
        dt = max(T * bump_rel, 1e-6)

        p_up_s = InversePowerOptionPricer.calculate_price(
            S=S + ds,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )
        p_dn_s = InversePowerOptionPricer.calculate_price(
            S=max(S - ds, InversePowerOptionPricer.EPS),
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )

        p_up_v = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma + dv,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )
        p_dn_v = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=max(sigma - dv, 0.0),
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )

        p_up_r = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r + dr,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )
        p_dn_r = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T,
            r=r - dr,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )

        p_up_t = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=T + dt,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )
        p_dn_t = InversePowerOptionPricer.calculate_price(
            S=S,
            K=K,
            T=max(T - dt, 0.0),
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            seed=None,
            normals=normals,
        )

        delta = (p_up_s - p_dn_s) / (2.0 * ds)
        gamma = (p_up_s - 2.0 * price + p_dn_s) / (ds**2)
        vega = (p_up_v - p_dn_v) / (2.0 * dv)
        rho = (p_up_r - p_dn_r) / (2.0 * dr)
        theta = -(p_up_t - p_dn_t) / (2.0 * dt)

        greeks = InversePowerGreeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
        )
        return float(price), greeks
