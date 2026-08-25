"""
Experimental inverse-power option pricer.

This module provides a Monte Carlo baseline for inverse-power payoff family:
- Call payoff: max(0, K^{-p} - S_T^{-p})
- Put payoff:  max(0, S_T^{-p} - K^{-p})

When p=1, it degenerates to classic inverse-option payoff shape.

Greek unit convention — DIFFERENT from inverse_options.py, do not mix:
- theta: annual (per unit of calendar time T, i.e. -dV/dT)
- vega:  per unit of volatility (dV/dsigma)
- rho:   per unit of risk-free rate (dV/dr)
research/pricing/inverse_options.py reports the same field names with **daily**
theta and **per-1%** vega/rho. Rescale before combining the two modules'
outputs (annual theta = daily theta * 365; per-unit vega = per-1% vega * 100).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np


@dataclass
class InversePowerQuote:
    """Input quote container for inverse-power pricing."""

    spot: float
    strike: float
    maturity: float
    rate: float
    sigma: float
    option_type: Literal["call", "put"]
    power: float = 1.0


@dataclass
class InversePowerGreeks:
    """Finite-difference Greeks for inverse-power option.

    Unit convention (explicitly declared so it cannot be confused with
    ``InverseGreeks``): theta is **annual** (-dV/dT with T in years) and vega
    is **per unit of volatility** (dV/dsigma). ``InverseGreeks`` in
    research/pricing/inverse_options.py reports daily theta and per-1% vega
    under the same field names — rescale before mixing the two.
    """

    # Unit metadata mirroring InverseGreeks.THETA_UNIT / VEGA_UNIT.
    THETA_UNIT: ClassVar[str] = "annual"
    VEGA_UNIT: ClassVar[str] = "per_unit_vol"
    RHO_UNIT: ClassVar[str] = "per_unit_rate"

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True)
class _InversePowerFDBase:
    """Shared base inputs for finite-difference pricing with common normals."""

    normals: np.ndarray
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: Literal["call", "put"]
    power: float
    n_paths: int


@dataclass(frozen=True)
class _InversePowerFDBumps:
    """Finite-difference bump sizes."""

    ds: float
    dv: float
    dr: float
    dt: float


@dataclass(frozen=True)
class _InversePowerFDPrices:
    """Finite-difference prices for each bumped dimension.

    ``vega_forward_only`` / ``theta_forward_only`` record that the down-bump
    collapsed onto a boundary (sigma=0 or T=0, where the pricer switches to the
    intrinsic branch). In that case the realized down-step is smaller than the
    nominal ``2 * bump`` denominator, so the Greek must be computed as a
    one-sided forward difference against the unbumped price.
    """

    p_up_s: float
    p_dn_s: float
    p_up_v: float
    p_dn_v: float
    p_up_r: float
    p_dn_r: float
    p_up_t: float
    p_dn_t: float
    vega_forward_only: bool = False
    theta_forward_only: bool = False


class InversePowerOptionPricer:
    """Monte Carlo baseline pricer for inverse-power options.

    Greeks are raw finite-difference derivatives: theta annual (-dV/dT), vega
    per unit vol (dV/dsigma), rho per unit rate (dV/dr). See
    ``InversePowerGreeks`` for the unit metadata.
    """

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
        # Inverse options pay in BTC, so the price must be taken under the
        # stock-numeraire measure: V = exp(-rT) * E^Q[payoff * S_T / S_0].
        # The previous implementation used plain E^Q[payoff], which is wrong
        # even at r = 0 and contradicts the closed-form InverseOptionPricer.
        likelihood_ratio = st / S
        return float(np.exp(-r * T) * np.mean(payoff * likelihood_ratio))

    @staticmethod
    def calculate_price_from_quote(
        quote: InversePowerQuote,
        n_paths: int = 100_000,
        seed: int | None = 42,
    ) -> float:
        """Price from an InversePowerQuote container."""
        return InversePowerOptionPricer.calculate_price(
            S=quote.spot,
            K=quote.strike,
            T=quote.maturity,
            r=quote.rate,
            sigma=quote.sigma,
            option_type=quote.option_type,
            power=quote.power,
            n_paths=n_paths,
            seed=seed,
        )

    @staticmethod
    def _price_with_normals(
        *,
        normals: np.ndarray,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        power: float,
        n_paths: int,
    ) -> float:
        return InversePowerOptionPricer.calculate_price(
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

    @staticmethod
    def _finite_difference_bumps(
        *, S: float, T: float, r: float, sigma: float, bump_rel: float
    ) -> _InversePowerFDBumps:
        ds = max(S * bump_rel, 1e-6)
        dv = max(sigma * bump_rel, 1e-6)
        dr = max(abs(r) * bump_rel, 1e-6)
        dt = max(T * bump_rel, 1e-6)
        return _InversePowerFDBumps(ds=ds, dv=dv, dr=dr, dt=dt)

    @staticmethod
    def _finite_difference_prices(
        *,
        base: _InversePowerFDBase,
        bumps: _InversePowerFDBumps,
    ) -> _InversePowerFDPrices:
        base_kwargs = {
            "normals": base.normals,
            "S": base.S,
            "K": base.K,
            "T": base.T,
            "r": base.r,
            "sigma": base.sigma,
            "option_type": base.option_type,
            "power": base.power,
            "n_paths": base.n_paths,
        }

        def eval_price(**overrides: float) -> float:
            return InversePowerOptionPricer._price_with_normals(
                **{**base_kwargs, **overrides}
            )

        return _InversePowerFDPrices(
            p_up_s=eval_price(S=base.S + bumps.ds),
            p_dn_s=eval_price(S=max(base.S - bumps.ds, InversePowerOptionPricer.EPS)),
            p_up_v=eval_price(sigma=base.sigma + bumps.dv),
            p_dn_v=eval_price(sigma=max(base.sigma - bumps.dv, 0.0)),
            p_up_r=eval_price(r=base.r + bumps.dr),
            p_dn_r=eval_price(r=base.r - bumps.dr),
            p_up_t=eval_price(T=base.T + bumps.dt),
            p_dn_t=eval_price(T=max(base.T - bumps.dt, 0.0)),
            vega_forward_only=base.sigma - bumps.dv <= 0.0,
            theta_forward_only=base.T - bumps.dt <= 0.0,
        )

    @staticmethod
    def _build_fd_base(
        *,
        normals: np.ndarray,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        power: float,
        n_paths: int,
    ) -> _InversePowerFDBase:
        return _InversePowerFDBase(
            normals=normals,
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
        )

    @staticmethod
    def _fd_price_context(
        *, normals: np.ndarray, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Literal["call", "put"], power: float, n_paths: int, bump_rel: float,
    ) -> tuple[float, _InversePowerFDBase, _InversePowerFDBumps]:
        price = InversePowerOptionPricer._price_with_normals(normals=normals, S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, power=power, n_paths=n_paths)
        bumps = InversePowerOptionPricer._finite_difference_bumps(
            S=S,
            T=T,
            r=r,
            sigma=sigma,
            bump_rel=bump_rel,
        ); fd_base = InversePowerOptionPricer._build_fd_base(
            normals=normals,
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
        )
        return price, fd_base, bumps

    @staticmethod
    def _finite_difference_greeks(
        *,
        price: float,
        bumps: _InversePowerFDBumps,
        fd_prices: _InversePowerFDPrices,
    ) -> InversePowerGreeks:
        delta = (fd_prices.p_up_s - fd_prices.p_dn_s) / (2.0 * bumps.ds)
        gamma = (fd_prices.p_up_s - 2.0 * price + fd_prices.p_dn_s) / (bumps.ds**2)
        if fd_prices.vega_forward_only:
            # The sigma down-bump collapsed onto the sigma=0 intrinsic branch:
            # the realized step is `bumps.dv`, not `2 * bumps.dv`, so a
            # symmetric-difference quotient would silently halve (and bias) vega.
            vega = (fd_prices.p_up_v - price) / bumps.dv
        else:
            vega = (fd_prices.p_up_v - fd_prices.p_dn_v) / (2.0 * bumps.dv)
        rho = (fd_prices.p_up_r - fd_prices.p_dn_r) / (2.0 * bumps.dr)
        if fd_prices.theta_forward_only:
            # The T down-bump collapsed onto the T=0 intrinsic branch: the
            # realized step is `bumps.dt`, not `2 * bumps.dt`, so theta must use
            # a one-sided forward difference to stay unbiased.
            theta = -(fd_prices.p_up_t - price) / bumps.dt
        else:
            theta = -(fd_prices.p_up_t - fd_prices.p_dn_t) / (2.0 * bumps.dt)
        return InversePowerGreeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
        )

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
        price, fd_base, bumps = InversePowerOptionPricer._fd_price_context(
            normals=normals,
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            power=power,
            n_paths=n_paths,
            bump_rel=bump_rel,
        )
        fd_prices = InversePowerOptionPricer._finite_difference_prices(base=fd_base, bumps=bumps)
        greeks = InversePowerOptionPricer._finite_difference_greeks(
            price=price,
            bumps=bumps,
            fd_prices=fd_prices,
        )
        return float(price), greeks
