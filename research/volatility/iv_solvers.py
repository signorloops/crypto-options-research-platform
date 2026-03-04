"""Implied-volatility solver utilities shared by volatility modules."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from scipy.stats import norm

try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _jaeckel_iv

    HAS_JAECKEL_SOLVER = True
except ImportError:
    HAS_JAECKEL_SOLVER = False

from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)

JAECKEL_FALLBACK_EXCEPTIONS = (
    ValueError,
    TypeError,
    ArithmeticError,
    FloatingPointError,
    OverflowError,
    ZeroDivisionError,
    RuntimeError,
)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)

    return float(vega)


def _option_price_bounds(
    S: float, K: float, T: float, r: float, is_call: bool
) -> Tuple[float, float]:
    """No-arbitrage bounds for vanilla option prices."""
    discount = np.exp(-r * T)
    if is_call:
        return max(0.0, S - K * discount), S
    return max(0.0, K * discount - S), K * discount


def _bisection_bracket_outcome(
    *,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
) -> float | None:
    sigma_low, sigma_high = 0.001, 5.0
    price_low = black_scholes_price(S, K, T, r, sigma_low, is_call)
    price_high = black_scholes_price(S, K, T, r, sigma_high, is_call)
    if market_price < price_low:
        return sigma_low
    if market_price > price_high:
        return sigma_high
    return None


def _initial_sigma_guess(*, market_price: float, S: float, T: float) -> float:
    sigma0 = np.sqrt(2 * np.pi / max(T, 1e-12)) * (market_price / max(S, 1e-12))
    return float(np.clip(sigma0, 0.01, 3.0))


def _halley_sigma_update(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    diff: float,
) -> float | None:
    vega = black_scholes_vega(S, K, T, r, sigma)
    if vega < 1e-12:
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    volga = vega * d1 * d2 / max(sigma, 1e-12)
    denom = 2.0 * vega * vega - diff * volga
    step = (2.0 * diff * vega) / denom if abs(denom) > 1e-14 else diff / vega
    sigma_new = sigma - step
    return float(np.clip(sigma_new, 0.001, 5.0))


def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Solve implied volatility with bisection."""
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    if market_price <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )
    sigma_low, sigma_high = 0.001, 5.0
    bracket_outcome = _bisection_bracket_outcome(
        market_price=market_price, S=S, K=K, T=T, r=r, is_call=is_call
    )
    if bracket_outcome is not None:
        return float(bracket_outcome)

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = black_scholes_price(S, K, T, r, sigma_mid, is_call)

        if abs(price_mid - market_price) < tol:
            return float(sigma_mid)

        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return float((sigma_low + sigma_high) / 2)


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    initial_sigma: float = 0.3,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Solve implied volatility with Newton-Raphson and bisection fallback."""
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    sigma = initial_sigma

    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        diff = market_price - price

        if abs(diff) < tol:
            return float(sigma)

        vega = black_scholes_vega(S, K, T, r, sigma)

        if vega < 1e-10:
            return implied_volatility_bisection(market_price, S, K, T, r, is_call, tol, max_iter)

        sigma_new = sigma + diff / vega

        sigma_new = max(0.001, min(5.0, sigma_new))

        if abs(sigma_new - sigma) < tol:
            return float(sigma_new)

        sigma = sigma_new

    return float(sigma)


def _implied_volatility_lbr_fallback(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> float:
    """近似 Let's-Be-Rational 风格 IV 回退求解器。"""
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))

    if market_price <= 0 or T <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    sigma = _initial_sigma_guess(market_price=market_price, S=S, T=T)

    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)

        sigma_new = _halley_sigma_update(S=S, K=K, T=T, r=r, sigma=sigma, diff=diff)
        if sigma_new is None:
            break

        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new

    return implied_volatility_bisection(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=1e-8,
        max_iter=200,
    )


def implied_volatility_jaeckel(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> float:
    """Prefer Jaeckel LBR solver when available, otherwise use local fallback."""
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))

    if market_price <= 0 or T <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    if HAS_JAECKEL_SOLVER:
        flag = "c" if is_call else "p"
        try:
            iv = _jaeckel_iv(market_price, S, K, T, r, flag)
            if np.isfinite(iv):
                return float(np.clip(iv, 0.0, 5.0))
        except JAECKEL_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - fallback path
            logger.debug(
                "Jaeckel solver failed, fallback to local LBR",
                extra=log_extra(error=str(exc), strike=K, expiry=T),
            )

    return _implied_volatility_lbr_fallback(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=tol,
        max_iter=max_iter,
    )


def implied_volatility_lbr(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> float:
    """兼容入口：对外保持 lbr 命名，内部优先走 Jaeckel LBR。"""
    return implied_volatility_jaeckel(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=tol,
        max_iter=max_iter,
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    method: str = "hybrid",
) -> float:
    """Solve implied volatility using the selected method."""
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    method = method.lower()
    if method == "bisection":
        return implied_volatility_bisection(market_price, S, K, T, r, is_call)
    if method == "newton":
        return implied_volatility_newton(market_price, S, K, T, r, is_call)
    if method in {"lbr", "jaeckel"}:
        return implied_volatility_jaeckel(market_price, S, K, T, r, is_call)
    return implied_volatility_jaeckel(market_price, S, K, T, r, is_call)


__all__ = [
    "black_scholes_price",
    "black_scholes_vega",
    "implied_volatility_bisection",
    "implied_volatility_newton",
    "implied_volatility_jaeckel",
    "implied_volatility_lbr",
    "implied_volatility",
]
