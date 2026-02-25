"""Volatility surface fitting helpers.

This module keeps fitting logic outside the facade class to reduce coupling and
method complexity in ``research.volatility.implied``.
"""

from __future__ import annotations

from typing import Dict, Type

import numpy as np

try:
    from scipy.optimize import minimize

    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False


def svi_total_variance(k: np.ndarray, params) -> np.ndarray:
    return params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )


def ssvi_total_variance(k: np.ndarray, theta: np.ndarray, params) -> np.ndarray:
    theta_safe = np.maximum(theta, 1e-10)
    phi = params.eta * np.power(theta_safe, -params.lam)
    term = phi * k + params.rho
    inner = np.maximum(term * term + 1.0 - params.rho**2, 1e-12)
    return 0.5 * theta_safe * (1.0 + params.rho * phi * k + np.sqrt(inner))


def fit_ssvi(surface, ssvi_params_cls: Type, expiry_tol: float = 0.01):
    if len(surface.points) < 8:
        return None

    expiries = sorted({float(p.expiry) for p in surface.points})
    atm_expiries = []
    atm_total_vars = []

    for expiry in expiries:
        expiry_points = [p for p in surface.points if abs(p.expiry - expiry) <= expiry_tol]
        if len(expiry_points) < 3:
            continue

        k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
        w = np.array([p.volatility**2 * p.expiry for p in expiry_points], dtype=float)
        w = np.maximum(w, 1e-10)
        weights = np.exp(-8.0 * np.abs(k))
        theta_atm = float(np.average(w, weights=weights))
        atm_expiries.append(float(expiry))
        atm_total_vars.append(theta_atm)

    if len(atm_expiries) < 2:
        return None

    x_exp = np.array(atm_expiries, dtype=float)
    y_theta = np.maximum(np.array(atm_total_vars, dtype=float), 1e-10)
    y_theta = np.maximum.accumulate(y_theta)

    def theta_of_t(t: np.ndarray) -> np.ndarray:
        return np.interp(t, x_exp, y_theta, left=y_theta[0], right=y_theta[-1])

    if HAS_SCIPY_OPT:
        k_obs = np.array([p.log_moneyness for p in surface.points], dtype=float)
        t_obs = np.array([float(p.expiry) for p in surface.points], dtype=float)
        w_obs = np.array([p.volatility**2 * p.expiry for p in surface.points], dtype=float)
        w_obs = np.maximum(w_obs, 1e-10)

        def objective(x: np.ndarray) -> float:
            params = ssvi_params_cls(rho=float(x[0]), eta=float(x[1]), lam=float(x[2]))
            theta = theta_of_t(t_obs)
            w_model = ssvi_total_variance(k_obs, theta, params)
            if not np.all(np.isfinite(w_model)):
                return 1e9

            eta_upper = 2.0 / (np.sqrt(np.max(theta)) * (1.0 + abs(params.rho) + 1e-8))
            penalty = 0.0
            if params.eta > eta_upper:
                penalty += 1e5 * (params.eta - eta_upper) ** 2
            return float(np.mean((w_model - w_obs) ** 2) + penalty)

        init = np.array([-0.2, 1.0, 0.2], dtype=float)
        bounds = [(-0.999, 0.999), (1e-4, 10.0), (0.0, 0.5)]
        result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
        x = result.x if result.success else init
        rho, eta, lam = float(x[0]), float(x[1]), float(x[2])
    else:
        rho, eta, lam = -0.2, 1.0, 0.2

    eta_upper = 2.0 / (np.sqrt(np.max(y_theta)) * (1.0 + abs(rho) + 1e-8))
    eta = float(np.clip(eta, 1e-4, eta_upper))

    surface._ssvi_params = ssvi_params_cls(rho=rho, eta=eta, lam=lam)
    surface._ssvi_atm_expiries = x_exp
    surface._ssvi_atm_total_vars = y_theta
    return surface._ssvi_params


def fit_svi(surface, expiry: float, svi_params_cls: Type, expiry_tol: float = 0.01):
    expiry_points = [p for p in surface.points if abs(p.expiry - expiry) <= expiry_tol]
    if len(expiry_points) < 5:
        return None

    k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
    w = np.array([p.volatility**2 * p.expiry for p in expiry_points], dtype=float)
    w = np.maximum(w, 1e-8)

    if not HAS_SCIPY_OPT:
        params = svi_params_cls(
            a=float(np.min(w) * 0.8),
            b=float(max(1e-4, np.std(w))),
            rho=0.0,
            m=float(np.median(k)),
            sigma=0.1,
        )
        surface._svi_params[float(expiry)] = params
        return params

    def objective(x: np.ndarray) -> float:
        p = svi_params_cls(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
        model_w = svi_total_variance(k, p)
        if np.any(model_w <= 0):
            return 1e9
        return float(np.mean((model_w - w) ** 2))

    init = np.array(
        [float(np.min(w) * 0.8), float(max(1e-4, np.std(w))), 0.0, float(np.median(k)), 0.1]
    )
    bounds = [
        (-5.0, 5.0),
        (1e-8, 10.0),
        (-0.999, 0.999),
        (-5.0, 5.0),
        (1e-6, 5.0),
    ]
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    x = result.x if result.success else init
    params = svi_params_cls(
        a=float(x[0]), b=float(x[1]), rho=float(x[2]), m=float(x[3]), sigma=float(x[4])
    )
    surface._svi_params[float(expiry)] = params
    return params


def fit_all_svi(surface) -> Dict[float, object]:
    surface._svi_params = {}
    for expiry in sorted({float(p.expiry) for p in surface.points}):
        surface.fit_svi(expiry)
    return surface._svi_params
