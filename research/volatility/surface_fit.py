"""Volatility surface fitting helpers.

This module keeps fitting logic outside the facade class to reduce coupling and
method complexity in ``research.volatility.implied``.
"""

from __future__ import annotations

from typing import Dict, Type, Tuple

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


def _extract_ssvi_atm_curve(surface, expiry_tol: float) -> tuple[np.ndarray, np.ndarray] | None:
    expiries = sorted({float(p.expiry) for p in surface.points})
    atm_expiries: list[float] = []
    atm_total_vars: list[float] = []
    for expiry in expiries:
        expiry_points = [p for p in surface.points if abs(p.expiry - expiry) <= expiry_tol]
        if len(expiry_points) < 3:
            continue
        k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
        w = np.maximum(np.array([p.volatility**2 * p.expiry for p in expiry_points], dtype=float), 1e-10)
        theta_atm = float(np.average(w, weights=np.exp(-8.0 * np.abs(k))))
        atm_expiries.append(float(expiry))
        atm_total_vars.append(theta_atm)
    if len(atm_expiries) < 2:
        return None
    x_exp = np.array(atm_expiries, dtype=float)
    y_theta = np.maximum.accumulate(np.maximum(np.array(atm_total_vars, dtype=float), 1e-10))
    return x_exp, y_theta


def _ssvi_theta_at_expiry(
    t: np.ndarray, x_exp: np.ndarray, y_theta: np.ndarray
) -> np.ndarray:
    return np.interp(t, x_exp, y_theta, left=y_theta[0], right=y_theta[-1])


def _ssvi_eta_upper_bound(max_theta: float, rho: float) -> float:
    return 2.0 / (np.sqrt(max(max_theta, 1e-10)) * (1.0 + abs(rho) + 1e-8))


def _ssvi_objective(
    x: np.ndarray,
    ssvi_params_cls: Type,
    k_obs: np.ndarray,
    t_obs: np.ndarray,
    w_obs: np.ndarray,
    x_exp: np.ndarray,
    y_theta: np.ndarray,
) -> float:
    params = ssvi_params_cls(rho=float(x[0]), eta=float(x[1]), lam=float(x[2]))
    theta = _ssvi_theta_at_expiry(t_obs, x_exp, y_theta)
    w_model = ssvi_total_variance(k_obs, theta, params)
    if not np.all(np.isfinite(w_model)):
        return 1e9

    eta_upper = _ssvi_eta_upper_bound(float(np.max(theta)), params.rho)
    penalty = 0.0
    if params.eta > eta_upper:
        penalty += 1e5 * (params.eta - eta_upper) ** 2
    return float(np.mean((w_model - w_obs) ** 2) + penalty)


def _fit_ssvi_parameters(
    surface,
    ssvi_params_cls: Type,
    x_exp: np.ndarray,
    y_theta: np.ndarray,
) -> tuple[float, float, float]:
    if not HAS_SCIPY_OPT:
        return -0.2, 1.0, 0.2

    k_obs = np.array([p.log_moneyness for p in surface.points], dtype=float)
    t_obs = np.array([float(p.expiry) for p in surface.points], dtype=float)
    w_obs = np.maximum(
        np.array([p.volatility**2 * p.expiry for p in surface.points], dtype=float),
        1e-10,
    )
    init = np.array([-0.2, 1.0, 0.2], dtype=float)
    bounds = [(-0.999, 0.999), (1e-4, 10.0), (0.0, 0.5)]
    result = minimize(
        _ssvi_objective,
        init,
        args=(ssvi_params_cls, k_obs, t_obs, w_obs, x_exp, y_theta),
        method="L-BFGS-B",
        bounds=bounds,
    )
    x = result.x if result.success else init
    return float(x[0]), float(x[1]), float(x[2])


def fit_ssvi(surface, ssvi_params_cls: Type, expiry_tol: float = 0.01):
    if len(surface.points) < 8:
        return None
    if (atm_curve := _extract_ssvi_atm_curve(surface, expiry_tol)) is None:
        return None

    x_exp, y_theta = atm_curve
    rho, eta, lam = _fit_ssvi_parameters(surface, ssvi_params_cls, x_exp, y_theta)
    eta_upper = _ssvi_eta_upper_bound(float(np.max(y_theta)), rho)
    eta = float(np.clip(eta, 1e-4, eta_upper))
    surface._ssvi_params = ssvi_params_cls(rho=rho, eta=eta, lam=lam)
    surface._ssvi_atm_expiries = x_exp
    surface._ssvi_atm_total_vars = y_theta
    return surface._ssvi_params


def _svi_observations(
    surface, expiry: float, expiry_tol: float
) -> Tuple[np.ndarray, np.ndarray] | None:
    expiry_points = [p for p in surface.points if abs(p.expiry - expiry) <= expiry_tol]
    if len(expiry_points) < 5:
        return None
    k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
    w = np.maximum(
        np.array([p.volatility**2 * p.expiry for p in expiry_points], dtype=float),
        1e-8,
    )
    return k, w


def _default_svi_params(svi_params_cls: Type, k: np.ndarray, w: np.ndarray):
    return svi_params_cls(
        a=float(np.min(w) * 0.8),
        b=float(max(1e-4, np.std(w))),
        rho=0.0,
        m=float(np.median(k)),
        sigma=0.1,
    )


def _fit_svi_vector(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    init = np.array(
        [
            float(np.min(w) * 0.8),
            float(max(1e-4, np.std(w))),
            0.0,
            float(np.median(k)),
            0.1,
        ]
    )
    bounds = [
        (-5.0, 5.0),
        (1e-8, 10.0),
        (-0.999, 0.999),
        (-5.0, 5.0),
        (1e-6, 5.0),
    ]

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = map(float, x)
        k_shift = k - m
        model_w = a + b * (rho * k_shift + np.sqrt(k_shift**2 + sigma**2))
        if np.any(model_w <= 0):
            return 1e9
        return float(np.mean((model_w - w) ** 2))

    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    return np.asarray(result.x if result.success else init, dtype=float)


def fit_svi(surface, expiry: float, svi_params_cls: Type, expiry_tol: float = 0.01):
    observations = _svi_observations(surface, expiry, expiry_tol)
    if observations is None:
        return None
    k, w = observations
    if not HAS_SCIPY_OPT:
        params = _default_svi_params(svi_params_cls, k, w)
        surface._svi_params[float(expiry)] = params
        return params
    x = _fit_svi_vector(k, w)
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
