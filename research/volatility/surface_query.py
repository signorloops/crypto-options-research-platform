"""Volatility surface query helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from research.volatility.iv_solvers import black_scholes_price


def vol_from_ssvi(surface, strike: float, expiry: float, ssvi_total_variance_fn) -> Optional[float]:
    if (
        surface._ssvi_params is None
        or surface._ssvi_atm_expiries is None
        or surface._ssvi_atm_total_vars is None
    ):
        surface.fit_ssvi()

    if (
        surface._ssvi_params is None
        or surface._ssvi_atm_expiries is None
        or surface._ssvi_atm_total_vars is None
    ):
        return None

    theta = np.interp(
        float(expiry),
        surface._ssvi_atm_expiries,
        surface._ssvi_atm_total_vars,
        left=float(surface._ssvi_atm_total_vars[0]),
        right=float(surface._ssvi_atm_total_vars[-1]),
    )
    theta = float(max(theta, 1e-10))
    k = np.log(strike / surface.points[0].underlying_price)
    total_var = ssvi_total_variance_fn(
        np.array([k], dtype=float),
        np.array([theta], dtype=float),
        surface._ssvi_params,
    )[0]
    vol = np.sqrt(max(float(total_var), 1e-10) / max(float(expiry), 1e-8))
    return float(np.clip(vol, 0.01, 2.0))


def vol_from_svi(surface, strike: float, expiry: float, svi_total_variance_fn) -> Optional[float]:
    if not surface._svi_params:
        surface.fit_all_svi()

    if not surface._svi_params:
        return None

    nearest_expiry = min(surface._svi_params.keys(), key=lambda x: abs(x - expiry))
    params = surface._svi_params[nearest_expiry]
    k = np.log(strike / surface.points[0].underlying_price)
    total_var = svi_total_variance_fn(np.array([k]), params)[0]
    t_eff = max(expiry, 1e-8)
    vol = np.sqrt(max(total_var, 1e-10) / t_eff)
    return float(np.clip(vol, 0.01, 2.0))


def vol_from_idw(surface, strike: float, expiry: float) -> float:
    points_array = np.array([[p.log_moneyness, p.expiry, p.volatility] for p in surface.points])

    x_target = np.log(strike / surface.points[0].underlying_price)
    y_target = expiry
    distances = np.sqrt((points_array[:, 0] - x_target) ** 2 + (points_array[:, 1] - y_target) ** 2)
    distances = np.maximum(distances, 1e-10)

    weights = 1.0 / distances
    weights /= weights.sum()
    vol = np.sum(weights * points_array[:, 2])
    return float(np.clip(vol, 0.01, 2.0))


def get_volatility(surface, strike: float, expiry: float, method: str = "linear") -> float:
    if len(surface.points) == 0:
        return 0.2

    if len(surface.points) == 1:
        return surface.points[0].volatility

    if method == "ssvi":
        ssvi_vol = surface._vol_from_ssvi(strike, expiry)
        if ssvi_vol is not None:
            return ssvi_vol

    if method == "svi":
        svi_vol = surface._vol_from_svi(strike, expiry)
        if svi_vol is not None:
            return svi_vol

    return surface._vol_from_idw(strike, expiry)


def get_skew(
    surface,
    expiry: float,
    stabilize_short_maturity: bool = False,
    short_maturity_threshold: float = 14.0 / 365.0,
    atm_anchor_window: float = 0.10,
    max_adjacent_jump: float = 0.20,
) -> List[Tuple[float, float]]:
    points = [p for p in surface.points if abs(p.expiry - expiry) < 0.01]
    if not points:
        return []

    ordered_points = sorted(points, key=lambda x: x.strike)
    skew = [(p.moneyness, p.volatility) for p in ordered_points]
    if not stabilize_short_maturity or expiry > short_maturity_threshold or len(skew) < 3:
        return skew

    moneyness = np.array([m for m, _ in skew], dtype=float)
    vols = np.array([v for _, v in skew], dtype=float)
    log_m = np.log(np.maximum(moneyness, 1e-12))

    window = max(atm_anchor_window, 1e-6)
    atm_weights = np.exp(-np.abs(log_m) / window)
    atm_anchor = float(np.average(vols, weights=atm_weights))
    smoothed = (1.0 - atm_weights) * vols + atm_weights * atm_anchor

    jump_cap = max(max_adjacent_jump, 1e-6)
    for i in range(1, len(smoothed)):
        smoothed[i] = float(
            np.clip(smoothed[i], smoothed[i - 1] - jump_cap, smoothed[i - 1] + jump_cap)
        )
    for i in range(len(smoothed) - 2, -1, -1):
        smoothed[i] = float(
            np.clip(smoothed[i], smoothed[i + 1] - jump_cap, smoothed[i + 1] + jump_cap)
        )

    return [(float(m), float(v)) for m, v in zip(moneyness, smoothed)]


def get_term_structure(surface, moneyness: float = 1.0) -> List[Tuple[float, float]]:
    points = [p for p in surface.points if abs(p.moneyness - moneyness) < 0.05]
    if not points:
        return []
    return [(p.expiry, p.volatility) for p in sorted(points, key=lambda x: x.expiry)]


def atm_volatility(surface, expiry: float) -> float:
    return surface.get_volatility(surface.points[0].underlying_price, expiry)


def summary(surface) -> Dict:
    if not surface.points:
        return {}

    vols = [p.volatility for p in surface.points]
    return {
        "n_points": len(surface.points),
        "min_vol": min(vols),
        "max_vol": max(vols),
        "mean_vol": np.mean(vols),
        "atm_vol": surface.atm_volatility(surface.points[0].expiry),
    }


def check_butterfly_arbitrage(
    surface, expiry: float, n_strikes: int = 25, tol: float = 1e-6
) -> Dict[str, object]:
    if not surface.points:
        return {"has_arbitrage": False, "violations": []}

    spot = surface.points[0].underlying_price
    risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.05"))
    strikes = np.linspace(0.6 * spot, 1.4 * spot, n_strikes)
    vols = np.array([surface.get_volatility(float(k), expiry, method="svi") for k in strikes])
    calls = np.array(
        [
            black_scholes_price(spot, float(k), expiry, risk_free_rate, float(vol), True)
            for k, vol in zip(strikes, vols)
        ]
    )

    violations = []
    for i in range(1, len(strikes) - 1):
        h1 = strikes[i] - strikes[i - 1]
        h2 = strikes[i + 1] - strikes[i]
        second_diff = (calls[i + 1] - calls[i]) / h2 - (calls[i] - calls[i - 1]) / h1
        if second_diff < -tol:
            violations.append((float(strikes[i]), float(second_diff)))
    return {"has_arbitrage": len(violations) > 0, "violations": violations}


def check_calendar_arbitrage(
    surface, moneyness_grid: Optional[List[float]] = None, tol: float = 1e-6
) -> Dict[str, object]:
    if not surface.points:
        return {"has_arbitrage": False, "violations": []}

    expiries = sorted({float(p.expiry) for p in surface.points})
    if len(expiries) < 2:
        return {"has_arbitrage": False, "violations": []}

    spot = surface.points[0].underlying_price
    if moneyness_grid is None:
        moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]

    violations = []
    for m in moneyness_grid:
        strike = spot * m
        total_vars = []
        for expiry in expiries:
            vol = surface.get_volatility(strike, expiry, method="svi")
            total_vars.append(vol * vol * expiry)
        for i in range(1, len(total_vars)):
            if total_vars[i] + tol < total_vars[i - 1]:
                violations.append((float(m), float(expiries[i - 1]), float(expiries[i])))
    return {"has_arbitrage": len(violations) > 0, "violations": violations}


def validate_no_arbitrage(surface) -> Dict[str, object]:
    expiries = sorted({float(p.expiry) for p in surface.points})
    butterfly = {str(expiry): surface.check_butterfly_arbitrage(expiry) for expiry in expiries}
    calendar = surface.check_calendar_arbitrage()
    has_bfly = any(v["has_arbitrage"] for v in butterfly.values())
    return {
        "butterfly": butterfly,
        "calendar": calendar,
        "no_arbitrage": (not has_bfly) and (not calendar["has_arbitrage"]),
    }


def detect_arbitrage_opportunities(surface) -> Dict[str, object]:
    checks = surface.validate_no_arbitrage()
    findings = []

    for expiry, detail in checks.get("butterfly", {}).items():
        if not detail.get("has_arbitrage", False):
            continue
        for strike, second_diff in detail.get("violations", []):
            findings.append(
                {
                    "type": "butterfly",
                    "expiry": float(expiry),
                    "strike": float(strike),
                    "severity": float(abs(second_diff)),
                    "detail": float(second_diff),
                }
            )

    calendar = checks.get("calendar", {})
    if calendar.get("has_arbitrage", False):
        for moneyness, t_prev, t_next in calendar.get("violations", []):
            findings.append(
                {
                    "type": "calendar",
                    "moneyness": float(moneyness),
                    "expiry_prev": float(t_prev),
                    "expiry_next": float(t_next),
                    "severity": float(abs(t_next - t_prev)),
                }
            )

    findings_sorted = sorted(findings, key=lambda x: float(x.get("severity", 0.0)), reverse=True)
    return {
        "has_arbitrage": len(findings_sorted) > 0,
        "n_findings": len(findings_sorted),
        "findings": findings_sorted,
        "summary": checks,
    }
