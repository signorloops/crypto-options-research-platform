"""
Utilities for implied-volatility surface stability audits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.volatility.implied import VolatilitySurface


def _skew_metrics(skew: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Compute simple smoothness diagnostics for a skew curve."""
    if len(skew) < 2:
        return {
            "n_points": float(len(skew)),
            "max_adjacent_jump": 0.0,
            "mean_abs_adjacent_jump": 0.0,
            "curvature_l1": 0.0,
        }

    vols = np.array([float(v) for _, v in skew], dtype=float)
    jumps = np.diff(vols)
    curvature = np.diff(jumps) if len(jumps) > 1 else np.array([], dtype=float)
    return {
        "n_points": float(len(skew)),
        "max_adjacent_jump": float(np.max(np.abs(jumps))),
        "mean_abs_adjacent_jump": float(np.mean(np.abs(jumps))),
        "curvature_l1": float(np.mean(np.abs(curvature))) if curvature.size else 0.0,
    }


def _safe_mean(values: List[float]) -> float:
    """Return mean value with empty-list guard."""
    if not values:
        return 0.0
    return float(np.mean(values))


def audit_surface_stability(
    surface: VolatilitySurface,
    expiries: Optional[Sequence[float]] = None,
    short_maturity_threshold: float = 14.0 / 365.0,
) -> Dict[str, object]:
    """
    Audit skew smoothness before/after short-maturity stabilization.

    Returns a dict suitable for JSON/Markdown reporting.
    """
    if expiries is None:
        expiries = sorted({float(point.expiry) for point in surface.points})
    expiry_list = [float(expiry) for expiry in expiries]

    rows: List[Dict[str, object]] = []
    for expiry in expiry_list:
        raw_skew = surface.get_skew(
            expiry=expiry,
            stabilize_short_maturity=False,
            short_maturity_threshold=short_maturity_threshold,
        )
        stabilized_skew = surface.get_skew(
            expiry=expiry,
            stabilize_short_maturity=True,
            short_maturity_threshold=short_maturity_threshold,
        )
        raw = _skew_metrics(raw_skew)
        stabilized = _skew_metrics(stabilized_skew)
        is_short = bool(expiry <= short_maturity_threshold)

        rows.append(
            {
                "expiry_years": float(expiry),
                "expiry_days": float(expiry * 365.0),
                "is_short_maturity": is_short,
                "n_points": int(raw["n_points"]),
                "raw_max_adjacent_jump": float(raw["max_adjacent_jump"]),
                "stabilized_max_adjacent_jump": float(stabilized["max_adjacent_jump"]),
                "raw_mean_adjacent_jump": float(raw["mean_abs_adjacent_jump"]),
                "stabilized_mean_adjacent_jump": float(stabilized["mean_abs_adjacent_jump"]),
                "raw_curvature_l1": float(raw["curvature_l1"]),
                "stabilized_curvature_l1": float(stabilized["curvature_l1"]),
                "max_jump_reduction": float(
                    raw["max_adjacent_jump"] - stabilized["max_adjacent_jump"]
                ),
                "mean_jump_reduction": float(
                    raw["mean_abs_adjacent_jump"] - stabilized["mean_abs_adjacent_jump"]
                ),
            }
        )

    no_arbitrage = surface.validate_no_arbitrage()
    butterfly = no_arbitrage.get("butterfly", {})
    calendar = no_arbitrage.get("calendar", {})
    butterfly_violations = sum(
        len(detail.get("violations", []))
        for detail in butterfly.values()
        if isinstance(detail, dict)
    )
    calendar_violations = len(calendar.get("violations", []))
    short_rows = [row for row in rows if bool(row["is_short_maturity"])]

    summary = {
        "n_expiries": len(rows),
        "short_maturity_buckets": len(short_rows),
        "avg_max_jump_reduction_short": _safe_mean(
            [float(row["max_jump_reduction"]) for row in short_rows]
        ),
        "avg_mean_jump_reduction_short": _safe_mean(
            [float(row["mean_jump_reduction"]) for row in short_rows]
        ),
        "max_reduction_short": (
            max(float(row["max_jump_reduction"]) for row in short_rows) if short_rows else 0.0
        ),
        "no_arbitrage": bool(no_arbitrage.get("no_arbitrage", False)),
        "butterfly_violations": int(butterfly_violations),
        "calendar_violations": int(calendar_violations),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "short_maturity_threshold_days": float(short_maturity_threshold * 365.0),
        "summary": summary,
        "expiries": rows,
        "no_arbitrage_checks": no_arbitrage,
    }
