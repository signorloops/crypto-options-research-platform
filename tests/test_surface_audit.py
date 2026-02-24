"""
Tests for IV surface stability audit helpers.
"""

from research.volatility.implied import VolatilityPoint, VolatilitySurface
from research.volatility.surface_audit import audit_surface_stability


def _build_choppy_surface() -> VolatilitySurface:
    surface = VolatilitySurface()
    spot = 100.0

    short_expiry = 7.0 / 365.0
    long_expiry = 30.0 / 365.0
    moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]
    short_vols = [0.55, 0.18, 0.60, 0.22, 0.58]
    long_vols = [0.44, 0.43, 0.42, 0.43, 0.44]

    for moneyness, vol in zip(moneyness_grid, short_vols):
        surface.add_point(
            VolatilityPoint(
                strike=spot * moneyness,
                expiry=short_expiry,
                volatility=vol,
                underlying_price=spot,
                is_call=True,
            )
        )

    for moneyness, vol in zip(moneyness_grid, long_vols):
        surface.add_point(
            VolatilityPoint(
                strike=spot * moneyness,
                expiry=long_expiry,
                volatility=vol,
                underlying_price=spot,
                is_call=True,
            )
        )

    return surface


def test_audit_surface_stability_returns_expected_structure():
    surface = _build_choppy_surface()
    report = audit_surface_stability(surface)

    assert "generated_at" in report
    assert "summary" in report
    assert "expiries" in report
    assert "no_arbitrage_checks" in report

    summary = report["summary"]
    assert summary["n_expiries"] == 2
    assert summary["short_maturity_buckets"] == 1


def test_short_maturity_stabilization_reduces_max_jump():
    surface = _build_choppy_surface()
    report = audit_surface_stability(surface)
    short_rows = [row for row in report["expiries"] if row["is_short_maturity"]]

    assert len(short_rows) == 1
    short_row = short_rows[0]
    assert short_row["stabilized_max_adjacent_jump"] < short_row["raw_max_adjacent_jump"]
    assert short_row["max_jump_reduction"] > 0.0


def test_audit_summary_contains_arbitrage_counters():
    surface = _build_choppy_surface()
    report = audit_surface_stability(surface)
    summary = report["summary"]

    assert "no_arbitrage" in summary
    assert "butterfly_violations" in summary
    assert "calendar_violations" in summary
    assert isinstance(summary["butterfly_violations"], int)
    assert isinstance(summary["calendar_violations"], int)
