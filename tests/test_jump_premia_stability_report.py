"""
Tests for jump premia stability reporting script.
"""

import numpy as np

from validation_scripts.jump_premia_stability_report import (
    build_report,
    build_synthetic_prices,
    render_markdown,
)


def test_build_synthetic_prices_is_finite_and_deterministic():
    series_a = build_synthetic_prices(seed=42, n_points=120)
    series_b = build_synthetic_prices(seed=42, n_points=120)

    assert len(series_a) == 120
    assert np.all(np.isfinite(series_a.to_numpy()))
    assert np.allclose(series_a.to_numpy(), series_b.to_numpy())


def test_build_report_and_markdown_include_core_metrics():
    prices = build_synthetic_prices(seed=7, n_points=240)
    report = build_report(prices=prices, window=48, jump_zscore=2.2)
    summary = report["summary"]

    assert summary["n_points"] == 240
    assert summary["window"] == 48
    assert "latest_net_jump_premium" in summary
    assert "net_jump_premium_std" in summary
    assert summary["net_jump_premium_std"] >= 0.0

    markdown = render_markdown(report)
    assert "# Jump Premia Stability Report" in markdown
    assert "| Latest net jump premium |" in markdown
    assert "| Net jump premium std |" in markdown
