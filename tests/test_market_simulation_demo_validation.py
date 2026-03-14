"""Tests for the notebook-01 validation helper."""

from pathlib import Path

from scripts.backtest.validate_market_simulation_demo import (
    build_notebook_demo_market_data,
    run_validation,
)


def test_build_notebook_demo_market_data_returns_required_frames() -> None:
    market_data = build_notebook_demo_market_data(days=2, seed=42)

    assert set(market_data) == {"spot", "order_book", "options"}
    assert not market_data["spot"].empty
    assert not market_data["order_book"].empty
    assert not market_data["options"].empty
    assert market_data["options"]["timestamp"].nunique() == 1


def test_run_validation_writes_metrics_and_plot(tmp_path: Path) -> None:
    report = run_validation(output_dir=tmp_path, days=2, seed=42)

    assert report["naive"]["trade_count"] > 0
    assert report["avellaneda_stoikov"]["trade_count"] > 0
    assert (tmp_path / "strategy_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
