"""Tests for the backtest results dashboard page."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from execution.research_dashboard import create_dashboard_app


SAMPLE_BACKTEST = {
    "Strategy A": {
        "summary": {
            "position": -1.0,
            "cash": 100500.0,
            "total_pnl": 500.0,
            "sharpe_ratio": 2.5,
            "max_drawdown": -0.05,
            "trade_count": 100,
            "buy_count": 50,
            "sell_count": 50,
            "spread_capture": 600.0,
            "adverse_selection_cost": 50.0,
            "inventory_cost": 10.0,
            "hedging_cost": 40.0,
        },
        "pnl_history": [
            ["2024-01-01T00:05:00", 5.0],
            ["2024-01-01T00:10:00", 12.0],
            ["2024-01-01T00:15:00", 18.0],
        ],
        "position_history": [
            ["2024-01-01T00:05:00", -0.1],
            ["2024-01-01T00:10:00", 0.2],
            ["2024-01-01T00:15:00", -0.3],
        ],
    },
    "Strategy B": {
        "summary": {
            "position": 0.5,
            "cash": 100200.0,
            "total_pnl": 200.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "trade_count": 80,
            "buy_count": 45,
            "sell_count": 35,
            "spread_capture": 300.0,
            "adverse_selection_cost": 60.0,
            "inventory_cost": 15.0,
            "hedging_cost": 25.0,
        },
        "pnl_history": [
            ["2024-01-01T00:05:00", 2.0],
            ["2024-01-01T00:10:00", 8.0],
        ],
        "position_history": [
            ["2024-01-01T00:05:00", 0.1],
            ["2024-01-01T00:10:00", 0.3],
        ],
    },
}


def _write_backtest_json(tmp_path, data=None):
    """Write sample backtest JSON and return results dir."""
    subdir = tmp_path / "backtest_with_output"
    subdir.mkdir(parents=True)
    path = subdir / "backtest_results_test.json"
    path.write_text(json.dumps(data or SAMPLE_BACKTEST), encoding="utf-8")
    return tmp_path


def test_backtest_api_lists_json_files(tmp_path):
    results_dir = _write_backtest_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/api/backtest/results")
    assert response.status_code == 200
    files = response.json()["files"]
    assert len(files) == 1
    assert files[0]["name"] == "backtest_results_test.json"


def test_backtest_page_renders_html(tmp_path):
    results_dir = _write_backtest_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/backtest")
    assert response.status_code == 200
    assert "Strategy A" in response.text
    assert "Total PnL" in response.text
    assert "Cumulative PnL" in response.text


def test_backtest_page_selects_strategy(tmp_path):
    results_dir = _write_backtest_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get(
            "/backtest",
            params={
                "file": "backtest_with_output/backtest_results_test.json",
                "strategy": "Strategy B",
            },
        )
    assert response.status_code == 200
    assert "Strategy B" in response.text


def test_backtest_page_renders_strategy_tabs(tmp_path):
    results_dir = _write_backtest_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/backtest")
    assert response.status_code == 200
    assert "Strategies" in response.text
    assert (
        "/backtest?file=backtest_with_output/backtest_results_test.json&strategy=Strategy B"
        in response.text
    )


def test_backtest_page_no_files_shows_message(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/backtest")
    assert response.status_code == 200
    assert "No backtest results found" in response.text


def test_backtest_page_missing_file_returns_404(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/backtest", params={"file": "nonexistent.json"})
    assert response.status_code == 404


def test_backtest_page_rejects_file_outside_results(tmp_path):
    results_dir = tmp_path / "results"
    _write_backtest_json(results_dir)
    (tmp_path / "outside.json").write_text(json.dumps(SAMPLE_BACKTEST), encoding="utf-8")

    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/backtest", params={"file": "../outside.json"})
    assert response.status_code == 404
