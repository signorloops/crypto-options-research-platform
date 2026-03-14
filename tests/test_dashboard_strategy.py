"""Tests for the strategy comparison dashboard page."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from execution.research_dashboard import create_dashboard_app


SAMPLE_STRATEGIES = {
    "Naive MM": {
        "summary": {
            "total_pnl": 1665.0,
            "sharpe_ratio": 710.0,
            "max_drawdown": -0.10,
            "trade_count": 322,
            "buy_count": 148,
            "sell_count": 174,
            "spread_capture": 1963.0,
        },
        "pnl_history": [
            ["2024-01-01T00:05:00", 5.0],
            ["2024-01-01T00:10:00", 50.0],
            ["2024-01-01T00:15:00", 100.0],
        ],
    },
    "A-S Model": {
        "summary": {
            "total_pnl": 21.0,
            "sharpe_ratio": 1788.0,
            "max_drawdown": -0.107,
            "trade_count": 322,
            "buy_count": 148,
            "sell_count": 174,
            "spread_capture": 318.0,
        },
        "pnl_history": [
            ["2024-01-01T00:05:00", 0.06],
            ["2024-01-01T00:10:00", 5.0],
            ["2024-01-01T00:15:00", 10.0],
        ],
    },
}


def _write_strategy_json(tmp_path):
    subdir = tmp_path / "backtest_with_output"
    subdir.mkdir()
    path = subdir / "backtest_results_test.json"
    path.write_text(json.dumps(SAMPLE_STRATEGIES), encoding="utf-8")
    return tmp_path


def test_strategy_page_renders_comparison(tmp_path):
    results_dir = _write_strategy_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/strategy")
    assert response.status_code == 200
    assert "Naive MM" in response.text
    assert "A-S Model" in response.text
    assert "Strategy Comparison" in response.text


def test_strategy_api_returns_metrics(tmp_path):
    results_dir = _write_strategy_json(tmp_path)
    app = create_dashboard_app(results_dir=results_dir)
    with TestClient(app) as client:
        response = client.get("/api/strategy/compare")
    assert response.status_code == 200
    payload = response.json()
    strategies = payload["strategies"]
    assert "Naive MM" in strategies
    assert "A-S Model" in strategies
    assert strategies["Naive MM"]["total_pnl"] == 1665.0
    assert strategies["A-S Model"]["sharpe_ratio"] == 1788.0


def test_strategy_page_no_files_shows_message(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/strategy")
    assert response.status_code == 200
    assert "No backtest results found" in response.text


def test_strategy_api_no_files_returns_empty(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/strategy/compare")
    assert response.status_code == 200
    assert response.json()["strategies"] == {}
