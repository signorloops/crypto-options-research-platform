"""Tests for the risk dashboard page."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from execution.research_dashboard import create_dashboard_app


def test_risk_page_renders_with_demo_data(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/risk")
    assert response.status_code == 200
    assert "Risk Dashboard" in response.text
    assert "Circuit Breaker" in response.text
    assert "NORMAL" in response.text
    assert "Greeks Exposure" in response.text
    assert "Value at Risk" in response.text


def test_risk_api_returns_demo_data(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/risk/status")
    assert response.status_code == 200
    data = response.json()
    assert data["circuit_breaker"]["state"] == "NORMAL"
    assert "delta" in data["greeks"]
    assert "var_95" in data["var"]
    assert data["regime"]["state"] == "LOW"


def test_risk_page_loads_snapshot_file(tmp_path):
    snapshot = {
        "circuit_breaker": {"state": "WARNING", "multiplier": 0.5, "violations": []},
        "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -20.0, "vega": 80.0},
        "var": {"var_95": -1000.0, "var_99": -2000.0, "cvar_95": -1500.0, "cvar_99": -3000.0},
        "regime": {"state": "HIGH", "volatility_percentile": 90.0},
    }
    (tmp_path / "risk_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/risk")
    assert response.status_code == 200
    assert "WARNING" in response.text

    with TestClient(app) as client:
        api_response = client.get("/api/risk/status")
    assert api_response.json()["circuit_breaker"]["state"] == "WARNING"
    assert api_response.json()["regime"]["state"] == "HIGH"


def test_risk_page_renders_violation_table(tmp_path):
    snapshot = {
        "circuit_breaker": {
            "state": "RESTRICTED",
            "multiplier": 0.25,
            "violations": [
                {"time": "2024-01-01T12:00:00", "type": "daily_loss", "detail": "Loss 8%"},
                {"time": "2024-01-01T14:00:00", "type": "position", "detail": "Over limit"},
            ],
        },
        "greeks": {"delta": 0.0},
        "var": {"var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0, "cvar_99": 0.0},
        "regime": {"state": "MEDIUM", "volatility_percentile": 55.0},
    }
    (tmp_path / "risk_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/risk")
    assert response.status_code == 200
    assert "RESTRICTED" in response.text
    assert "daily_loss" in response.text
    assert "Loss 8%" in response.text
