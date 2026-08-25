"""Risk dashboard page — circuit breaker, Greeks, VaR, regime status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from execution.dashboard.chart_builders import (
    circuit_breaker_badge,
    greeks_bar_chart,
    regime_badge,
)
from execution.dashboard.data_helpers import _to_float
from execution.dashboard.templates import base_layout, data_table, metric_card


def _coerce_str(value: Any, default: str) -> str:
    """Default on null (dict.get's default only fires on a missing key)."""
    if value is None:
        return default
    return value if isinstance(value, str) and value else default

# Default demo data when no snapshot file exists
DEMO_RISK_DATA: Dict[str, Any] = {
    "circuit_breaker": {
        "state": "NORMAL",
        "multiplier": 1.0,
        "violations": [
            {"time": "2024-01-01T12:00:00", "type": "daily_loss", "detail": "Loss reached 4.8%"},
            {"time": "2024-01-01T14:30:00", "type": "position_limit", "detail": "Position 95% of limit"},
        ],
    },
    "greeks": {
        "delta": 0.15,
        "gamma": -0.02,
        "theta": -45.0,
        "vega": 120.0,
    },
    "var": {
        "var_95": -2500.0,
        "var_99": -4200.0,
        "cvar_95": -3100.0,
        "cvar_99": -5800.0,
    },
    "regime": {
        "state": "LOW",
        "volatility_percentile": 25.0,
    },
}


def _load_risk_snapshot(results_dir: Path) -> Dict[str, Any]:
    """Load risk snapshot from JSON, or return demo data."""
    snapshot_path = results_dir / "risk_snapshot.json"
    if snapshot_path.exists():
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEMO_RISK_DATA


def _build_risk_page(data: Dict[str, Any]) -> str:
    """Build full risk dashboard HTML."""
    cb = data.get("circuit_breaker", {})
    greeks = data.get("greeks", {})
    var_data = data.get("var", {})
    regime = data.get("regime", {})

    # Circuit breaker section
    # dict.get's default only fires when the key is absent; explicit nulls
    # must be coerced too, otherwise str.upper()/float formatting raise.
    cb_badge = circuit_breaker_badge(_coerce_str(cb.get("state"), "NORMAL"))
    cb_multiplier = f'{_to_float(cb.get("multiplier"), 1.0):.2f}'

    violations = cb.get("violations", [])
    if violations:
        violation_headers = ["Time", "Type", "Detail"]
        violation_rows = [
            [v.get("time", ""), v.get("type", ""), v.get("detail", "")]
            for v in violations[:20]
        ]
        violations_html = data_table(violation_headers, violation_rows)
    else:
        violations_html = "<p>No violations recorded.</p>"

    # Greeks section
    # Drop null greeks: the bar-chart color logic compares values against 0,
    # and an explicit null (not just a missing key) would raise TypeError.
    greeks_chart = greeks_bar_chart({k: v for k, v in greeks.items() if v is not None})

    # VaR section
    var_cards = "".join([
        metric_card("VaR 95%", f'${_to_float(var_data.get("var_95")):,.0f}'),
        metric_card("VaR 99%", f'${_to_float(var_data.get("var_99")):,.0f}'),
        metric_card("CVaR 95%", f'${_to_float(var_data.get("cvar_95")):,.0f}'),
        metric_card("CVaR 99%", f'${_to_float(var_data.get("cvar_99")):,.0f}'),
    ])

    # Regime section
    regime_html = regime_badge(_coerce_str(regime.get("state"), "LOW"))
    vol_pct = _to_float(regime.get("volatility_percentile"))

    body = f"""
<div class="card">
  <h1>Risk Dashboard</h1>
</div>
<div class="card">
  <h2>Circuit Breaker</h2>
  <p>State: {cb_badge} &nbsp; Multiplier: <b>{cb_multiplier}x</b></p>
</div>
<div class="card">
  <h2>Violation History</h2>
  {violations_html}
</div>
<div class="card">
  <h2>Greeks Exposure</h2>
  {greeks_chart}
</div>
<div class="card">
  <h2>Value at Risk</h2>
  <div class="metrics-grid">{var_cards}</div>
</div>
<div class="card">
  <h2>Market Regime</h2>
  <p>State: {regime_html} &nbsp; Volatility Percentile: <b>{vol_pct:.1f}%</b></p>
</div>
"""
    return base_layout("Risk", "Risk", body)


def register_risk_routes(app: FastAPI, directory: Path) -> None:
    """Register risk dashboard page and API."""

    @app.get("/risk", response_class=HTMLResponse)
    async def risk_page() -> HTMLResponse:
        data = _load_risk_snapshot(directory)
        html = _build_risk_page(data)
        return HTMLResponse(content=html)

    @app.get("/api/risk/status", response_class=JSONResponse)
    async def risk_status() -> dict:
        return _load_risk_snapshot(directory)
