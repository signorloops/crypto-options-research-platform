"""FastAPI application factory — assembles all dashboard page routers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from execution.dashboard.data_helpers import DEFAULT_RESULTS_DIR
from execution.dashboard.pages.backtest import register_backtest_routes
from execution.dashboard.pages.deviation import register_deviation_routes
from execution.dashboard.pages.overview import register_overview_routes
from execution.dashboard.pages.risk import register_risk_routes
from execution.dashboard.pages.strategy import register_strategy_routes


def create_dashboard_app(results_dir: Optional[Path] = None) -> FastAPI:
    """Create the multi-page dashboard application."""
    directory = results_dir or DEFAULT_RESULTS_DIR
    app = FastAPI(title="CORP Research Dashboard", version="2.0.0")

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict:
        return {"status": "healthy", "results_dir": str(directory)}

    register_overview_routes(app, directory)
    register_deviation_routes(app, directory)
    register_backtest_routes(app, directory)
    register_strategy_routes(app, directory)
    register_risk_routes(app, directory)

    return app
