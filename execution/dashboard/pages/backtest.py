"""Backtest results page — view individual backtest run metrics and charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from execution.dashboard.chart_builders import (
    pnl_line_chart,
    pnl_sampled_chart,
    position_chart,
)
from execution.dashboard.data_helpers import (
    DEFAULT_RESULTS_DIR,
    available_json_results,
    load_backtest_json,
)
from execution.dashboard.templates import (
    base_layout,
    data_table,
    file_selector,
    metric_card,
)


def _fmt(value: Any) -> str:
    """Format a numeric value for display."""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def _build_metrics_html(summary: Dict[str, Any]) -> str:
    """Build metrics grid from a strategy summary dict."""
    cards = [
        metric_card("Total PnL", _fmt(summary.get("total_pnl", summary.get("final_pnl", 0)))),
        metric_card("Sharpe Ratio", _fmt(summary.get("sharpe_ratio", summary.get("sharpe", 0)))),
        metric_card("Max Drawdown", _fmt(summary.get("max_drawdown", 0))),
        metric_card("Trade Count", str(summary.get("trade_count", summary.get("total_trades", 0)))),
        metric_card("Buy / Sell",
                     f'{summary.get("buy_count", summary.get("buy_trades", 0))} / '
                     f'{summary.get("sell_count", summary.get("sell_trades", 0))}'),
        metric_card("Spread Capture", _fmt(summary.get("spread_capture", 0))),
        metric_card("Adverse Selection", _fmt(summary.get("adverse_selection_cost", 0))),
        metric_card("Inventory Cost", _fmt(summary.get("inventory_cost", 0))),
    ]
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def _build_backtest_page(
    data: Dict[str, Any],
    strategy_name: str,
    file_path: str,
    json_files: List[Dict[str, str]],
) -> str:
    """Build full backtest page HTML for a single strategy."""
    summary = data.get("summary", data.get("metrics", {}))
    metrics_html = _build_metrics_html(summary)

    pnl_history = data.get("pnl_history", [])
    pnl_sampled = data.get("pnl_history_sampled", [])
    if pnl_history:
        pnl_html = pnl_line_chart(pnl_history, f"Cumulative PnL — {strategy_name}")
    elif pnl_sampled:
        pnl_html = pnl_sampled_chart(pnl_sampled, f"Cumulative PnL — {strategy_name}")
    else:
        pnl_html = "<p>No PnL data.</p>"

    position_data = data.get("position_history", [])
    pos_html = position_chart(position_data, f"Position — {strategy_name}") if position_data else ""

    selector = file_selector(json_files, file_path, "/backtest")

    # Strategy selector tabs
    strategy_names = list(data.keys()) if "summary" not in data and "metrics" not in data else []
    strategy_tabs = ""
    if strategy_names:
        tabs = " | ".join(
            f'<a href="/backtest?file={file_path}&strategy={name}">'
            f'{"<b>" + name + "</b>" if name == strategy_name else name}</a>'
            for name in strategy_names
        )
        strategy_tabs = f'<div class="card"><h2>Strategies</h2>{tabs}</div>'

    body = f"""
{selector}
{strategy_tabs}
<div class="card"><h2>{strategy_name}</h2></div>
{metrics_html}
<div class="card">{pnl_html}</div>
{"<div class='card'>" + pos_html + "</div>" if pos_html else ""}
<div class="card"><h2>Summary</h2>{_summary_table(summary)}</div>
"""
    return base_layout("Backtest", "Backtest", body)


def _summary_table(summary: Dict[str, Any]) -> str:
    """Render summary dict as a two-column table."""
    headers = ["Metric", "Value"]
    rows = [[k, _fmt(v)] for k, v in summary.items()]
    return data_table(headers, rows)


def register_backtest_routes(app: FastAPI, directory: Path) -> None:
    """Register backtest results page and API."""

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(
        file: Optional[str] = Query(default=None),
        strategy: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        json_files = available_json_results(directory)
        if not json_files and not file:
            body = '<div class="card"><h2>No backtest results found</h2><p>Run a backtest first to generate result files.</p></div>'
            return HTMLResponse(content=base_layout("Backtest", "Backtest", body))

        file_path = file or json_files[0]["path"]
        full_path = directory / file_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        raw = load_backtest_json(full_path)

        # Multi-strategy file: pick first or requested
        strategy_names = list(raw.keys())
        if not strategy_names:
            raise HTTPException(status_code=422, detail="Empty backtest result file")

        selected_strategy = strategy if strategy in strategy_names else strategy_names[0]
        strategy_data = raw[selected_strategy]

        # Inject all strategy names for tab navigation
        strategy_data_with_names = raw

        html = _build_backtest_page(strategy_data, selected_strategy, file_path, json_files)

        # Add strategy tabs
        if len(strategy_names) > 1:
            tabs = " | ".join(
                f'<a href="/backtest?file={file_path}&strategy={name}">'
                f'{"<b>" + name + "</b>" if name == selected_strategy else name}</a>'
                for name in strategy_names
            )
            tabs_html = f'<div class="card"><h2>Strategies</h2>{tabs}</div>'
            html = html.replace("<!-- strategy-tabs -->", tabs_html)

        return HTMLResponse(content=html)

    @app.get("/api/backtest/results", response_class=JSONResponse)
    async def backtest_results() -> dict:
        return {"files": available_json_results(directory)}
