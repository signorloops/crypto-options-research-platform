"""Strategy comparison page — side-by-side metrics and charts for multiple strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from execution.dashboard.chart_builders import comparison_bars, multi_pnl_overlay
from execution.dashboard.data_helpers import available_json_results, load_backtest_json, normalize_strategy_results_payload, resolve_json_result_path
from execution.dashboard.templates import base_layout, data_table, file_selector


def _extract_metric(summary: Dict[str, Any], *keys: str) -> float:
    """Extract a metric trying multiple key names."""
    for k in keys:
        if k in summary:
            val = summary[k]
            return float(val) if val is not None else 0.0
    return 0.0


def _build_comparison_table(strategies: Dict[str, Dict[str, Any]]) -> str:
    """Build HTML table comparing metrics across strategies."""
    headers = ["Strategy", "Total PnL", "Sharpe", "Max DD", "Trades", "Spread Capture"]
    rows = []
    for name, data in strategies.items():
        s = data.get("summary", data.get("metrics", {}))
        rows.append([
            name,
            f'{_extract_metric(s, "total_pnl", "final_pnl"):,.2f}',
            f'{_extract_metric(s, "sharpe_ratio", "sharpe"):.2f}',
            f'{_extract_metric(s, "max_drawdown"):.4f}',
            str(int(_extract_metric(s, "trade_count", "total_trades"))),
            f'{_extract_metric(s, "spread_capture"):,.2f}',
        ])
    return data_table(headers, rows)


def _build_strategy_page(
    strategies: Dict[str, Dict[str, Any]],
    file_path: str,
    json_files: List[Dict[str, str]],
) -> str:
    """Build full strategy comparison page."""
    selector = file_selector(json_files, file_path, "/strategy")
    table_html = _build_comparison_table(strategies)

    # PnL overlay
    pnl_overlay = multi_pnl_overlay(strategies)

    # Bar charts
    names = list(strategies.keys())
    pnl_values = [
        _extract_metric(
            strategies[n].get("summary", strategies[n].get("metrics", {})),
            "total_pnl", "final_pnl",
        )
        for n in names
    ]
    sharpe_values = [
        _extract_metric(
            strategies[n].get("summary", strategies[n].get("metrics", {})),
            "sharpe_ratio", "sharpe",
        )
        for n in names
    ]
    trade_values = [
        _extract_metric(
            strategies[n].get("summary", strategies[n].get("metrics", {})),
            "trade_count", "total_trades",
        )
        for n in names
    ]

    pnl_bars = comparison_bars(names, pnl_values, "Total PnL", y_label="PnL")
    sharpe_bars = comparison_bars(names, sharpe_values, "Sharpe Ratio", y_label="Sharpe")
    trade_bars = comparison_bars(names, trade_values, "Trade Count", y_label="Trades")

    body = f"""
{selector}
<div class="card"><h2>Strategy Comparison</h2>{table_html}</div>
<div class="card">{pnl_overlay}</div>
<div class="chart-grid">
  <div class="card">{pnl_bars}</div>
  <div class="card">{sharpe_bars}</div>
</div>
<div class="card">{trade_bars}</div>
"""
    return base_layout("Strategy", "Strategy", body)


def register_strategy_routes(app: FastAPI, directory: Path) -> None:
    """Register strategy comparison page and API."""

    @app.get("/strategy", response_class=HTMLResponse)
    async def strategy_page(
        file: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        json_files = available_json_results(directory)
        if not json_files:
            body = '<div class="card"><h2>No backtest results found</h2><p>Run a backtest first to generate result files.</p></div>'
            return HTMLResponse(content=base_layout("Strategy", "Strategy", body))

        full_path, file_path, json_files = resolve_json_result_path(directory, file)
        strategies = normalize_strategy_results_payload(load_backtest_json(full_path), file_name=full_path.name)
        if not strategies:
            raise HTTPException(status_code=422, detail="Empty backtest result file")

        html = _build_strategy_page(strategies, file_path, json_files)
        return HTMLResponse(content=html)

    @app.get("/api/strategy/compare", response_class=JSONResponse)
    async def strategy_compare(
        file: Optional[str] = Query(default=None),
    ) -> dict:
        json_files = available_json_results(directory)
        if not json_files:
            return {"strategies": {}, "files": []}

        full_path, file_path, json_files = resolve_json_result_path(directory, file)
        strategies = normalize_strategy_results_payload(load_backtest_json(full_path), file_name=full_path.name)
        result = {}
        for name, data in strategies.items():
            s = data.get("summary", data.get("metrics", {}))
            result[name] = {
                "total_pnl": _extract_metric(s, "total_pnl", "final_pnl"),
                "sharpe_ratio": _extract_metric(s, "sharpe_ratio", "sharpe"),
                "max_drawdown": _extract_metric(s, "max_drawdown"),
                "trade_count": int(_extract_metric(s, "trade_count", "total_trades")),
                "spread_capture": _extract_metric(s, "spread_capture"),
            }
        return {"strategies": result, "file": file_path}
