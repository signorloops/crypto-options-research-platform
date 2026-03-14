"""Overview page — file browser, time series chart, return distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from execution.dashboard.chart_builders import (
    deviation_heatmap,
    returns_histogram,
    timeseries_line,
)
from execution.dashboard.data_helpers import (
    _build_file_options,
    _build_summary_rows,
    _find_time_column,
    _find_value_column,
    _load_results_dataframe,
    available_result_files,
)
from execution.dashboard.templates import base_layout


def _resolve_dashboard_axis(
    df: pd.DataFrame, time_col: Optional[str]
) -> tuple:
    if time_col:
        return df, time_col
    indexed = df.reset_index(drop=False).rename(columns={"index": "index"})
    return indexed, "index"


def _build_dashboard_summary(df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    return {
        "rows": len(df),
        "value_column": value_col,
        "mean": float(df[value_col].mean()),
        "std": float(df[value_col].std(ddof=0) if len(df) > 1 else 0.0),
        "min": float(df[value_col].min()),
        "max": float(df[value_col].max()),
    }


def _render_deviation_alerts_table(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return "<p>No alerts over 300 bps.</p>"
    header = "".join(f"<th>{key}</th>" for key in alerts[0].keys())
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in alerts[:20]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def _build_deviation_section(df: pd.DataFrame) -> str:
    from execution.dashboard.pages.deviation import build_cross_market_deviation_report

    try:
        deviation_report = build_cross_market_deviation_report(df, threshold_bps=300.0)
    except HTTPException:
        return ""
    heatmap_df = pd.DataFrame(deviation_report["heatmap_records"])
    if heatmap_df.empty:
        return ""
    alerts_table = _render_deviation_alerts_table(deviation_report["alerts"])
    heatmap_html = deviation_heatmap(heatmap_df)
    return (
        f'<div class="card">{heatmap_html}</div>'
        f'<div class="card"><h2>Deviation Alerts</h2>{alerts_table}</div>'
    )


def _build_dashboard_html(
    df: pd.DataFrame, selected: Path, files: List[Path]
) -> str:
    time_col = _find_time_column(df)
    value_col = _find_value_column(df)
    if value_col is None:
        raise HTTPException(status_code=422, detail="No numeric column available for plotting")

    plot_df, x_axis = _resolve_dashboard_axis(df, time_col)

    primary_fig_html = timeseries_line(
        plot_df, x=x_axis, y=value_col, title=f"{selected.name}: {value_col}"
    )
    returns = plot_df[value_col].pct_change().dropna()
    returns_fig_html = returns_histogram(returns, title=f"Return Distribution ({value_col})")

    summary = _build_dashboard_summary(plot_df, value_col)
    file_options = _build_file_options(files, selected)
    summary_rows = _build_summary_rows(summary)
    deviation_section = _build_deviation_section(plot_df)

    body = f"""
<div class="card">
  <h1>CORP Research Dashboard</h1>
  <form method="get" class="toolbar">
    <label for="file">Result File</label>
    <select id="file" name="file">{file_options}</select>
    <button type="submit">Load</button>
  </form>
</div>
<div class="card">{primary_fig_html}</div>
<div class="card">{returns_fig_html}</div>
{deviation_section}
<div class="card"><h2>Summary</h2><table>{summary_rows}</table></div>
"""
    return base_layout("Overview", "Overview", body)


def register_overview_routes(app: FastAPI, directory: Path) -> None:
    """Register overview page and file-list API."""

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(file: Optional[str] = Query(default=None)) -> HTMLResponse:
        df, selected = _load_results_dataframe(directory, file)
        html = _build_dashboard_html(df, selected, available_result_files(directory))
        return HTMLResponse(content=html)

    @app.get("/api/files", response_class=JSONResponse)
    async def files() -> dict:
        return {"files": [path.name for path in available_result_files(directory)]}
