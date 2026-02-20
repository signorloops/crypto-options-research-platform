"""Lightweight research dashboard for backtest/result inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

DEFAULT_RESULTS_DIR = Path(os.getenv("CORP_RESULTS_DIR", "results"))
TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
VALUE_CANDIDATES = ("equity", "portfolio_value", "cum_pnl", "pnl", "value", "close")


def available_result_files(results_dir: Path) -> List[Path]:
    """Return sorted result files supported by the dashboard."""
    files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.parquet"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _find_value_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in VALUE_CANDIDATES:
        if candidate in df.columns:
            return candidate

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return numeric_cols[0] if numeric_cols else None


def _load_results_dataframe(
    results_dir: Path, file_name: Optional[str]
) -> Tuple[pd.DataFrame, Path]:
    candidates = available_result_files(results_dir)
    if not candidates:
        raise HTTPException(status_code=404, detail="No results files found")

    if file_name:
        selected = results_dir / file_name
        if selected not in candidates:
            raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
    else:
        selected = candidates[0]

    if selected.suffix == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)

    if df.empty:
        raise HTTPException(status_code=422, detail=f"Result file is empty: {selected.name}")

    time_col = _find_time_column(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)

    return df, selected


def _build_dashboard_html(df: pd.DataFrame, selected: Path, files: List[Path]) -> str:
    time_col = _find_time_column(df)
    value_col = _find_value_column(df)
    if value_col is None:
        raise HTTPException(status_code=422, detail="No numeric column available for plotting")

    if time_col:
        x_axis = time_col
    else:
        df = df.reset_index(drop=False).rename(columns={"index": "index"})
        x_axis = "index"

    primary_fig = px.line(
        df,
        x=x_axis,
        y=value_col,
        title=f"{selected.name}: {value_col}",
        template="plotly_white",
    )

    returns = df[value_col].pct_change().dropna()
    returns_fig = px.histogram(
        returns,
        nbins=40,
        title=f"Return Distribution ({value_col})",
        template="plotly_white",
    )

    summary = {
        "rows": len(df),
        "value_column": value_col,
        "mean": float(df[value_col].mean()),
        "std": float(df[value_col].std(ddof=0) if len(df) > 1 else 0.0),
        "min": float(df[value_col].min()),
        "max": float(df[value_col].max()),
    }

    file_options = "\n".join(
        f'<option value="{path.name}" {"selected" if path == selected else ""}>{path.name}</option>'
        for path in files
    )

    summary_rows = "".join(
        (
            f"<tr><th>{key}</th><td>{value:.6f}</td></tr>"
            if isinstance(value, float)
            else f"<tr><th>{key}</th><td>{value}</td></tr>"
        )
        for key, value in summary.items()
    )

    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>CORP Research Dashboard</title>
    <style>
      body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #1f2937; background: #f4f6f8; }}
      .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
      .card {{ background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); padding: 20px; margin-bottom: 20px; }}
      h1 {{ margin: 0 0 16px; font-size: 28px; }}
      .toolbar {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
      select, button {{ padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; background: white; }}
      button {{ background: #111827; color: white; cursor: pointer; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
      @media (max-width: 768px) {{ .container {{ padding: 12px; }} h1 {{ font-size: 22px; }} }}
    </style>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"card\">
        <h1>CORP Research Dashboard</h1>
        <form method=\"get\" class=\"toolbar\">
          <label for=\"file\">Result File</label>
          <select id=\"file\" name=\"file\">{file_options}</select>
          <button type=\"submit\">Load</button>
        </form>
      </div>

      <div class=\"card\">{primary_fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
      <div class=\"card\">{returns_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
      <div class=\"card\">
        <h2>Summary</h2>
        <table>{summary_rows}</table>
      </div>
    </div>
  </body>
</html>
"""


def create_dashboard_app(results_dir: Optional[Path] = None) -> FastAPI:
    """Create dashboard API application."""
    directory = results_dir or DEFAULT_RESULTS_DIR
    app = FastAPI(title="CORP Research Dashboard", version="1.0.0")

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict:
        return {"status": "healthy", "results_dir": str(directory)}

    @app.get("/api/files", response_class=JSONResponse)
    async def files() -> dict:
        return {"files": [path.name for path in available_result_files(directory)]}

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(file: Optional[str] = Query(default=None)) -> HTMLResponse:
        df, selected = _load_results_dataframe(directory, file)
        html = _build_dashboard_html(df, selected, available_result_files(directory))
        return HTMLResponse(content=html)

    return app


app = create_dashboard_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "execution.research_dashboard:app",
        host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.getenv("DASHBOARD_PORT", "8501")),
        reload=False,
    )
