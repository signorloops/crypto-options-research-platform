"""Lightweight research dashboard for backtest/result inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from data.quote_integration import (
    build_cex_defi_deviation_dataset,
    build_cex_defi_deviation_dataset_live,
)

DEFAULT_RESULTS_DIR = Path(os.getenv("CORP_RESULTS_DIR", "results"))
TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
VALUE_CANDIDATES = ("equity", "portfolio_value", "cum_pnl", "pnl", "value", "close")
MARKET_PRICE_CANDIDATES = ("market_price", "quote_price", "option_price", "price")
MODEL_PRICE_CANDIDATES = ("model_price", "theoretical_price", "benchmark_price", "fair_value")
DELTA_CANDIDATES = ("delta", "abs_delta", "delta_abs")
EXPIRY_CANDIDATES = ("expiry_years", "maturity", "time_to_expiry", "tau")
VENUE_CANDIDATES = ("venue", "exchange", "source", "market")


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


def _find_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _bucket_expiry(expiry_years: pd.Series) -> pd.Series:
    days = expiry_years.astype(float) * 365.0
    bins = [-float("inf"), 7.0, 30.0, 90.0, 180.0, float("inf")]
    labels = ["<=7D", "8-30D", "31-90D", "91-180D", ">180D"]
    return pd.cut(days, bins=bins, labels=labels)


def _bucket_abs_delta(abs_delta: pd.Series) -> pd.Series:
    bins = [-float("inf"), 0.10, 0.25, 0.40, 0.60, 1.0]
    labels = ["0-10d", "10-25d", "25-40d", "40-60d", "60-100d"]
    clipped = abs_delta.astype(float).clip(lower=0.0, upper=1.0)
    return pd.cut(clipped, bins=bins, labels=labels)


def _resolve_deviation_columns(
    df: pd.DataFrame,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """Resolve market/model and optional bucket columns."""
    market_col = _find_first_existing(df, MARKET_PRICE_CANDIDATES)
    model_col = _find_first_existing(df, MODEL_PRICE_CANDIDATES)
    if market_col is None or model_col is None:
        raise HTTPException(
            status_code=422,
            detail="Deviation analysis requires market_price and model_price style columns",
        )
    expiry_col = _find_first_existing(df, EXPIRY_CANDIDATES)
    delta_col = _find_first_existing(df, DELTA_CANDIDATES)
    venue_col = _find_first_existing(df, VENUE_CANDIDATES)
    return market_col, model_col, expiry_col, delta_col, venue_col


def _prepare_deviation_frame(
    df: pd.DataFrame,
    *,
    market_col: str,
    model_col: str,
    expiry_col: Optional[str],
    delta_col: Optional[str],
) -> pd.DataFrame:
    """Build normalized frame with deviation metrics and buckets."""
    work = df.copy()
    work["market_px"] = pd.to_numeric(work[market_col], errors="coerce")
    work["model_px"] = pd.to_numeric(work[model_col], errors="coerce")
    work = work.dropna(subset=["market_px", "model_px"])
    if work.empty:
        raise HTTPException(status_code=422, detail="No valid rows for deviation analysis")
    denom = work["model_px"].abs().clip(lower=1e-12)
    work["deviation_bps"] = (work["market_px"] - work["model_px"]) / denom * 10000.0
    work["abs_deviation_bps"] = work["deviation_bps"].abs()

    if expiry_col:
        work["expiry_bucket"] = _bucket_expiry(
            pd.to_numeric(work[expiry_col], errors="coerce").fillna(0.0)
        )
    else:
        work["expiry_bucket"] = "ALL"
    if delta_col:
        abs_delta = pd.to_numeric(work[delta_col], errors="coerce").abs().fillna(0.5)
        work["delta_bucket"] = _bucket_abs_delta(abs_delta)
    else:
        work["delta_bucket"] = "ALL"
    return work


def _build_deviation_heatmap(work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create heatmap table and pivot matrix for abs deviation bps."""
    heatmap = (
        work.groupby(["expiry_bucket", "delta_bucket"], observed=False)["abs_deviation_bps"]
        .mean()
        .reset_index()
    )
    heatmap["abs_deviation_bps"] = heatmap["abs_deviation_bps"].fillna(0.0)
    pivot = heatmap.pivot(index="expiry_bucket", columns="delta_bucket", values="abs_deviation_bps")
    pivot = pivot.fillna(0.0)
    return heatmap, pivot


def _build_deviation_alerts(
    work: pd.DataFrame,
    *,
    threshold_bps: float,
    venue_col: Optional[str],
    expiry_col: Optional[str],
    delta_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Build sorted alert table and selected output columns."""
    alerts = work[work["abs_deviation_bps"] >= float(threshold_bps)].copy()
    alert_cols = [c for c in [venue_col, expiry_col, delta_col] if c is not None]
    alert_cols += ["market_px", "model_px", "deviation_bps", "abs_deviation_bps"]
    alerts = alerts.sort_values("abs_deviation_bps", ascending=False)
    return alerts, alert_cols


def build_cross_market_deviation_report(
    df: pd.DataFrame, threshold_bps: float = 300.0
) -> Dict[str, Any]:
    """Build cross-market deviation heatmap and threshold alerts."""
    market_col, model_col, expiry_col, delta_col, venue_col = _resolve_deviation_columns(df)
    work = _prepare_deviation_frame(df, market_col=market_col, model_col=model_col, expiry_col=expiry_col, delta_col=delta_col)
    heatmap, pivot = _build_deviation_heatmap(work)
    alerts, alert_cols = _build_deviation_alerts(work, threshold_bps=float(threshold_bps), venue_col=venue_col, expiry_col=expiry_col, delta_col=delta_col)
    summary = {"n_rows": int(len(work)), "n_alerts": int(len(alerts)), "max_abs_deviation_bps": float(work["abs_deviation_bps"].max()), "mean_abs_deviation_bps": float(work["abs_deviation_bps"].mean()), "threshold_bps": float(threshold_bps)}
    return {
        "summary": summary,
        "heatmap_matrix": pivot.to_dict(),
        "heatmap_records": heatmap.to_dict(orient="records"),
        "alerts": alerts[alert_cols].head(50).to_dict(orient="records"),
        "columns": {
            "market_price_column": market_col,
            "model_price_column": model_col,
            "expiry_column": expiry_col,
            "delta_column": delta_col,
            "venue_column": venue_col,
        },
    }


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


def _build_file_options(files: List[Path], selected: Path) -> str:
    return "\n".join(
        f'<option value="{path.name}" {"selected" if path == selected else ""}>{path.name}</option>'
        for path in files
    )


def _build_summary_rows(summary: Dict[str, Any]) -> str:
    return "".join(
        (
            f"<tr><th>{key}</th><td>{value:.6f}</td></tr>"
            if isinstance(value, float)
            else f"<tr><th>{key}</th><td>{value}</td></tr>"
        )
        for key, value in summary.items()
    )


def _build_deviation_section(df: pd.DataFrame) -> str:
    try:
        deviation_report = build_cross_market_deviation_report(df, threshold_bps=300.0)
    except HTTPException:
        return ""

    heatmap_df = pd.DataFrame(deviation_report["heatmap_records"])
    if heatmap_df.empty:
        return ""

    heatmap_fig = px.density_heatmap(
        heatmap_df,
        x="delta_bucket",
        y="expiry_bucket",
        z="abs_deviation_bps",
        histfunc="avg",
        color_continuous_scale="RdBu_r",
        title="Cross-Market Deviation Heatmap (abs bps)",
    )
    alerts = deviation_report["alerts"]
    if alerts:
        header = "".join(f"<th>{k}</th>" for k in alerts[0].keys())
        rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
            for row in alerts[:20]
        )
        alerts_table = f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
    else:
        alerts_table = "<p>No alerts over 300 bps.</p>"

    return (
        f'<div class="card">{heatmap_fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        f'<div class="card"><h2>Deviation Alerts</h2>{alerts_table}</div>'
    )


def _resolve_dashboard_axis(df: pd.DataFrame, time_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Return plotting frame and x-axis column for dashboard charts."""
    if time_col:
        return df, time_col
    indexed = df.reset_index(drop=False).rename(columns={"index": "index"})
    return indexed, "index"


def _build_dashboard_summary(df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    """Compute summary stats for dashboard metrics card."""
    return {
        "rows": len(df),
        "value_column": value_col,
        "mean": float(df[value_col].mean()),
        "std": float(df[value_col].std(ddof=0) if len(df) > 1 else 0.0),
        "min": float(df[value_col].min()),
        "max": float(df[value_col].max()),
    }


def _render_dashboard_html(
    *, file_options: str,
    primary_fig_html: str,
    returns_fig_html: str,
    deviation_section: str, summary_rows: str,
) -> str:
    """Render full dashboard HTML page."""
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>CORP Research Dashboard</title>
    <style>body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #1f2937; background: #f4f6f8; }}
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
      <div class=\"card\">{primary_fig_html}</div><div class=\"card\">{returns_fig_html}</div>
      {deviation_section}
      <div class=\"card\"><h2>Summary</h2><table>{summary_rows}</table></div>
    </div>
  </body>
</html>
"""


def _build_dashboard_html(df: pd.DataFrame, selected: Path, files: List[Path]) -> str:
    time_col = _find_time_column(df)
    value_col = _find_value_column(df)
    if value_col is None:
        raise HTTPException(status_code=422, detail="No numeric column available for plotting")

    plot_df, x_axis = _resolve_dashboard_axis(df, time_col)

    primary_fig = px.line(
        plot_df,
        x=x_axis,
        y=value_col,
        title=f"{selected.name}: {value_col}",
        template="plotly_white",
    )

    returns = plot_df[value_col].pct_change().dropna()
    returns_fig = px.histogram(
        returns,
        nbins=40,
        title=f"Return Distribution ({value_col})",
        template="plotly_white",
    )

    summary = _build_dashboard_summary(plot_df, value_col)

    file_options = _build_file_options(files, selected)
    summary_rows = _build_summary_rows(summary)
    deviation_section = _build_deviation_section(plot_df)
    return _render_dashboard_html(
        file_options=file_options,
        primary_fig_html=primary_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        returns_fig_html=returns_fig.to_html(full_html=False, include_plotlyjs=False),
        deviation_section=deviation_section,
        summary_rows=summary_rows,
    )


async def _build_live_deviation_report(
    *,
    threshold_bps: float,
    align_tolerance_seconds: float,
    cex_file: Optional[str],
    cex_provider: Optional[str],
    underlying: str,
    defi_file: Optional[str],
) -> dict:
    """Build live/file-based CEX-DEFI deviation report payload."""
    cex_source = cex_file or os.getenv("CEX_QUOTES_FILE", "")
    provider = cex_provider or os.getenv("CEX_QUOTES_PROVIDER", "")
    defi_source = defi_file or os.getenv("DEFI_QUOTES_FILE", "")
    if not defi_source:
        raise HTTPException(status_code=422, detail="defi_file is required (query or env)")
    if not cex_source and not provider:
        raise HTTPException(status_code=422, detail="Either cex_file or cex_provider is required (query or env)")
    try:
        if cex_source:
            dataset = build_cex_defi_deviation_dataset(
                Path(cex_source),
                Path(defi_source),
                align_tolerance_seconds=align_tolerance_seconds,
            )
            source_meta = {"mode": "file", "cex_file": cex_source, "defi_file": defi_source}
        else:
            dataset = await build_cex_defi_deviation_dataset_live(
                provider,
                Path(defi_source),
                underlying=underlying,
                align_tolerance_seconds=align_tolerance_seconds,
            )
            source_meta = {"mode": "provider", "cex_provider": provider, "underlying": underlying, "defi_file": defi_source}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    report = build_cross_market_deviation_report(dataset, threshold_bps=float(threshold_bps))
    report["sources"] = {
        **source_meta,
        "align_tolerance_seconds": float(align_tolerance_seconds),
        "rows_aligned": int(len(dataset)),
    }
    return report


def _register_basic_routes(app: FastAPI, directory: Path) -> None:
    """Register health and file-list endpoints."""

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict:
        return {"status": "healthy", "results_dir": str(directory)}

    @app.get("/api/files", response_class=JSONResponse)
    async def files() -> dict:
        return {"files": [path.name for path in available_result_files(directory)]}


def _register_deviation_routes(app: FastAPI, directory: Path) -> None:
    """Register historical and live deviation endpoints."""

    @app.get("/api/deviation", response_class=JSONResponse)
    async def deviation(
        file: Optional[str] = Query(default=None),
        threshold_bps: float = Query(default=300.0, ge=0.0),
    ) -> dict:
        df, _ = _load_results_dataframe(directory, file)
        return build_cross_market_deviation_report(df, threshold_bps=float(threshold_bps))

    @app.get("/api/deviation/live", response_class=JSONResponse)
    async def deviation_live(
        threshold_bps: float = Query(default=300.0, ge=0.0),
        align_tolerance_seconds: float = Query(default=60.0, ge=0.0),
        cex_file: Optional[str] = Query(default=None),
        cex_provider: Optional[str] = Query(default=None),
        underlying: str = Query(default="BTC-USD"),
        defi_file: Optional[str] = Query(default=None),
    ) -> dict:
        return await _build_live_deviation_report(
            threshold_bps=float(threshold_bps),
            align_tolerance_seconds=float(align_tolerance_seconds),
            cex_file=cex_file,
            cex_provider=cex_provider,
            underlying=underlying,
            defi_file=defi_file,
        )


def _register_dashboard_route(app: FastAPI, directory: Path) -> None:
    """Register HTML dashboard endpoint."""

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(file: Optional[str] = Query(default=None)) -> HTMLResponse:
        df, selected = _load_results_dataframe(directory, file)
        html = _build_dashboard_html(df, selected, available_result_files(directory))
        return HTMLResponse(content=html)


def create_dashboard_app(results_dir: Optional[Path] = None) -> FastAPI:
    """Create dashboard API application."""
    directory = results_dir or DEFAULT_RESULTS_DIR
    app = FastAPI(title="CORP Research Dashboard", version="1.0.0")
    _register_basic_routes(app, directory)
    _register_deviation_routes(app, directory)
    _register_dashboard_route(app, directory)
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
