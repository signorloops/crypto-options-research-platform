"""Deviation analysis page — historical and live CEX/DeFi deviation endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from data.quote_integration import (
    build_cex_defi_deviation_dataset,
    build_cex_defi_deviation_dataset_live,
)
from execution.dashboard.data_helpers import (
    DELTA_CANDIDATES,
    EXPIRY_CANDIDATES,
    MARKET_PRICE_CANDIDATES,
    MODEL_PRICE_CANDIDATES,
    VENUE_CANDIDATES,
    _find_first_existing,
    _load_results_dataframe,
)


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
    work = _prepare_deviation_frame(
        df, market_col=market_col, model_col=model_col,
        expiry_col=expiry_col, delta_col=delta_col,
    )
    heatmap, pivot = _build_deviation_heatmap(work)
    alerts, alert_cols = _build_deviation_alerts(
        work, threshold_bps=float(threshold_bps),
        venue_col=venue_col, expiry_col=expiry_col, delta_col=delta_col,
    )
    summary = {
        "n_rows": int(len(work)),
        "n_alerts": int(len(alerts)),
        "max_abs_deviation_bps": float(work["abs_deviation_bps"].max()),
        "mean_abs_deviation_bps": float(work["abs_deviation_bps"].mean()),
        "threshold_bps": float(threshold_bps),
    }
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


# ── Live deviation helpers ────────────────────────────────────────

def _resolve_live_deviation_sources(
    *,
    cex_file: Optional[str],
    cex_provider: Optional[str],
    defi_file: Optional[str],
) -> Tuple[str, str, str]:
    cex_source = cex_file or os.getenv("CEX_QUOTES_FILE", "")
    provider = cex_provider or os.getenv("CEX_QUOTES_PROVIDER", "")
    defi_source = defi_file or os.getenv("DEFI_QUOTES_FILE", "")
    if not defi_source:
        raise HTTPException(status_code=422, detail="defi_file is required (query or env)")
    if not cex_source and not provider:
        raise HTTPException(
            status_code=422,
            detail="Either cex_file or cex_provider is required (query or env)",
        )
    return cex_source, provider, defi_source


def _load_live_dataset_from_file(
    *,
    cex_source: str,
    defi_source: str,
    align_tolerance_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset = build_cex_defi_deviation_dataset(
        Path(cex_source),
        Path(defi_source),
        align_tolerance_seconds=align_tolerance_seconds,
    )
    source_meta = {"mode": "file", "cex_file": cex_source, "defi_file": defi_source}
    return dataset, source_meta


async def _load_live_dataset_from_provider(
    *,
    provider: str,
    defi_source: str,
    underlying: str,
    align_tolerance_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset = await build_cex_defi_deviation_dataset_live(
        provider,
        Path(defi_source),
        underlying=underlying,
        align_tolerance_seconds=align_tolerance_seconds,
    )
    source_meta = {
        "mode": "provider",
        "cex_provider": provider,
        "underlying": underlying,
        "defi_file": defi_source,
    }
    return dataset, source_meta


async def _load_live_deviation_dataset_and_meta(
    *,
    cex_source: str,
    provider: str,
    defi_source: str,
    underlying: str,
    align_tolerance_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        if cex_source:
            return _load_live_dataset_from_file(
                cex_source=cex_source,
                defi_source=defi_source,
                align_tolerance_seconds=align_tolerance_seconds,
            )
        return await _load_live_dataset_from_provider(
            provider=provider,
            defi_source=defi_source,
            underlying=underlying,
            align_tolerance_seconds=align_tolerance_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _attach_live_source_metadata(
    *,
    report: Dict[str, Any],
    source_meta: Dict[str, Any],
    align_tolerance_seconds: float,
    rows_aligned: int,
) -> None:
    report["sources"] = {
        **source_meta,
        "align_tolerance_seconds": float(align_tolerance_seconds),
        "rows_aligned": int(rows_aligned),
    }


async def _build_live_deviation_report(
    *,
    threshold_bps: float,
    align_tolerance_seconds: float,
    cex_file: Optional[str],
    cex_provider: Optional[str],
    underlying: str,
    defi_file: Optional[str],
) -> dict:
    cex_source, provider, defi_source = _resolve_live_deviation_sources(
        cex_file=cex_file, cex_provider=cex_provider, defi_file=defi_file,
    )
    dataset, source_meta = await _load_live_deviation_dataset_and_meta(
        cex_source=cex_source, provider=provider, defi_source=defi_source,
        underlying=underlying, align_tolerance_seconds=align_tolerance_seconds,
    )
    report = build_cross_market_deviation_report(dataset, threshold_bps=float(threshold_bps))
    _attach_live_source_metadata(
        report=report, source_meta=source_meta,
        align_tolerance_seconds=align_tolerance_seconds, rows_aligned=len(dataset),
    )
    return report


def register_deviation_routes(app: FastAPI, directory: Path) -> None:
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
