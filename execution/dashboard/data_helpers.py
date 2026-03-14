"""Data loading, file discovery, and column detection utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

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


def _load_results_dataframe(
    results_dir: Path, file_name: Optional[str]
) -> Tuple[pd.DataFrame, Path]:
    candidates = available_result_files(results_dir)
    if not candidates:
        raise HTTPException(status_code=404, detail="No results files found")
    selected = (results_dir / file_name) if file_name else candidates[0]
    if file_name and selected not in candidates:
        raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
    df = pd.read_parquet(selected) if selected.suffix == ".parquet" else pd.read_csv(selected)
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


# --- JSON backtest result helpers ---

def load_backtest_json(path: Path) -> Dict[str, Any]:
    """Load a backtest results JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_json_results(results_dir: Path) -> List[Dict[str, str]]:
    """Scan results/ subdirectories for JSON backtest files."""
    entries: List[Dict[str, str]] = []
    for subdir in ("backtest_with_output", "backtest_full_history"):
        folder = results_dir / subdir
        if not folder.is_dir():
            continue
        for json_file in sorted(folder.glob("*.json"), reverse=True):
            entries.append({
                "name": json_file.name,
                "subdir": subdir,
                "path": str(json_file.relative_to(results_dir)),
            })
    return entries
