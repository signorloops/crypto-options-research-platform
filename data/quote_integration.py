"""Cross-venue quote integration helpers for CEX vs DeFi monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
PRICE_CANDIDATES = ("market_price", "quote_price", "option_price", "price")
DELTA_CANDIDATES = ("delta", "abs_delta", "delta_abs")
EXPIRY_CANDIDATES = ("expiry_years", "maturity", "time_to_expiry", "tau")
SYMBOL_CANDIDATES = ("symbol", "instrument", "contract")
OPTION_TYPE_CANDIDATES = ("option_type", "type", "right")
VENUE_CANDIDATES = ("venue", "exchange", "source", "market")


def _find_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Quote source not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_quotes(df: pd.DataFrame, fallback_venue: str) -> pd.DataFrame:
    time_col = _find_first_existing(df, TIMESTAMP_CANDIDATES)
    price_col = _find_first_existing(df, PRICE_CANDIDATES)
    if price_col is None:
        raise ValueError("Quote data requires a price column")

    symbol_col = _find_first_existing(df, SYMBOL_CANDIDATES)
    option_type_col = _find_first_existing(df, OPTION_TYPE_CANDIDATES)
    expiry_col = _find_first_existing(df, EXPIRY_CANDIDATES)
    delta_col = _find_first_existing(df, DELTA_CANDIDATES)
    venue_col = _find_first_existing(df, VENUE_CANDIDATES)

    out = pd.DataFrame()
    if time_col is not None:
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        out["timestamp"] = pd.NaT

    out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    out["symbol"] = df[symbol_col].astype(str) if symbol_col else "UNKNOWN"
    out["option_type"] = df[option_type_col].astype(str).str.lower() if option_type_col else "unknown"
    out["expiry_years"] = pd.to_numeric(df[expiry_col], errors="coerce") if expiry_col else 0.0
    out["delta"] = pd.to_numeric(df[delta_col], errors="coerce") if delta_col else 0.5
    out["venue"] = df[venue_col].astype(str) if venue_col else fallback_venue

    out = out.dropna(subset=["price"])
    if out.empty:
        raise ValueError("Quote data has no valid rows after normalization")

    out["timestamp"] = out["timestamp"].fillna(method="ffill").fillna(method="bfill")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        raise ValueError("Quote data requires at least one valid timestamp")

    out["expiry_bucket"] = out["expiry_years"].astype(float).round(4)
    out["delta_bucket"] = out["delta"].astype(float).abs().round(2)
    out["ts_bucket"] = pd.to_datetime(out["timestamp"]).dt.floor("min")
    return out


def build_cex_defi_deviation_dataset(cex_path: Path, defi_path: Path) -> pd.DataFrame:
    """Build aligned CEX-vs-DeFi quote pairs for deviation analysis."""
    cex_raw = _load_table(cex_path)
    defi_raw = _load_table(defi_path)

    cex = _normalize_quotes(cex_raw, fallback_venue="cex")
    defi = _normalize_quotes(defi_raw, fallback_venue="defi")

    join_cols = ["ts_bucket", "symbol", "option_type", "expiry_bucket", "delta_bucket"]
    merged = cex.merge(defi, on=join_cols, suffixes=("_cex", "_defi"))
    if merged.empty:
        raise ValueError("No aligned CEX/DeFi rows after key-based merge")

    out = pd.DataFrame(
        {
            "timestamp": merged["ts_bucket"],
            "symbol": merged["symbol"],
            "option_type": merged["option_type"],
            "maturity": merged["expiry_bucket"],
            "delta": merged["delta_bucket"],
            "market_price": merged["price_cex"],
            "model_price": merged["price_defi"],
            "cex_venue": merged["venue_cex"],
            "defi_venue": merged["venue_defi"],
            "venue": "cex_vs_defi",
        }
    )
    return out.sort_values("timestamp")
