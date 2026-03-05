"""Cross-venue quote integration helpers for CEX vs DeFi monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
PRICE_CANDIDATES = ("market_price", "quote_price", "option_price", "price")
DELTA_CANDIDATES = ("delta", "abs_delta", "delta_abs")
EXPIRY_CANDIDATES = ("expiry_years", "maturity", "time_to_expiry", "tau")
SYMBOL_CANDIDATES = ("symbol", "instrument", "contract")
OPTION_TYPE_CANDIDATES = ("option_type", "type", "right")
VENUE_CANDIDATES = ("venue", "exchange", "source", "market")


def _find_first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
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


def _quote_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "time_col": _find_first_existing(df, TIMESTAMP_CANDIDATES),
        "price_col": _find_first_existing(df, PRICE_CANDIDATES),
        "symbol_col": _find_first_existing(df, SYMBOL_CANDIDATES),
        "option_type_col": _find_first_existing(df, OPTION_TYPE_CANDIDATES),
        "expiry_col": _find_first_existing(df, EXPIRY_CANDIDATES),
        "delta_col": _find_first_existing(df, DELTA_CANDIDATES),
        "venue_col": _find_first_existing(df, VENUE_CANDIDATES),
    }


def _timestamp_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column], errors="coerce", utc=True)


def _numeric_series(df: pd.DataFrame, column: str | None, *, default: float) -> pd.Series:
    if column is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _string_series(
    df: pd.DataFrame,
    column: str | None,
    *,
    default: str,
    lowercase: bool = False,
) -> pd.Series:
    series = (
        pd.Series(default, index=df.index, dtype=object)
        if column is None
        else df[column].astype(str)
    )
    return series.str.lower() if lowercase else series


def _normalized_quotes_frame(
    df: pd.DataFrame, *, columns: dict[str, str | None], fallback_venue: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamp_series(df, columns["time_col"]),
            "price": _numeric_series(df, columns["price_col"], default=float("nan")),
            "symbol": _string_series(df, columns["symbol_col"], default="UNKNOWN"),
            "option_type": _string_series(
                df,
                columns["option_type_col"],
                default="unknown",
                lowercase=True,
            ),
            "expiry_years": _numeric_series(df, columns["expiry_col"], default=0.0),
            "delta": _numeric_series(df, columns["delta_col"], default=0.5),
            "venue": _string_series(
                df,
                columns["venue_col"],
                default=fallback_venue,
            ),
        }
    )


def _finalize_normalized_quotes(out: pd.DataFrame) -> pd.DataFrame:
    out = out.dropna(subset=["price"])
    if out.empty:
        raise ValueError("Quote data has no valid rows after normalization")
    out["timestamp"] = out["timestamp"].ffill().bfill()
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        raise ValueError("Quote data requires at least one valid timestamp")
    out["expiry_bucket"] = out["expiry_years"].astype(float).round(4)
    out["delta_bucket"] = out["delta"].astype(float).abs().round(2)
    out["ts_bucket"] = pd.to_datetime(out["timestamp"]).dt.floor("min")
    return out


def _normalize_quotes(df: pd.DataFrame, fallback_venue: str) -> pd.DataFrame:
    columns = _quote_column_map(df)
    if columns["price_col"] is None:
        raise ValueError("Quote data requires a price column")
    out = _normalized_quotes_frame(df, columns=columns, fallback_venue=fallback_venue)
    return _finalize_normalized_quotes(out)


def _normalize_option_type(raw_value: Any, symbol: str) -> str:
    value = str(raw_value).strip().lower() if raw_value is not None else ""
    if value in {"c", "call"}:
        return "call"
    if value in {"p", "put"}:
        return "put"

    suffix = symbol.rsplit("-", 1)[-1].strip().lower()
    if suffix in {"c", "call"}:
        return "call"
    if suffix in {"p", "put"}:
        return "put"
    return "unknown"


def _normalize_okx_option_summary(
    rows: list[dict[str, Any]], underlying: str = "BTC-USD"
) -> pd.DataFrame:
    if not rows:
        raise ValueError("OKX option summary returned no rows")
    now = pd.Timestamp.now(tz="UTC")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = _normalize_okx_option_row(row=row, now=now, underlying=underlying)
        if normalized is not None:
            normalized_rows.append(normalized)
    if not normalized_rows:
        raise ValueError("No valid rows in OKX option summary payload")
    return _normalize_quotes(pd.DataFrame(normalized_rows), fallback_venue="okx")


def _first_present_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _okx_row_price(row: dict[str, Any]) -> float | None:
    price_raw = _first_present_value(
        row,
        ["markPx", "markPrice", "last", "lastPr", "askPx", "bidPx"],
    )
    price = pd.to_numeric(price_raw, errors="coerce")
    if pd.isna(price):
        return None
    return float(price)


def _okx_row_timestamp(row: dict[str, Any], now: pd.Timestamp) -> pd.Timestamp:
    timestamp_raw = _first_present_value(row, ["ts", "uTime", "cTime"])
    timestamp_numeric = pd.to_numeric(timestamp_raw, errors="coerce")
    timestamp = pd.to_datetime(timestamp_numeric, unit="ms", utc=True, errors="coerce")
    if pd.isna(timestamp):
        return now
    return timestamp


def _okx_row_expiry_years(row: dict[str, Any], now: pd.Timestamp) -> float:
    expiry_ts = pd.to_datetime(
        pd.to_numeric(row.get("expTime"), errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    if pd.isna(expiry_ts):
        return 0.0
    return max((expiry_ts - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)


def _okx_row_delta(row: dict[str, Any]) -> float:
    delta = pd.to_numeric(row.get("delta"), errors="coerce")
    if pd.isna(delta):
        return 0.5
    return float(delta)


def _normalize_okx_option_row(
    *, row: dict[str, Any], now: pd.Timestamp, underlying: str
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    price = _okx_row_price(row)
    if price is None:
        return None
    symbol = str(row.get("instId") or row.get("instFamily") or underlying)
    option_type = _normalize_option_type(row.get("optType"), symbol)
    timestamp = _okx_row_timestamp(row, now)
    expiry_years = _okx_row_expiry_years(row, now)
    delta = _okx_row_delta(row)
    return {
        "timestamp": timestamp,
        "price": price,
        "symbol": symbol,
        "option_type": option_type,
        "expiry_years": float(expiry_years),
        "delta": delta,
        "venue": "okx",
    }


def _merge_quotes_by_bucket_keys(cex: pd.DataFrame, defi: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["ts_bucket", "symbol", "option_type", "expiry_bucket", "delta_bucket"]
    return cex.merge(defi, on=join_cols, suffixes=("_cex", "_defi"))


def _merge_quotes_by_nearest_timestamp(
    cex: pd.DataFrame,
    defi: pd.DataFrame,
    *,
    align_tolerance_seconds: float,
) -> pd.DataFrame:
    key_cols = ["symbol", "option_type", "expiry_bucket", "delta_bucket"]
    tolerance = pd.Timedelta(seconds=max(float(align_tolerance_seconds), 0.0))
    cex_asof = cex.sort_values([*key_cols, "timestamp"])
    defi_asof = defi.sort_values([*key_cols, "timestamp"])
    merged = pd.merge_asof(
        cex_asof,
        defi_asof,
        on="timestamp",
        by=key_cols,
        direction="nearest",
        tolerance=tolerance,
        suffixes=("_cex", "_defi"),
    )
    merged = merged.dropna(subset=["price_defi"])
    if not merged.empty:
        merged["ts_bucket"] = merged["timestamp"].dt.floor("min")
    return merged


def _aligned_timestamp(merged: pd.DataFrame) -> pd.Series:
    if "ts_bucket" in merged.columns:
        return merged["ts_bucket"]
    return merged["timestamp"]


def _aligned_quotes_output(merged: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _aligned_timestamp(merged),
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


def _align_cex_defi_quotes(
    cex: pd.DataFrame,
    defi: pd.DataFrame,
    *,
    align_tolerance_seconds: float = 60.0,
) -> pd.DataFrame:
    merged = _merge_quotes_by_bucket_keys(cex, defi)
    if merged.empty:
        merged = _merge_quotes_by_nearest_timestamp(
            cex,
            defi,
            align_tolerance_seconds=align_tolerance_seconds,
        )
    if merged.empty:
        raise ValueError("No aligned CEX/DeFi rows after key-based merge")
    return _aligned_quotes_output(merged).sort_values("timestamp")


def build_cex_defi_deviation_dataset(
    cex_path: Path,
    defi_path: Path,
    *,
    align_tolerance_seconds: float = 60.0,
) -> pd.DataFrame:
    """Build aligned CEX-vs-DeFi quote pairs for deviation analysis."""
    cex_raw = _load_table(cex_path)
    defi_raw = _load_table(defi_path)

    cex = _normalize_quotes(cex_raw, fallback_venue="cex")
    defi = _normalize_quotes(defi_raw, fallback_venue="defi")
    return _align_cex_defi_quotes(
        cex,
        defi,
        align_tolerance_seconds=align_tolerance_seconds,
    )


async def build_cex_defi_deviation_dataset_live(
    cex_provider: str,
    defi_path: Path,
    *,
    underlying: str = "BTC-USD",
    align_tolerance_seconds: float = 60.0,
) -> pd.DataFrame:
    """Build CEX-vs-DeFi deviation dataset using a live CEX provider plus DeFi file source."""
    provider = cex_provider.strip().lower()
    if provider != "okx":
        raise ValueError(f"Unsupported cex_provider: {cex_provider}")

    from data.downloaders.okx import OKXClient

    client = OKXClient()
    try:
        cex_raw_rows = await client.get_option_market_data(underlying=underlying)
    finally:
        await client.disconnect()

    cex = _normalize_okx_option_summary(cex_raw_rows, underlying=underlying)
    defi_raw = _load_table(defi_path)
    defi = _normalize_quotes(defi_raw, fallback_venue="defi")
    return _align_cex_defi_quotes(
        cex,
        defi,
        align_tolerance_seconds=align_tolerance_seconds,
    )
