"""Cross-venue quote integration helpers for CEX vs DeFi monitoring."""

from __future__ import annotations

from pathlib import Path

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
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.NaT

    out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    out["symbol"] = df[symbol_col].astype(str) if symbol_col else "UNKNOWN"
    out["option_type"] = (
        df[option_type_col].astype(str).str.lower() if option_type_col else "unknown"
    )
    out["expiry_years"] = pd.to_numeric(df[expiry_col], errors="coerce") if expiry_col else 0.0
    out["delta"] = pd.to_numeric(df[delta_col], errors="coerce") if delta_col else 0.5
    out["venue"] = df[venue_col].astype(str) if venue_col else fallback_venue

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
        if not isinstance(row, dict):
            continue

        price_raw = (
            row.get("markPx")
            or row.get("markPrice")
            or row.get("last")
            or row.get("lastPr")
            or row.get("askPx")
            or row.get("bidPx")
        )
        price = pd.to_numeric(price_raw, errors="coerce")
        if pd.isna(price):
            continue

        symbol = str(row.get("instId") or row.get("instFamily") or underlying)
        option_type = _normalize_option_type(row.get("optType"), symbol)

        timestamp_raw = row.get("ts") or row.get("uTime") or row.get("cTime")
        if timestamp_raw is not None:
            timestamp_numeric = pd.to_numeric(timestamp_raw, errors="coerce")
            timestamp = pd.to_datetime(timestamp_numeric, unit="ms", utc=True, errors="coerce")
        else:
            timestamp = pd.NaT
        if pd.isna(timestamp):
            timestamp = now

        expiry_years = 0.0
        expiry_raw = row.get("expTime")
        if expiry_raw is not None:
            expiry_numeric = pd.to_numeric(expiry_raw, errors="coerce")
            expiry_ts = pd.to_datetime(expiry_numeric, unit="ms", utc=True, errors="coerce")
            if not pd.isna(expiry_ts):
                expiry_years = max((expiry_ts - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)

        delta = pd.to_numeric(row.get("delta"), errors="coerce")
        if pd.isna(delta):
            delta = 0.5

        normalized_rows.append(
            {
                "timestamp": timestamp,
                "price": float(price),
                "symbol": symbol,
                "option_type": option_type,
                "expiry_years": float(expiry_years),
                "delta": float(delta),
                "venue": "okx",
            }
        )

    if not normalized_rows:
        raise ValueError("No valid rows in OKX option summary payload")
    return _normalize_quotes(pd.DataFrame(normalized_rows), fallback_venue="okx")


def _align_cex_defi_quotes(
    cex: pd.DataFrame,
    defi: pd.DataFrame,
    *,
    align_tolerance_seconds: float = 60.0,
) -> pd.DataFrame:
    join_cols = ["ts_bucket", "symbol", "option_type", "expiry_bucket", "delta_bucket"]
    merged = cex.merge(defi, on=join_cols, suffixes=("_cex", "_defi"))
    if merged.empty:
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
    if merged.empty:
        raise ValueError("No aligned CEX/DeFi rows after key-based merge")

    out = pd.DataFrame(
        {
            "timestamp": (
                merged["ts_bucket"] if "ts_bucket" in merged.columns else merged["timestamp"]
            ),
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
