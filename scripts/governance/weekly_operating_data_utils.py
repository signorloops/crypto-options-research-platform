#!/usr/bin/env python3
"""Data extraction helpers for weekly operating audit reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.governance.report_utils import (
    JSON_REPORT_EXCEPTIONS,
    load_json_object as _load_json,
)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _pick_first_numeric(maps: Sequence[dict[str, Any]], keys: Sequence[str]) -> float | None:
    for mapping in maps:
        for key in keys:
            if key in mapping and (value := _to_float(mapping.get(key))) is not None:
                return value
    return None


def _pick_first_text(maps: Sequence[dict[str, Any]], keys: Sequence[str]) -> str | None:
    for mapping in maps:
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def infer_experiment_id(source: Path) -> str:
    stem = source.stem.strip().replace(" ", "_")
    return f"AUTO-{stem}"


def extract_strategy_rows(raw: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        summary_map = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        metrics_map = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        risk_map = payload.get("risk") if isinstance(payload.get("risk"), dict) else {}
        candidates = [summary_map, metrics_map, risk_map, payload]
        max_dd_raw = _pick_first_numeric(candidates, ["max_drawdown", "max_dd"])
        rows.append(
            {
                "strategy": str(strategy),
                "source_file": source.name,
                "pnl": _pick_first_numeric(candidates, ["total_pnl", "final_pnl", "pnl"]),
                "sharpe": _pick_first_numeric(
                    candidates,
                    ["sharpe_ratio", "sharpe", "deflated_sharpe"],
                ),
                "max_drawdown_abs": abs(max_dd_raw) if max_dd_raw is not None else None,
                "var_breach_rate": _pick_first_numeric(
                    candidates,
                    ["var_breach_rate", "var_exception_rate", "var_breach_ratio", "var_breach"],
                ),
                "fill_calibration_error": _pick_first_numeric(
                    candidates,
                    ["fill_calibration_error", "fill_error", "calibration_error"],
                ),
                "experiment_id": _pick_first_text(
                    candidates,
                    ["experiment_id", "experiment", "exp_id"],
                )
                or infer_experiment_id(source),
            }
        )
    return rows


def collect_strategy_snapshots(
    input_files: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    latest_by_strategy: dict[str, dict[str, Any]] = {}
    previous_by_strategy: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []

    for path in sorted(input_files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            rows = extract_strategy_rows(_load_json(path), path)
        except JSON_REPORT_EXCEPTIONS as exc:
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue
        for row in rows:
            strategy = row["strategy"]
            if strategy not in latest_by_strategy:
                latest_by_strategy[strategy] = row
            elif strategy not in previous_by_strategy:
                previous_by_strategy[strategy] = row
    return latest_by_strategy, previous_by_strategy, parse_errors
