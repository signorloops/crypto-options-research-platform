#!/usr/bin/env python3
"""Runtime/load helpers for weekly operating audit scripts."""

from __future__ import annotations

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


def load_threshold_map(path: Path, defaults: dict[str, float], *, label: str) -> dict[str, float]:
    if not path.exists():
        return dict(defaults)
    raw = _load_json(path)
    thresholds = dict(defaults)
    for key in defaults:
        if key in raw:
            value = _to_float(raw[key])
            if value is None:
                raise ValueError(f"Invalid {label} value for '{key}'")
            thresholds[key] = float(value)
    return thresholds


def load_optional_report(path: Path, *, missing_error: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "executed": False,
            "summary": {"all_passed": None},
            "error": missing_error,
            "path": str(resolved),
        }
    try:
        report = _load_json(resolved)
    except JSON_REPORT_EXCEPTIONS as exc:
        return {
            "executed": False,
            "summary": {"all_passed": None},
            "error": str(exc),
            "path": str(resolved),
        }
    if "executed" not in report:
        report["executed"] = True
    report["path"] = str(resolved)
    report["error"] = ""
    return report
