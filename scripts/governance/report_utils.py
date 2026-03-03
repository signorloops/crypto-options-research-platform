"""Shared helpers for governance report scripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

JSON_REPORT_EXCEPTIONS = (
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    ValueError,
    TypeError,
    KeyError,
)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load JSON file and ensure top-level object is a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


def load_optional_json_object(path: Path) -> dict[str, Any]:
    """Load optional JSON object path, returning empty dict when missing."""
    if not path.exists():
        return {}
    return load_json_object(path)


def as_bool(value: Any) -> bool:
    """Convert mixed scalar values into normalized boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def discover_input_files(results_dir: Path, pattern: str) -> list[Path]:
    """Discover matching files sorted by mtime descending."""
    candidates = sorted(results_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p for p in candidates if p.is_file()]


def format_markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Render simple markdown table from row mappings."""
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body])
