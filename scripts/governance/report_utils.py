"""Shared helpers for governance report scripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_json_object(path: Path) -> dict[str, Any]:
    """Load JSON file and ensure top-level object is a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


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
