"""Shared helper functions for weekly close-gate report assembly."""

from __future__ import annotations

from typing import Any


def collect_open_labels(items: Any) -> list[str]:
    """Collect non-empty labels from incomplete checklist/signoff entries."""
    if not isinstance(items, list):
        return []
    return [
        label
        for entry in items
        if isinstance(entry, dict) and not bool(entry.get("done"))
        if (label := str(entry.get("label", "")).strip())
    ]


def build_close_gate_summary(
    *,
    auto_blockers: list[str],
    pending_items: list[str],
    manual_missing: list[str],
    role_signoffs_missing: list[str],
) -> dict[str, int]:
    """Build top-line counts for close-gate report sections."""
    return {
        "auto_blockers": len(auto_blockers),
        "pending_items": len(pending_items),
        "manual_missing": len(manual_missing),
        "role_signoffs_missing": len(role_signoffs_missing),
    }
