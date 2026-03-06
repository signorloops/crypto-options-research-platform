"""Tests for weekly close-gate helper utilities."""

from __future__ import annotations

from scripts.governance.weekly_close_gate_utils import (
    build_close_gate_summary,
    collect_open_labels,
)


def test_collect_open_labels_filters_blank_and_completed_entries():
    labels = collect_open_labels(
        [
            {"label": " Gray release ", "done": False},
            {"label": "", "done": False},
            {"label": "ADR", "done": True},
            "skip",
        ]
    )

    assert labels == ["Gray release"]


def test_build_close_gate_summary_counts_all_pending_groups():
    summary = build_close_gate_summary(
        auto_blockers=["performance baseline passed"],
        pending_items=["ADR"],
        manual_missing=["Gray release"],
        role_signoffs_missing=["Research"],
    )

    assert summary == {
        "auto_blockers": 1,
        "pending_items": 1,
        "manual_missing": 1,
        "role_signoffs_missing": 1,
    }
