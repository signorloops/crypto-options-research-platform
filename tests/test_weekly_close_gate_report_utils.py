"""Tests for weekly close-gate report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.governance.weekly_close_gate_report_utils import (
    build_close_gate_report,
    evaluate_close_gate,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_evaluate_close_gate_reports_missing_invalid_and_ready_states(tmp_path):
    missing_path = tmp_path / "missing.json"
    invalid_path = tmp_path / "invalid.json"
    ready_path = tmp_path / "ready.json"
    blocked_path = tmp_path / "blocked.json"

    _write(invalid_path, "{broken")
    _write(ready_path, json.dumps({"status": "READY_FOR_CLOSE"}))
    _write(blocked_path, json.dumps({"status": "PENDING_MANUAL_SIGNOFF"}))

    assert evaluate_close_gate(missing_path) == (False, "missing_signoff_json", {})
    assert evaluate_close_gate(invalid_path) == (False, "invalid_signoff_json", {})
    assert evaluate_close_gate(ready_path) == (
        True,
        "READY_FOR_CLOSE",
        {"status": "READY_FOR_CLOSE"},
    )
    assert evaluate_close_gate(blocked_path) == (
        False,
        "status=PENDING_MANUAL_SIGNOFF",
        {"status": "PENDING_MANUAL_SIGNOFF"},
    )


def test_build_close_gate_report_collects_pending_groups_and_actions(tmp_path):
    report = build_close_gate_report(
        signoff_json_path=tmp_path / "signoff.json",
        close_ready=False,
        close_detail="status=PENDING_MANUAL_SIGNOFF",
        signoff_payload={
            "status": "PENDING_MANUAL_SIGNOFF",
            "auto_blockers": ["latency baseline passed"],
            "pending_items": ["ADR"],
            "manual_items": [{"label": "Gray release", "done": False}],
            "role_signoffs": [{"label": "Research", "done": False}],
        },
    )

    assert report["status"] == "FAIL"
    assert report["summary"]["auto_blockers"] == 1
    assert report["summary"]["manual_missing"] == 1
    assert report["summary"]["role_signoffs_missing"] == 1
    assert "Next actions:" in report["pr_brief"]
