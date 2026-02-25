"""Tests for weekly decision log generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_decision_log.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_decision_log_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_decision_log module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_returns_approve_when_canary_proceeds():
    module = _load_module()
    audit = {
        "summary": {"exceptions": 0, "consistency_exceptions": 0},
        "checklist": {
            "minimum_regression_passed": True,
            "rollback_version_marked": True,
            "rollback_marker_from_tag": False,
        },
        "rollback_marker": {"tag": "HEAD-abc123", "source": "commit"},
        "incomplete_tasks": [],
    }
    canary = {"recommendation": "PROCEED_CANARY", "blockers": [], "warnings": []}

    report = module._build_report(audit, canary)

    assert report["decision"] == "APPROVE_CANARY"
    assert report["rollback"]["reference"] == "HEAD-abc123"


def test_main_strict_returns_nonzero_when_hold(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "audit.json"
    canary_json = tmp_path / "canary.json"
    _write(
        audit_json,
        json.dumps(
            {
                "summary": {"exceptions": 1, "consistency_exceptions": 0},
                "checklist": {
                    "minimum_regression_passed": True,
                    "rollback_version_marked": True,
                    "rollback_marker_from_tag": False,
                },
                "rollback_marker": {"tag": "HEAD-abc123", "source": "commit"},
                "incomplete_tasks": ["灰度发布完成"],
            }
        ),
    )
    _write(
        canary_json,
        json.dumps({"recommendation": "HOLD", "blockers": ["risk_exceptions=1"], "warnings": []}),
    )

    output_md = tmp_path / "weekly-decision-log.md"
    output_json = tmp_path / "weekly-decision-log.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_decision_log.py",
            "--audit-json",
            str(audit_json),
            "--canary-json",
            str(canary_json),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    generated = json.loads(output_json.read_text(encoding="utf-8"))
    assert generated["decision"] == "HOLD_AND_REMEDIATE"
