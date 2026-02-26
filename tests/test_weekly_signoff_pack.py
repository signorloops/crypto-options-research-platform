"""Tests for weekly manual sign-off pack generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_signoff_pack.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_signoff_pack_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_signoff_pack module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_marks_pending_manual_items():
    module = _load_module()
    report = module._build_report(
        audit={
            "summary": {"exceptions": 0, "consistency_exceptions": 0},
            "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
            "incomplete_tasks": ["灰度发布完成", "ADR"],
        },
        canary={"recommendation": "PROCEED_CANARY", "blockers": []},
        decision={"decision": "APPROVE_CANARY", "follow_up_tasks": ["灰度发布完成", "ADR"]},
        attribution={"attribution_snapshot": [{"strategy": "demo"}]},
        manual_status={
            "gray_release_completed": True,
            "observation_24h_completed": False,
            "rollback_decision_recorded": True,
            "pnl_attribution_confirmed": False,
            "change_and_rollback_recorded": False,
            "adr_signed": False,
            "signoffs": {"research": "alice"},
        },
    )

    assert report["status"] == "PENDING_MANUAL_SIGNOFF"
    assert "24h 观察完成" in report["pending_items"]
    assert "收益归因表确认" in report["pending_items"]


def test_main_strict_returns_nonzero_when_pending(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "audit.json"
    canary_json = tmp_path / "canary.json"
    decision_json = tmp_path / "decision.json"
    attribution_json = tmp_path / "attribution.json"
    manual_json = tmp_path / "manual.json"

    _write(
        audit_json,
        json.dumps(
            {
                "summary": {"exceptions": 0, "consistency_exceptions": 0},
                "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
                "incomplete_tasks": ["灰度发布完成", "ADR"],
            }
        ),
    )
    _write(canary_json, json.dumps({"recommendation": "PROCEED_CANARY", "blockers": []}))
    _write(decision_json, json.dumps({"decision": "APPROVE_CANARY", "follow_up_tasks": []}))
    _write(attribution_json, json.dumps({"attribution_snapshot": []}))
    _write(manual_json, json.dumps({"gray_release_completed": False}))

    output_md = tmp_path / "weekly-signoff-pack.md"
    output_json = tmp_path / "weekly-signoff-pack.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_signoff_pack.py",
            "--audit-json",
            str(audit_json),
            "--canary-json",
            str(canary_json),
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
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
    assert generated["status"] == "PENDING_MANUAL_SIGNOFF"


def test_main_strict_returns_zero_when_all_confirmed(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "audit.json"
    canary_json = tmp_path / "canary.json"
    decision_json = tmp_path / "decision.json"
    attribution_json = tmp_path / "attribution.json"
    manual_json = tmp_path / "manual.json"

    _write(
        audit_json,
        json.dumps(
            {
                "summary": {"exceptions": 0, "consistency_exceptions": 0},
                "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
                "incomplete_tasks": [],
            }
        ),
    )
    _write(canary_json, json.dumps({"recommendation": "PROCEED_CANARY", "blockers": []}))
    _write(decision_json, json.dumps({"decision": "APPROVE_CANARY", "follow_up_tasks": []}))
    _write(attribution_json, json.dumps({"attribution_snapshot": [{"strategy": "mm"}]}))
    _write(
        manual_json,
        json.dumps(
            {
                "gray_release_completed": True,
                "observation_24h_completed": True,
                "rollback_decision_recorded": True,
                "pnl_attribution_confirmed": True,
                "change_and_rollback_recorded": True,
                "adr_signed": True,
                "signoffs": {
                    "research": "alice",
                    "engineering": "bob",
                    "risk": "carol",
                },
            }
        ),
    )

    output_md = tmp_path / "weekly-signoff-pack.md"
    output_json = tmp_path / "weekly-signoff-pack.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_signoff_pack.py",
            "--audit-json",
            str(audit_json),
            "--canary-json",
            str(canary_json),
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    generated = json.loads(output_json.read_text(encoding="utf-8"))
    assert generated["status"] == "READY_FOR_CLOSE"


def test_main_creates_manual_template_when_missing(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "audit.json"
    canary_json = tmp_path / "canary.json"
    decision_json = tmp_path / "decision.json"
    attribution_json = tmp_path / "attribution.json"
    manual_json = tmp_path / "manual.json"

    _write(
        audit_json,
        json.dumps(
            {
                "summary": {"exceptions": 0, "consistency_exceptions": 0},
                "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
                "incomplete_tasks": [],
            }
        ),
    )
    _write(canary_json, json.dumps({"recommendation": "PROCEED_CANARY", "blockers": []}))
    _write(decision_json, json.dumps({"decision": "APPROVE_CANARY", "follow_up_tasks": []}))
    _write(attribution_json, json.dumps({"attribution_snapshot": []}))

    output_md = tmp_path / "weekly-signoff-pack.md"
    output_json = tmp_path / "weekly-signoff-pack.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_signoff_pack.py",
            "--audit-json",
            str(audit_json),
            "--canary-json",
            str(canary_json),
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert manual_json.exists()
    manual_status = json.loads(manual_json.read_text(encoding="utf-8"))
    assert manual_status["gray_release_completed"] is False
    assert manual_status["signoffs"]["risk"] == ""
