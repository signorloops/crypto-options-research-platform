"""Tests for weekly canary rollout checklist generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_canary_checklist.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "weekly_canary_checklist_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_canary_checklist module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_recommends_proceed_when_no_blockers():
    module = _load_module()
    audit = {
        "summary": {"exceptions": 0, "consistency_exceptions": 0},
        "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
        "regression": {"passed": True},
        "thresholds": {"min_sharpe": 0.5},
        "rollback_marker": {"tag": "v0.1.0"},
        "kpi_snapshot": [],
    }
    attribution = {"attribution_snapshot": []}

    report = module._build_report(audit, attribution)

    assert report["recommendation"] == "PROCEED_CANARY"
    assert report["blockers"] == []


def test_main_strict_exits_nonzero_when_hold(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "audit.json"
    attribution_json = tmp_path / "attr.json"
    _write(
        audit_json,
        json.dumps(
            {
                "summary": {"exceptions": 1, "consistency_exceptions": 0},
                "checklist": {"minimum_regression_passed": True, "rollback_version_marked": True},
                "regression": {"passed": True},
                "thresholds": {},
                "rollback_marker": {"tag": "v0.1.0"},
                "kpi_snapshot": [],
            }
        ),
    )
    _write(attribution_json, json.dumps({"attribution_snapshot": []}))

    output_md = tmp_path / "weekly-canary-checklist.md"
    output_json = tmp_path / "weekly-canary-checklist.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_canary_checklist.py",
            "--audit-json",
            str(audit_json),
            "--attribution-json",
            str(attribution_json),
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
    assert generated["recommendation"] == "HOLD"
