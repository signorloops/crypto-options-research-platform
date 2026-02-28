"""Tests for weekly ADR draft generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_adr_draft.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_adr_draft_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_adr_draft module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_main_generates_adr_markdown(tmp_path, monkeypatch):
    module = _load_module()
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "adr.md"
    _write(
        audit_path,
        json.dumps(
            {
                "generated_at_utc": "2026-02-25T00:00:00Z",
                "summary": {"strategies": 2, "exceptions": 0},
                "checklist": {
                    "minimum_regression_passed": True,
                    "change_log_complete": True,
                    "rollback_version_marked": True,
                },
                "regression": {"passed": True},
                "rollback_marker": {"tag": "HEAD-abc12345"},
                "change_log": {
                    "entries": [
                        {
                            "date": "2026-02-25",
                            "commit": "abc12345",
                            "subject": "feat: weekly audit",
                        }
                    ]
                },
                "risk_exceptions": [],
                "incomplete_tasks": ["ADR"],
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_adr_draft.py",
            "--audit-json",
            str(audit_path),
            "--output-md",
            str(output_path),
            "--owner",
            "quant-team",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "Weekly ADR Draft" in text
    assert "Proceed with staged rollout" in text
    assert "HEAD-abc12345" in text
    assert "- [ ] ADR" in text
