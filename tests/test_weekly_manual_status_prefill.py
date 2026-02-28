"""Tests for weekly manual status auto-prefill helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "governance"
    / "weekly_manual_status_prefill.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "weekly_manual_status_prefill_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_manual_status_prefill module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_main_prefills_objective_fields(tmp_path, monkeypatch):
    module = _load_module()
    decision_json = tmp_path / "weekly-decision-log.json"
    attribution_json = tmp_path / "weekly-pnl-attribution.json"
    manual_json = tmp_path / "weekly-manual-status.json"

    _write(
        decision_json,
        json.dumps(
            {
                "decision": "APPROVE_CANARY",
                "rollback": {"reference": "HEAD-abc123", "source": "commit"},
            }
        ),
    )
    _write(
        attribution_json,
        json.dumps({"summary": {"strategies": 2, "missing_entries": 0}}),
    )
    _write(
        manual_json,
        json.dumps(
            {
                "gray_release_completed": False,
                "observation_24h_completed": False,
                "rollback_decision_recorded": False,
                "pnl_attribution_confirmed": False,
                "change_and_rollback_recorded": False,
                "adr_signed": False,
                "signoffs": {"research": "", "engineering": "", "risk": ""},
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_manual_status_prefill.py",
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0

    updated = json.loads(manual_json.read_text(encoding="utf-8"))
    assert updated["rollback_decision_recorded"] is True
    assert updated["change_and_rollback_recorded"] is True
    assert updated["pnl_attribution_confirmed"] is True
    assert updated["gray_release_completed"] is False
    assert updated["adr_signed"] is False
    assert updated["signoffs"]["risk"] == ""


def test_main_does_not_downgrade_existing_true_values(tmp_path, monkeypatch):
    module = _load_module()
    decision_json = tmp_path / "weekly-decision-log.json"
    attribution_json = tmp_path / "weekly-pnl-attribution.json"
    manual_json = tmp_path / "weekly-manual-status.json"

    _write(decision_json, json.dumps({"decision": "", "rollback": {"reference": ""}}))
    _write(
        attribution_json,
        json.dumps({"summary": {"strategies": 2, "missing_entries": 2}}),
    )
    _write(
        manual_json,
        json.dumps(
            {
                "gray_release_completed": False,
                "observation_24h_completed": False,
                "rollback_decision_recorded": True,
                "pnl_attribution_confirmed": True,
                "change_and_rollback_recorded": True,
                "adr_signed": False,
                "signoffs": {"research": "", "engineering": "", "risk": ""},
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_manual_status_prefill.py",
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0

    updated = json.loads(manual_json.read_text(encoding="utf-8"))
    assert updated["rollback_decision_recorded"] is True
    assert updated["change_and_rollback_recorded"] is True
    assert updated["pnl_attribution_confirmed"] is True


def test_main_creates_template_when_manual_file_missing(tmp_path, monkeypatch):
    module = _load_module()
    decision_json = tmp_path / "weekly-decision-log.json"
    attribution_json = tmp_path / "weekly-pnl-attribution.json"
    manual_json = tmp_path / "weekly-manual-status.json"

    _write(decision_json, json.dumps({"decision": ""}))
    _write(attribution_json, json.dumps({"summary": {"strategies": 0, "missing_entries": 0}}))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_manual_status_prefill.py",
            "--decision-json",
            str(decision_json),
            "--attribution-json",
            str(attribution_json),
            "--manual-status-json",
            str(manual_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0
    assert manual_json.exists()

    generated = json.loads(manual_json.read_text(encoding="utf-8"))
    assert generated["gray_release_completed"] is False
    assert generated["rollback_decision_recorded"] is False
    assert generated["signoffs"]["research"] == ""
