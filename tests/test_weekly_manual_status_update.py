"""Tests for explicit weekly manual status updates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "governance"
    / "weekly_manual_status_update.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "weekly_manual_status_update_test_module",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_manual_status_update module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_main_applies_explicit_checks_and_signoffs(tmp_path, monkeypatch):
    module = _load_module()
    decision_json = tmp_path / "weekly-decision-log.json"
    manual_json = tmp_path / "weekly-manual-status.json"
    output_md = tmp_path / "weekly-manual-status.md"

    _write(
        decision_json,
        json.dumps(
            {
                "decision": "APPROVE_CANARY",
                "rollback": {"reference": "backup-release-demo", "source": "tag"},
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_manual_status_update.py",
            "--decision-json",
            str(decision_json),
            "--manual-status-json",
            str(manual_json),
            "--output-md",
            str(output_md),
            "--check",
            "gray_release_completed=true",
            "--check",
            "observation_24h_completed=yes",
            "--signoff",
            "research=alice",
            "--signoff",
            "engineering=bob",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    payload = json.loads(manual_json.read_text(encoding="utf-8"))
    assert payload["gray_release_completed"] is True
    assert payload["observation_24h_completed"] is True
    assert payload["signoffs"]["research"] == "alice"
    assert payload["signoffs"]["engineering"] == "bob"
    assert payload["signoffs"]["risk"] == ""

    markdown = output_md.read_text(encoding="utf-8")
    assert "- Decision: `APPROVE_CANARY`" in markdown
    assert "- [x] 灰度发布完成" in markdown
    assert "- [x] 24h 观察完成" in markdown
    assert "- [x] Research 签字: `alice`" in markdown
    assert "- [ ] Risk 签字: `TBD`" in markdown


def test_main_can_clear_existing_signoff(tmp_path, monkeypatch):
    module = _load_module()
    manual_json = tmp_path / "weekly-manual-status.json"

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
                "signoffs": {"research": "alice", "engineering": "bob", "risk": "carol"},
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_manual_status_update.py",
            "--manual-status-json",
            str(manual_json),
            "--clear-signoff",
            "engineering",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    payload = json.loads(manual_json.read_text(encoding="utf-8"))
    assert payload["signoffs"]["research"] == "alice"
    assert payload["signoffs"]["engineering"] == ""
    assert payload["signoffs"]["risk"] == "carol"


def test_invalid_manual_check_assignment_is_rejected():
    module = _load_module()

    try:
        module._parse_check_assignment("unknown=true")
    except argparse.ArgumentTypeError as exc:
        assert "unsupported manual check key" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_invalid_boolean_value_is_rejected():
    module = _load_module()

    try:
        module._parse_check_assignment("gray_release_completed=maybe")
    except argparse.ArgumentTypeError as exc:
        assert "Unsupported boolean value" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_invalid_signoff_role_is_rejected():
    module = _load_module()

    try:
        module._parse_signoff_assignment("ops=alice")
    except argparse.ArgumentTypeError as exc:
        assert "unsupported signoff role" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_placeholder_signoff_value_is_rejected():
    module = _load_module()

    try:
        module._parse_signoff_assignment("research=research_owner")
    except argparse.ArgumentTypeError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")
