"""Tests for online/offline consistency replay report generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "governance"
    / "online_offline_consistency_replay.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "online_offline_consistency_replay_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load online_offline_consistency_replay module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_pass_when_within_thresholds():
    module = _load_module()
    report = module._build_report(
        audit={
            "summary": {
                "consistency_exceptions": 0,
                "consistency_pairs": 2,
            }
        },
        live={
            "summary": {
                "n_rows": 4,
                "n_alerts": 0,
                "max_abs_deviation_bps": 120.0,
                "mean_abs_deviation_bps": 45.0,
            },
            "alerts": [],
        },
        thresholds={
            "max_offline_consistency_exceptions": 0,
            "max_online_alerts": 0,
            "max_online_abs_deviation_bps": 300.0,
        },
    )

    assert report["status"] == "PASS"
    assert report["breaches"] == []
    assert report["summary"]["data_ready"] is True


def test_build_report_fail_when_online_deviation_too_high():
    module = _load_module()
    report = module._build_report(
        audit={"summary": {"consistency_exceptions": 0, "consistency_pairs": 1}},
        live={
            "summary": {
                "n_rows": 1,
                "n_alerts": 1,
                "max_abs_deviation_bps": 600.0,
                "mean_abs_deviation_bps": 600.0,
            },
            "alerts": [
                {"venue": "cex_vs_defi", "maturity": 0.05, "delta": 0.25, "deviation_bps": 600.0}
            ],
        },
        thresholds={
            "max_offline_consistency_exceptions": 0,
            "max_online_alerts": 0,
            "max_online_abs_deviation_bps": 300.0,
        },
    )

    assert report["status"] == "FAIL"
    assert any("online_max_abs_deviation_bps" in item for item in report["breaches"])
    assert report["summary"]["data_ready"] is True


def test_main_non_strict_allows_missing_live_snapshot(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "weekly-operating-audit.json"
    live_json = tmp_path / "live-deviation-snapshot.json"
    thresholds_json = tmp_path / "thresholds.json"
    output_md = tmp_path / "online-offline-consistency-replay.md"
    output_json = tmp_path / "online-offline-consistency-replay.json"

    _write(audit_json, json.dumps({"summary": {"consistency_exceptions": 0, "consistency_pairs": 0}}))
    _write(
        thresholds_json,
        json.dumps(
            {
                "max_offline_consistency_exceptions": 0,
                "max_online_alerts": 0,
                "max_online_abs_deviation_bps": 300.0,
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_offline_consistency_replay.py",
            "--audit-json",
            str(audit_json),
            "--live-json",
            str(live_json),
            "--thresholds",
            str(thresholds_json),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "PENDING_DATA"
    assert report["summary"]["data_ready"] is False


def test_main_strict_fails_when_pending_data(tmp_path, monkeypatch):
    module = _load_module()
    audit_json = tmp_path / "weekly-operating-audit.json"
    live_json = tmp_path / "live-deviation-snapshot.json"
    output_md = tmp_path / "online-offline-consistency-replay.md"
    output_json = tmp_path / "online-offline-consistency-replay.json"

    _write(audit_json, json.dumps({"summary": {"consistency_exceptions": 0, "consistency_pairs": 0}}))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_offline_consistency_replay.py",
            "--audit-json",
            str(audit_json),
            "--live-json",
            str(live_json),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--strict",
        ],
    )

    exit_code = module.main()
    assert exit_code == 2
