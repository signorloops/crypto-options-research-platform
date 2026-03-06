"""Tests for governance Makefile orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_dry_run(*args: str) -> str:
    completed = subprocess.run(
        ["make", "-n", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_live_deviation_snapshot_target_uses_fixture_inputs_by_default():
    stdout = _make_dry_run("live-deviation-snapshot")

    assert "--cex-file tests/fixtures/live_deviation/governance_cex.csv" in stdout
    assert "--defi-file tests/fixtures/live_deviation/governance_defi.csv" in stdout


def test_weekly_operating_audit_runs_live_snapshot_before_consistency_replay():
    stdout = _make_dry_run("weekly-operating-audit")

    live_index = stdout.index("make live-deviation-snapshot")
    replay_index = stdout.index("make weekly-consistency-replay")
    assert live_index < replay_index


def test_weekly_manual_update_target_routes_arguments_to_script():
    stdout = _make_dry_run(
        "weekly-manual-update",
        "MANUAL_ARGS=--check gray_release_completed=true --signoff research=alice",
    )

    assert "scripts/governance/weekly_manual_status_update.py" in stdout
    assert "--manual-status-json artifacts/weekly-manual-status.json" in stdout
    assert "--output-md artifacts/weekly-manual-status.md" in stdout
    assert "--check gray_release_completed=true --signoff research=alice" in stdout
