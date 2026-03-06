"""Tests for GitHub weekly operating audit workflow drift."""

from __future__ import annotations

from pathlib import Path


def test_weekly_workflow_runs_single_source_of_truth_make_target():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "weekly-operating-audit.yml"
    ).read_text(encoding="utf-8")

    assert 'make weekly-operating-audit ADR_OWNER="weekly-ci"' in workflow
    assert "python scripts/governance/weekly_operating_audit.py \\" not in workflow


def test_weekly_workflow_uploads_extended_governance_artifacts():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "weekly-operating-audit.yml"
    ).read_text(encoding="utf-8")

    assert "artifacts/algorithm-performance-baseline.json" in workflow
    assert "artifacts/performance/latency_benchmark_report.json" in workflow
    assert "artifacts/live-deviation-snapshot.json" in workflow
    assert "artifacts/online-offline-consistency-replay.json" in workflow
    assert "artifacts/weekly-manual-status.md" in workflow
