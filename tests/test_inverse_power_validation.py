"""Tests for inverse-power validation script."""

import subprocess
from pathlib import Path

from validation_scripts.inverse_power_validation import (
    build_validation_grid,
    render_markdown,
    run_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_validation_grid_shape():
    grid = build_validation_grid()
    assert len(grid) == 3 * 3 * 3 * 2 * 2 * 2
    assert {"S", "K", "T", "sigma", "r", "option_type"}.issubset(grid[0].keys())


def test_run_validation_outputs_summary_fields():
    report = run_validation(n_paths=6000, seed=42)
    summary = report["summary"]

    assert report["n_cases"] > 0
    assert summary["max_abs_error"] >= 0.0
    assert summary["mean_abs_error"] >= 0.0
    assert summary["max_rel_error"] >= 0.0


def test_render_markdown_contains_key_metrics():
    report = {
        "generated_at": "2026-02-25T00:00:00+00:00",
        "n_cases": 2,
        "n_paths": 10000,
        "seed": 42,
        "summary": {
            "max_abs_error": 1e-4,
            "mean_abs_error": 2e-5,
            "p95_abs_error": 9e-5,
            "max_rel_error": 0.05,
            "mean_rel_error": 0.01,
        },
    }
    markdown = render_markdown(report)

    assert "# Inverse-Power Validation Report" in markdown
    assert "| Max abs error |" in markdown
    assert "| Mean rel error |" in markdown


def test_inverse_power_validation_script_runs_as_standalone(tmp_path: Path):
    output_md = tmp_path / "inverse-power-validation-report.md"
    output_json = tmp_path / "inverse-power-validation-report.json"

    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "validation_scripts/inverse_power_validation.py",
            "--n-paths",
            "1024",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_md.exists()
    assert output_json.exists()
