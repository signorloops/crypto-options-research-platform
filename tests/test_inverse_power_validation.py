"""Tests for inverse-power validation script."""

import subprocess
from pathlib import Path

from validation_scripts.inverse_power_validation import (
    DEFAULT_MAX_REL_ERROR,
    REL_ERROR_DENOMINATOR_FLOOR,
    build_validation_grid,
    render_markdown,
    run_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gate_defaults_are_meaningful_for_inverse_pricing():
    """Gate defaults must be able to fail on realistic inverse-price scales.

    Regression guard: the script used to gate only on max_abs_error=6e-4 while
    inverse premiums are ~2e-6 BTC, so the gate could never fail. The relative
    gate is the primary accuracy check; the absolute gate is a NaN tripwire.
    """
    # A 5% relative tolerance on a ~2e-6 BTC premium is ~1e-7 BTC, far below
    # the 6e-4 absolute gate, i.e. the relative gate binds first.
    assert DEFAULT_MAX_REL_ERROR == 0.05
    assert REL_ERROR_DENOMINATOR_FLOOR < 2e-6


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
    # The relative gate only scopes cases above the denominator floor; the
    # report must expose that scope so a failure can be interpreted.
    assert summary["rel_gate_floor"] > 0.0
    assert 0 < summary["n_rel_gate_cases"] <= report["n_cases"]


def test_run_validation_relative_gate_excludes_sub_floor_premiums():
    """max_rel_error must exclude premiums below the rel-error denominator floor.

    Deep-OTM inverse premiums are ~1e-10 BTC, where relative Monte Carlo error
    is pure noise (rel ~0.3 regardless of accuracy). Including them would make
    a 5% relative gate unpassable; they are covered by the absolute gate.
    """
    report = run_validation(n_paths=6000, seed=42)
    from validation_scripts.inverse_power_validation import (
        REL_ERROR_DENOMINATOR_FLOOR,
    )

    gated = [
        c["rel_error"]
        for c in report["cases"]
        if abs(c["closed_form"]) >= REL_ERROR_DENOMINATOR_FLOOR
    ]
    assert len(gated) == report["summary"]["n_rel_gate_cases"]
    assert report["summary"]["max_rel_error"] == max(gated)


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
            "rel_gate_floor": 1e-7,
            "n_rel_gate_cases": 173,
        },
    }
    markdown = render_markdown(report)

    assert "# Inverse-Power Validation Report" in markdown
    assert "| Max abs error |" in markdown
    assert "| Mean rel error |" in markdown
    assert "| Max rel error |" in markdown
    assert "| Rel gate floor |" in markdown


def test_inverse_power_validation_script_runs_as_standalone(tmp_path: Path):
    output_md = tmp_path / "inverse-power-validation-report.md"
    output_json = tmp_path / "inverse-power-validation-report.json"

    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "validation_scripts/inverse_power_validation.py",
            "--n-paths",
            "1024",
            # At 1024 paths Monte Carlo noise drives max_rel_error to ~0.3, so
            # the standalone smoke run uses a loosened relative gate.
            "--max-rel-error",
            "1.0",
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
