"""
Tests for weekly research-audit summary renderer.
"""

from validation_scripts.research_audit_weekly_summary import render_weekly_summary


def test_render_weekly_summary_contains_core_metrics_and_pass_status():
    iv_report = {
        "summary": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.02,
        }
    }
    model_report = {
        "results": [
            {"model": "bates", "rmse": 55.0, "mae": 38.0},
        ]
    }
    drift_report = {
        "current_generated_at": "2026-02-25T00:00:00+00:00",
        "passed": True,
        "model_zoo": {
            "current_best_model": "bates",
            "current_best_rmse": 55.0,
            "best_rmse_increase_pct": 3.2,
        },
        "iv_surface": {
            "baseline_avg_max_jump_reduction_short": 0.03,
            "avg_max_jump_reduction_drop_pct": 10.5,
        },
        "violations": [],
    }

    markdown = render_weekly_summary(iv_report, model_report, drift_report)

    assert "# Research Audit Weekly Card" in markdown
    assert "- Status: `PASS`" in markdown
    assert "| Best model | `bates` |" in markdown
    assert "| Best RMSE increase vs baseline | `3.200000%` |" in markdown
    assert "## Violations" in markdown
    assert "- None" in markdown


def test_render_weekly_summary_shows_fail_violations():
    iv_report = {"summary": {"no_arbitrage": False, "avg_max_jump_reduction_short": 0.0}}
    model_report = {"results": []}
    drift_report = {
        "current_generated_at": "2026-02-25T00:00:00+00:00",
        "passed": False,
        "model_zoo": {
            "current_best_model": "heston",
            "current_best_rmse": 180.0,
            "best_rmse_increase_pct": 60.0,
        },
        "iv_surface": {
            "baseline_avg_max_jump_reduction_short": 0.03,
            "avg_max_jump_reduction_drop_pct": 80.0,
        },
        "violations": ["Best-model RMSE increase too large"],
    }

    markdown = render_weekly_summary(iv_report, model_report, drift_report)

    assert "- Status: `FAIL`" in markdown
    assert "- Best-model RMSE increase too large" in markdown
