"""
Tests for research-audit snapshot drift comparison.
"""

from validation_scripts.research_audit_compare import compare_snapshots


def _baseline() -> dict:
    return {
        "generated_at": "t0",
        "iv_surface": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.03,
        },
        "model_zoo": {
            "best_model": "bates",
            "best_rmse": 50.0,
        },
        "rough_jump": {},
    }


def test_compare_snapshots_passes_when_within_thresholds():
    baseline = _baseline()
    current = {
        "generated_at": "t1",
        "iv_surface": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.028,
        },
        "model_zoo": {
            "best_model": "bates",
            "best_rmse": 55.0,
        },
        "rough_jump": {},
    }
    diff = compare_snapshots(
        baseline=baseline,
        current=current,
        max_best_rmse_increase_pct=20.0,
        max_iv_reduction_drop_pct=20.0,
    )
    assert diff["passed"] is True
    assert diff["violations"] == []


def test_compare_snapshots_flags_model_change_and_rmse_regression():
    baseline = _baseline()
    current = {
        "generated_at": "t1",
        "iv_surface": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.01,
        },
        "model_zoo": {
            "best_model": "heston",
            "best_rmse": 90.0,
        },
        "rough_jump": {},
    }
    diff = compare_snapshots(
        baseline=baseline,
        current=current,
        max_best_rmse_increase_pct=20.0,
        max_iv_reduction_drop_pct=20.0,
        allow_best_model_change=False,
    )
    assert diff["passed"] is False
    assert any("Best model changed" in issue for issue in diff["violations"])
    assert any("RMSE increase too large" in issue for issue in diff["violations"])
    assert any("stabilization dropped too much" in issue for issue in diff["violations"])
