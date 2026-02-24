"""
Tests for IV surface stability report quality gates.
"""

from validation_scripts.iv_surface_stability_report import evaluate_quality_gates


def test_quality_gates_pass_when_thresholds_satisfied():
    report = {
        "summary": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.02,
        }
    }
    violations = evaluate_quality_gates(
        report=report,
        fail_on_arbitrage=True,
        min_short_max_jump_reduction=0.01,
    )
    assert violations == []


def test_quality_gates_fail_on_arbitrage_when_enabled():
    report = {
        "summary": {
            "no_arbitrage": False,
            "avg_max_jump_reduction_short": 0.02,
        }
    }
    violations = evaluate_quality_gates(
        report=report,
        fail_on_arbitrage=True,
        min_short_max_jump_reduction=0.01,
    )
    assert any("No-arbitrage" in violation for violation in violations)


def test_quality_gates_fail_on_short_jump_reduction_threshold():
    report = {
        "summary": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.005,
        }
    }
    violations = evaluate_quality_gates(
        report=report,
        fail_on_arbitrage=False,
        min_short_max_jump_reduction=0.01,
    )
    assert any("below threshold" in violation for violation in violations)
