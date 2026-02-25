"""
Tests for IV surface stability report quality gates.
"""

from validation_scripts.iv_surface_stability_report import evaluate_quality_gates
from validation_scripts.iv_surface_stability_report import (
    _attach_runtime_metadata,
    _strip_runtime_metadata,
)


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


def test_runtime_metadata_attach_and_strip_roundtrip():
    report = {
        "summary": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.02,
        }
    }
    enriched = _attach_runtime_metadata(
        report=report,
        fast_calibration=True,
        cache_hit=True,
        cache_key="k1",
        calibration_latency_sec=0.123,
    )

    assert enriched["summary"]["fast_calibration"] is True
    assert enriched["summary"]["cache_hit"] is True
    assert enriched["summary"]["cache_key"] == "k1"
    assert enriched["summary"]["calibration_latency_sec"] == 0.123

    stripped = _strip_runtime_metadata(enriched)
    assert "fast_calibration" not in stripped["summary"]
    assert "cache_hit" not in stripped["summary"]
    assert "cache_key" not in stripped["summary"]
    assert "calibration_latency_sec" not in stripped["summary"]
