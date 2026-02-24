"""
Tests for research audit snapshot builder.
"""

import json

from validation_scripts.research_audit_snapshot import (
    build_snapshot,
    parse_rough_jump_report,
)


def test_parse_rough_jump_report(tmp_path):
    report_path = tmp_path / "rough.txt"
    report_path.write_text(
        "mode price ci_low ci_high avg_jump_events_per_path avg_jump_intensity "
        "jump_intensity_std simulation_time_sec pricing_time_sec total_time_sec\n"
        "none 3.7 3.3 4.0 0.0 0.0 0.0 0.01 0.001 0.02\n"
        "clustered 4.3 3.8 4.8 2.4 4.9 1.1 0.01 0.001 0.02\n",
        encoding="utf-8",
    )

    parsed = parse_rough_jump_report(str(report_path))
    assert set(parsed.keys()) == {"none", "clustered"}
    assert parsed["clustered"]["price"] == 4.3
    assert parsed["none"]["avg_jump_events_per_path"] == 0.0


def test_build_snapshot_compacts_key_metrics():
    iv_report = {
        "summary": {
            "no_arbitrage": True,
            "avg_max_jump_reduction_short": 0.03,
            "avg_mean_jump_reduction_short": 0.02,
        }
    }
    model_report = {
        "quotes_source": "json:fixture",
        "n_quotes": 20,
        "results": [
            {"model": "bates", "rmse": 55.3, "mae": 38.2},
            {"model": "heston", "rmse": 270.5, "mae": 224.8},
        ],
    }
    rough_jump = {
        "none": {
            "price": 3.7,
            "ci_low": 3.3,
            "ci_high": 4.0,
            "avg_jump_events_per_path": 0.0,
            "total_time_sec": 0.01,
        }
    }

    snapshot = build_snapshot(
        iv_report=iv_report,
        model_report=model_report,
        rough_jump_by_mode=rough_jump,
    )

    assert snapshot["iv_surface"]["no_arbitrage"] is True
    assert snapshot["model_zoo"]["best_model"] == "bates"
    assert snapshot["model_zoo"]["best_rmse"] == 55.3
    assert "none" in snapshot["rough_jump"]
    json.dumps(snapshot)  # ensure serializable
