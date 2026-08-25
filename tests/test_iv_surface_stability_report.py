"""
Tests for IV surface stability report quality gates.
"""

from validation_scripts.iv_surface_stability_report import evaluate_quality_gates
from validation_scripts.iv_surface_stability_report import (
    _attach_runtime_metadata,
    _cache_key,
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


def test_cache_key_is_deterministic_and_varies_with_seed():
    assert _cache_key(42) == _cache_key(42)
    assert _cache_key(42) != _cache_key(43)


def test_cache_key_changes_when_audit_source_changes(tmp_path, monkeypatch):
    # Fixed behaviour: the old seed-only key served stale reports whenever
    # _build_synthetic_surface or audit_surface_stability changed, because the
    # key never covered the implementation. The key is folded with a hash of the
    # report and audit sources, so editing either must rotate the key.
    import types

    import validation_scripts.iv_surface_stability_report as report_module

    audit_source = tmp_path / "surface_audit.py"
    audit_source.write_text("def audit_surface_stability():\n    return {}\n", encoding="utf-8")
    fake_audit_module = types.ModuleType("fake_surface_audit")
    fake_audit_module.__file__ = str(audit_source)
    monkeypatch.setattr(report_module, "_surface_audit_module", fake_audit_module)

    key_before = report_module._cache_key(42)

    audit_source.write_text(
        "def audit_surface_stability():\n    return {'changed': True}\n", encoding="utf-8"
    )
    key_after_edit = report_module._cache_key(42)

    assert key_before != key_after_edit

    audit_source.write_text("def audit_surface_stability():\n    return {}\n", encoding="utf-8")
    assert report_module._cache_key(42) == key_before


def test_main_rebuilds_and_self_heals_on_corrupt_cache(tmp_path, monkeypatch, capsys):
    # Fixed behaviour: a corrupt/partially-written cache file used to raise and
    # abort the whole report run. It must fall back to a rebuild and rewrite the
    # cache so the next run hits it again.
    import json as json_module
    import sys as sys_module

    import validation_scripts.iv_surface_stability_report as report_module

    cache_dir = tmp_path / "cache" / "iv"
    cache_dir.mkdir(parents=True)
    corrupt_cache = tmp_path / "cache" / "iv" / f"{report_module._cache_key(7)}.json"
    corrupt_cache.write_text('{"summary": {"partial', encoding="utf-8")

    output_md = tmp_path / "report.md"
    output_json = tmp_path / "report.json"
    monkeypatch.setattr(
        sys_module,
        "argv",
        [
            "iv_surface_stability_report.py",
            "--seed",
            "7",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--fast-calibration",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    report_module.main()

    report = json_module.loads(output_json.read_text(encoding="utf-8"))
    assert report["summary"]["cache_hit"] is False
    assert report["summary"]["avg_max_jump_reduction_short"] > 0.0
    # The corrupt cache must have been replaced with a valid report.
    healed = json_module.loads(corrupt_cache.read_text(encoding="utf-8"))
    assert "cache_hit" not in healed["summary"]
    assert "ignoring unreadable cache file" in capsys.readouterr().err

    # Second run with the healed cache must hit the fast path.
    report_module.main()
    second_report = json_module.loads(output_json.read_text(encoding="utf-8"))
    assert second_report["summary"]["cache_hit"] is True
