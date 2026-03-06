"""Tests for performance latency benchmark script behavior."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "performance" / "latency_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("latency_benchmark_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load latency_benchmark module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regime_detector_benchmark_does_not_mutate_global_numpy_rng():
    module = _load_module()
    np.random.seed(20260304)
    _ = np.random.random()
    expected_next = np.random.random()

    np.random.seed(20260304)
    _ = np.random.random()
    bench = module.LatencyBenchmark(iterations=1)
    bench.benchmark_regime_detector()
    actual_next = np.random.random()

    assert actual_next == expected_next


def test_regime_detector_benchmark_uses_steady_state_retrain_interval(monkeypatch):
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=10)
    captured = {}

    class _DummyDetector:
        def __init__(self, config=None):
            captured["config"] = config

        def update(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(module, "VolatilityRegimeDetector", _DummyDetector)
    monkeypatch.setattr(bench, "_warmup_regime_detector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bench, "_measure", lambda func, *args, **kwargs: 1.0)

    bench.benchmark_regime_detector()

    assert captured["config"] is not None
    assert captured["config"].retrain_interval > bench.iterations


def test_constructor_rejects_non_positive_iterations():
    module = _load_module()
    with pytest.raises(ValueError):
        module.LatencyBenchmark(iterations=0)


def test_main_rejects_non_positive_iterations(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "latency_benchmark.py",
            "--iterations",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_script_runs_without_needing_pythonpath(tmp_path):
    report_path = tmp_path / "latency-benchmark.md"
    env = dict(os.environ)
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--iterations",
            "1",
            "--benchmarks",
            "greeks_calculation",
            "--quiet",
            "--report-path",
            str(report_path),
        ],
        cwd=SCRIPT_PATH.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()


def test_quiet_benchmark_suppresses_inner_stdout_and_stderr(monkeypatch, capsys):
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=1)
    bench._quiet = True

    def _noisy_price_and_greeks(*_args, **_kwargs):
        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)
        return None

    monkeypatch.setattr(
        module.InverseOptionPricer,
        "calculate_price_and_greeks",
        staticmethod(_noisy_price_and_greeks),
    )

    bench.benchmark_greeks_calculation()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_writes_json_report_and_strictly_fails_on_target_miss(tmp_path, monkeypatch):
    module = _load_module()
    report_path = tmp_path / "latency-benchmark.md"
    json_path = tmp_path / "latency-benchmark.json"

    def _fake_run_all(self, quiet=False, selected_benchmarks=None):
        self.results = {"greeks_calculation": [1.0]}
        self.last_all_passed = False
        return module.pd.DataFrame(
            [
                {
                    "name": "Greeks Calculation",
                    "target_ms": 15.0,
                    "mean_ms": 20.0,
                    "median_ms": 20.0,
                    "p50_ms": 20.0,
                    "p95_ms": 20.0,
                    "p99_ms": 20.0,
                    "min_ms": 20.0,
                    "max_ms": 20.0,
                    "std_ms": 0.0,
                    "meets_target": False,
                }
            ]
        )

    monkeypatch.setattr(module.LatencyBenchmark, "run_all_benchmarks", _fake_run_all)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "latency_benchmark.py",
            "--iterations",
            "1",
            "--report-path",
            str(report_path),
            "--output-json",
            str(json_path),
            "--fail-on-target-miss",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
    payload = module.json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["all_passed"] is False
    assert payload["benchmarks"][0]["name"] == "Greeks Calculation"


def test_default_latency_targets_are_tightened():
    module = _load_module()

    assert module.DEFAULT_LATENCY_TARGETS_MS == {
        "greeks_calculation": 4.0,
        "circuit_breaker": 2.0,
        "regime_detector": 3.0,
        "adaptive_hedger": 1.0,
        "quote_generation": 10.0,
        "end_to_end": 100.0,
    }


def test_run_all_benchmarks_collects_gc_between_benchmarks(monkeypatch):
    module = _load_module()
    events = []

    monkeypatch.setattr(module.gc, "collect", lambda: events.append("collect"))

    bench = module.LatencyBenchmark(iterations=1)
    monkeypatch.setattr(
        bench,
        "_benchmark_registry",
        lambda: {
            "demo": lambda: {
                "name": "Demo",
                "target_ms": 1.0,
                "mean_ms": 0.1,
                "median_ms": 0.1,
                "p50_ms": 0.1,
                "p95_ms": 0.1,
                "p99_ms": 0.1,
                "min_ms": 0.1,
                "max_ms": 0.1,
                "std_ms": 0.0,
                "meets_target": True,
            }
        },
    )

    bench.run_all_benchmarks(quiet=True)

    assert events == ["collect"]


def test_summarize_can_exclude_warmup_samples_from_percentiles():
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=4)

    summary = bench._summarize(
        "End-to-End",
        [200.0, 40.0, 42.0, 44.0],
        target_ms=100.0,
        warmup_samples=1,
    )

    assert summary["samples_excluded"] == 1
    assert summary["sample_count"] == 3
    assert summary["max_ms"] == 44.0
    assert summary["p95_ms"] < 100.0
    assert summary["meets_target"] is True


def test_end_to_end_summary_excludes_warmup_samples(monkeypatch):
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=10)

    class _DummyStrategy:
        def __init__(self, config=None):
            self.regime_detector = object()

        def quote(self, *_args, **_kwargs):
            return None

    captured = {}

    monkeypatch.setattr(module, "IntegratedMarketMakingStrategy", _DummyStrategy)
    monkeypatch.setattr(bench, "_measure", lambda func, *args, **kwargs: 1.0)
    monkeypatch.setattr(bench, "_warmup_regime_detector", lambda *_args, **_kwargs: None)

    def _capture(name, latencies, target_ms, warmup_samples=0):
        captured["name"] = name
        captured["warmup_samples"] = warmup_samples
        return {
            "name": name,
            "target_ms": target_ms,
            "mean_ms": 1.0,
            "median_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
            "min_ms": 1.0,
            "max_ms": 1.0,
            "std_ms": 0.0,
            "meets_target": True,
            "samples_excluded": warmup_samples,
            "sample_count": len(latencies) - warmup_samples,
        }

    monkeypatch.setattr(bench, "_summarize", _capture)

    bench.benchmark_end_to_end()

    assert captured["name"] == "End-to-End"
    assert captured["warmup_samples"] > 0


def test_end_to_end_benchmark_uses_risk_stable_strategy_config(monkeypatch):
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=1)
    captured = {}

    class _DummyStrategy:
        def __init__(self, config=None):
            captured["config"] = config
            self.regime_detector = object()

        def quote(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(module, "IntegratedMarketMakingStrategy", _DummyStrategy)
    monkeypatch.setattr(bench, "_measure", lambda func, *args, **kwargs: 1.0)
    monkeypatch.setattr(bench, "_warmup_regime_detector", lambda *_args, **_kwargs: None)

    bench.benchmark_end_to_end()

    assert captured["config"] is not None
    assert captured["config"].initial_capital >= 1_000_000.0
    assert captured["config"].regime_detector.retrain_interval > bench.iterations


def test_end_to_end_benchmark_warms_quote_pipeline_before_measuring(monkeypatch):
    module = _load_module()
    bench = module.LatencyBenchmark(iterations=5)
    observed = {"quote_calls": 0, "measure_calls": 0}

    class _DummyStrategy:
        def __init__(self, config=None):
            self.regime_detector = object()

        def quote(self, *_args, **_kwargs):
            observed["quote_calls"] += 1
            return None

    monkeypatch.setattr(module, "IntegratedMarketMakingStrategy", _DummyStrategy)
    monkeypatch.setattr(bench, "_warmup_regime_detector", lambda *_args, **_kwargs: None)

    def _measure(func, *args, **kwargs):
        observed["measure_calls"] += 1
        func(*args, **kwargs)
        return 1.0

    monkeypatch.setattr(bench, "_measure", _measure)

    bench.benchmark_end_to_end()

    assert observed["measure_calls"] == 5
    assert observed["quote_calls"] > observed["measure_calls"]
