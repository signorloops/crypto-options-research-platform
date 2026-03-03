"""Tests for performance latency benchmark script behavior."""

from __future__ import annotations

import importlib.util
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
