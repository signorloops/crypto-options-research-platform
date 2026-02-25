"""Tests for complexity governance checker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "governance" / "complexity_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("complexity_guard_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load complexity_guard module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _budget(*, max_function_loc: int = 1000) -> dict:
    return {
        "scope": {
            "include_dirs": ["core"],
            "exclude_dirs": ["tests", "__pycache__", ".git"],
        },
        "thresholds": {
            "max_python_files": 10,
            "max_total_loc": 2000,
            "max_avg_loc_per_file": 500,
            "max_file_loc": 1000,
            "soft_file_loc": 200,
            "max_files_over_soft_loc": 2,
            "max_function_loc": max_function_loc,
            "soft_function_loc": 50,
            "max_functions_over_soft_loc": 3,
            "max_function_args": 5,
            "max_methods_per_class": 10,
            "max_classes_over_method_soft_limit": 2,
            "soft_method_count_per_class": 5,
        },
    }


def _base_report(value: int) -> dict:
    check_row = {
        "metric": "total_loc",
        "value": value,
        "threshold": 100,
        "status": "FAIL",
    }
    return {
        "scope": {},
        "thresholds": {},
        "metrics": {"total_loc": value},
        "checks": [check_row],
        "violations": [check_row],
        "top_files_by_loc": [],
        "top_functions_by_loc": [],
        "top_classes_by_methods": [],
    }


def test_build_report_collects_metrics(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "core" / "sample.py",
        """
class Engine:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def run(self):
        return self.alpha + self.beta


def helper(x, y):
    return x * y
""".strip(),
    )

    report = module._build_report(tmp_path, _budget())

    assert report["metrics"]["python_files"] == 1
    assert report["metrics"]["max_function_args"] == 3
    assert report["metrics"]["max_methods_per_class"] == 2
    assert report["violations"] == []
    assert any(row["name"] == "run" for row in report["top_functions_by_loc"])


def test_main_returns_nonzero_in_strict_mode_on_violation(tmp_path, monkeypatch):
    module = _load_module()
    _write(
        tmp_path / "core" / "too_long.py",
        """
def very_long_function(a, b, c):
    x = a + b + c
    x += 1
    x += 2
    x += 3
    x += 4
    x += 5
    x += 6
    x += 7
    return x
""".strip(),
    )

    config_rel = Path("config") / "complexity_budget.json"
    _write(tmp_path / config_rel, json.dumps(_budget(max_function_loc=5), indent=2))

    report_md_rel = Path("artifacts") / "complexity-governance-report.md"
    report_json_rel = Path("artifacts") / "complexity-governance-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "complexity_guard.py",
            "--root",
            str(tmp_path),
            "--config",
            str(config_rel),
            "--report-md",
            str(report_md_rel),
            "--report-json",
            str(report_json_rel),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    report_md = (tmp_path / report_md_rel).read_text(encoding="utf-8")
    report_json = json.loads((tmp_path / report_json_rel).read_text(encoding="utf-8"))
    assert "FAIL" in report_md
    assert any(v["metric"] == "max_function_loc" for v in report_json["violations"])


def test_strict_regression_only_passes_when_not_worse_than_baseline(tmp_path, monkeypatch):
    module = _load_module()
    config_path = tmp_path / "config.json"
    baseline_path = tmp_path / "baseline.json"
    report_md = tmp_path / "artifacts" / "complexity.md"
    report_json = tmp_path / "artifacts" / "complexity.json"

    _write(config_path, "{}")
    _write(baseline_path, json.dumps({"metrics": {"total_loc": 130}}))
    monkeypatch.setattr(module, "_build_report", lambda root, config: _base_report(120))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "complexity_guard.py",
            "--root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--report-md",
            str(report_md.relative_to(tmp_path)),
            "--report-json",
            str(report_json.relative_to(tmp_path)),
            "--strict",
            "--strict-regression-only",
            "--baseline-json",
            str(baseline_path.relative_to(tmp_path)),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["violations"]
    assert payload["regressions"] == []


def test_strict_regression_only_fails_when_worse_than_baseline(tmp_path, monkeypatch):
    module = _load_module()
    config_path = tmp_path / "config.json"
    baseline_path = tmp_path / "baseline.json"
    report_md = tmp_path / "artifacts" / "complexity.md"
    report_json = tmp_path / "artifacts" / "complexity.json"

    _write(config_path, "{}")
    _write(baseline_path, json.dumps({"metrics": {"total_loc": 110}}))
    monkeypatch.setattr(module, "_build_report", lambda root, config: _base_report(120))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "complexity_guard.py",
            "--root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--report-md",
            str(report_md.relative_to(tmp_path)),
            "--report-json",
            str(report_json.relative_to(tmp_path)),
            "--strict",
            "--strict-regression-only",
            "--baseline-json",
            str(baseline_path.relative_to(tmp_path)),
        ],
    )

    exit_code = module.main()
    assert exit_code == 2
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert len(payload["regressions"]) == 1
    assert payload["regressions"][0]["reason"] == "worsened_existing_violation"
