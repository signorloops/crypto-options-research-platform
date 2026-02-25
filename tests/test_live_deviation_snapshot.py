"""Tests for live deviation snapshot governance script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "live_deviation_snapshot.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "live_deviation_snapshot_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load live_deviation_snapshot module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_main_file_mode_generates_snapshot_reports(tmp_path, monkeypatch):
    module = _load_module()
    cex_path = tmp_path / "cex.csv"
    defi_path = tmp_path / "defi.csv"
    output_md = tmp_path / "artifacts" / "live-deviation-snapshot.md"
    output_json = tmp_path / "artifacts" / "live-deviation-snapshot.json"

    _write(
        cex_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:00Z,BTC-OPT,call,0.05,0.25,1200,okx\n"
        ),
    )
    _write(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:20Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_deviation_snapshot.py",
            "--cex-file",
            str(cex_path),
            "--defi-file",
            str(defi_path),
            "--threshold-bps",
            "200",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 0
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["n_rows"] == 1
    assert payload["sources"]["mode"] == "file"


def test_main_strict_returns_nonzero_when_alerts_exist(tmp_path, monkeypatch):
    module = _load_module()
    cex_path = tmp_path / "cex.csv"
    defi_path = tmp_path / "defi.csv"
    output_md = tmp_path / "artifacts" / "live-deviation-snapshot.md"
    output_json = tmp_path / "artifacts" / "live-deviation-snapshot.json"

    _write(
        cex_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:00Z,BTC-OPT,call,0.05,0.25,1600,okx\n"
        ),
    )
    _write(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:20Z,BTC-OPT,call,0.05,0.25,1000,lyra\n"
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_deviation_snapshot.py",
            "--cex-file",
            str(cex_path),
            "--defi-file",
            str(defi_path),
            "--threshold-bps",
            "50",
            "--strict",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 2


def test_script_runs_via_file_entrypoint(tmp_path):
    cex_path = tmp_path / "cex.csv"
    defi_path = tmp_path / "defi.csv"
    output_md = tmp_path / "artifacts" / "live-deviation-snapshot.md"
    output_json = tmp_path / "artifacts" / "live-deviation-snapshot.json"

    _write(
        cex_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:00Z,BTC-OPT,call,0.05,0.25,1200,okx\n"
        ),
    )
    _write(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:20Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--cex-file",
            str(cex_path),
            "--defi-file",
            str(defi_path),
            "--threshold-bps",
            "200",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_md.exists()
    assert output_json.exists()
