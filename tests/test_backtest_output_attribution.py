"""Tests for attribution fields emitted by backtest output scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WITH_OUTPUT_PATH = ROOT / "scripts" / "backtest" / "run_backtest_with_output.py"
FULL_HISTORY_PATH = ROOT / "scripts" / "backtest" / "run_backtest_full_history.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_backtest_with_output_summary_contains_attribution_fields(tmp_path):
    module = _load_module(WITH_OUTPUT_PATH, "run_backtest_with_output_test_module")
    module.RESULTS_DIR = tmp_path

    state = module.BacktestState(
        position=0.3,
        cash=100_010.0,
        mid_price=50_000.0,
        trade_count=12,
        buy_count=6,
        sell_count=6,
        total_pnl=120.0,
    )

    out_file = module.save_results({"Demo": state}, "20260225_000000")
    payload = json.loads(Path(out_file).read_text(encoding="utf-8"))
    summary = payload["Demo"]["summary"]

    assert "spread_capture" in summary
    assert "adverse_selection_cost" in summary
    assert "inventory_cost" in summary
    assert "hedging_cost" in summary
    assert summary["spread_capture"] >= summary["total_pnl"]


def test_run_backtest_full_history_summary_contains_attribution_fields(tmp_path):
    module = _load_module(FULL_HISTORY_PATH, "run_backtest_full_history_test_module")
    module.RESULTS_DIR = tmp_path

    state = module.BacktestState(
        position=-0.2,
        cash=99_980.0,
        mid_price=45_000.0,
        trade_count=10,
        buy_count=5,
        sell_count=5,
        total_pnl=80.0,
    )

    out_file = module.save_detailed_results({"Demo": state}, "20260225_000000")
    payload = json.loads(Path(out_file).read_text(encoding="utf-8"))
    summary = payload["Demo"]["summary"]

    assert "spread_capture" in summary
    assert "adverse_selection_cost" in summary
    assert "inventory_cost" in summary
    assert "hedging_cost" in summary
    assert summary["spread_capture"] >= summary["final_pnl"]
