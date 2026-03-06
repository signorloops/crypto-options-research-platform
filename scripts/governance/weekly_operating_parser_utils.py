"""Shared parser-argument definitions for weekly operating audit CLI."""

from __future__ import annotations

from typing import Any


def build_weekly_operating_argument_specs() -> list[dict[str, Any]]:
    """Return ordered argument definitions for the weekly operating audit parser."""
    return [
        {"flags": ["--results-dir"], "dest": "results_dir", "default": "results", "help": "Directory for backtest outputs."},
        {"flags": ["--pattern"], "dest": "pattern", "default": "backtest*.json", "help": "Glob pattern in results dir."},
        {"flags": ["--inputs"], "dest": "inputs", "nargs": "*", "help": "Optional explicit input JSON files. If set, results-dir/pattern is ignored."},
        {"flags": ["--thresholds"], "dest": "thresholds", "default": "config/weekly_operating_thresholds.json", "help": "Path to thresholds JSON. Uses defaults when file is missing."},
        {"flags": ["--consistency-thresholds"], "dest": "consistency_thresholds", "default": "config/consistency_thresholds.json", "help": "Path to consistency thresholds JSON. Uses defaults when file is missing."},
        {"flags": ["--output-md"], "dest": "output_md", "default": "artifacts/weekly-operating-audit.md", "help": "Output markdown report path."},
        {"flags": ["--output-json"], "dest": "output_json", "default": "artifacts/weekly-operating-audit.json", "help": "Output JSON report path."},
        {"flags": ["--strict"], "dest": "strict", "action": "store_true", "help": "Exit non-zero when risk exceptions exist or no strategy rows can be extracted."},
        {"flags": ["--strict-close"], "dest": "strict_close", "action": "store_true", "help": "Exit non-zero unless weekly sign-off status is READY_FOR_CLOSE."},
        {"flags": ["--signoff-json"], "dest": "signoff_json", "default": "artifacts/weekly-signoff-pack.json", "help": "Path to weekly sign-off JSON used by --strict-close."},
        {"flags": ["--close-gate-only"], "dest": "close_gate_only", "action": "store_true", "help": "Only validate close gate status from --signoff-json."},
        {"flags": ["--close-gate-output-md"], "dest": "close_gate_output_md", "default": "artifacts/weekly-close-gate.md", "help": "Output markdown path for close gate summary."},
        {"flags": ["--close-gate-output-json"], "dest": "close_gate_output_json", "default": "artifacts/weekly-close-gate.json", "help": "Output JSON path for close gate summary."},
        {"flags": ["--regression-cmd"], "dest": "regression_cmd", "default": "", "help": "Optional regression command to execute and include in the audit report."},
        {"flags": ["--performance-json"], "dest": "performance_json", "default": "artifacts/algorithm-performance-baseline.json", "help": "Path to algorithm performance baseline JSON report."},
        {"flags": ["--require-performance"], "dest": "require_performance", "action": "store_true", "help": "Mark audit as incomplete when performance baseline report is missing or failing."},
        {"flags": ["--latency-json"], "dest": "latency_json", "default": "artifacts/performance/latency_benchmark_report.json", "help": "Path to latency benchmark JSON report."},
        {"flags": ["--require-latency"], "dest": "require_latency", "action": "store_true", "help": "Mark audit as incomplete when latency benchmark report is missing or failing."},
        {"flags": ["--change-log-days"], "dest": "change_log_days", "type": int, "default": 7, "help": "Look-back window (days) for auto-generated change log."},
    ]
