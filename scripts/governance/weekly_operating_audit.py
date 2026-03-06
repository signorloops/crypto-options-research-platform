#!/usr/bin/env python3
"""Generate weekly KPI snapshot and risk exception audit from backtest outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.report_utils import (
    JSON_REPORT_EXCEPTIONS,
    discover_input_files as _discover_input_files,
    write_json as _write_json,
    write_markdown as _write_markdown,
)
from scripts.governance.weekly_close_gate_report_utils import (
    build_close_gate_report as _build_close_gate_report,
    evaluate_close_gate as _evaluate_close_gate,
    write_close_gate_report as _write_close_gate_report,
)
from scripts.governance.weekly_git_utils import (
    collect_recent_changes as _collect_recent_changes_impl,
    detect_latest_tag as _detect_latest_tag_impl,
)
from scripts.governance.weekly_operating_parser_utils import (
    build_weekly_operating_argument_specs,
)
from scripts.governance.weekly_operating_cli_utils import (
    collect_issue_messages,
    resolve_input_files,
    run_regression_command,
)
from scripts.governance.status_action_utils import (
    MANUAL_CHECKLIST_ITEMS,
    build_incomplete_tasks,
)
from scripts.governance.weekly_operating_data_utils import (
    collect_strategy_snapshots,
)
from scripts.governance.weekly_operating_report_utils import (
    build_consistency_checks,
    build_operating_checklist,
    build_report_summary,
    build_risk_exceptions,
    normalize_change_log,
    normalize_optional_baseline_report,
    normalize_regression_report,
    normalize_rollback_marker,
    resolve_optional_report_check,
)
from scripts.governance.weekly_operating_runtime_utils import (
    load_optional_report,
    load_threshold_map,
)
from scripts.governance.weekly_operating_render_utils import (
    build_weekly_operating_markdown,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_sharpe": 0.5,
    "max_abs_drawdown": 0.25,
    "max_var_breach_rate": 0.05,
    "max_fill_calibration_error": 0.20,
}

DEFAULT_CONSISTENCY_THRESHOLDS: dict[str, float] = {
    "max_abs_pnl_diff": 20000.0,
    "max_abs_sharpe_diff": 1500.0,
    "max_abs_max_drawdown_diff": 0.20,
}


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _load_thresholds(path: Path) -> dict[str, float]:
    return load_threshold_map(path, DEFAULT_THRESHOLDS, label="threshold")


def _load_consistency_thresholds(path: Path) -> dict[str, float]:
    return load_threshold_map(path, DEFAULT_CONSISTENCY_THRESHOLDS, label="consistency threshold")


def _collect_recent_changes(repo_root: Path, since_days: int) -> dict[str, Any]:
    return _collect_recent_changes_impl(
        repo_root,
        since_days,
        runner=subprocess.run,
    )


def _detect_latest_tag(repo_root: Path) -> dict[str, Any]:
    return _detect_latest_tag_impl(
        repo_root,
        runner=subprocess.run,
    )


def _evaluate_rows(
    rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        breaches: list[str] = []
        sharpe = row.get("sharpe")
        max_dd = row.get("max_drawdown_abs")
        var_rate = row.get("var_breach_rate")
        fill_error = row.get("fill_calibration_error")

        if sharpe is not None and sharpe < thresholds["min_sharpe"]:
            breaches.append(f"sharpe<{thresholds['min_sharpe']}")
        if max_dd is not None and max_dd > thresholds["max_abs_drawdown"]:
            breaches.append(f"max_drawdown_abs>{thresholds['max_abs_drawdown']}")
        if var_rate is not None and var_rate > thresholds["max_var_breach_rate"]:
            breaches.append(f"var_breach_rate>{thresholds['max_var_breach_rate']}")
        if fill_error is not None and fill_error > thresholds["max_fill_calibration_error"]:
            breaches.append(f"fill_calibration_error>{thresholds['max_fill_calibration_error']}")

        evaluated.append(
            {
                **row,
                "status": "FAIL" if breaches else "PASS",
                "breached_rules": "; ".join(breaches),
            }
        )
    return evaluated


def _build_report(
    input_files: Sequence[Path],
    thresholds: dict[str, float],
    consistency_thresholds: dict[str, float] | None = None,
    regression_result: dict[str, Any] | None = None,
    change_log: dict[str, Any] | None = None,
    rollback_marker: dict[str, Any] | None = None,
    performance_result: dict[str, Any] | None = None,
    performance_required: bool = False,
    latency_result: dict[str, Any] | None = None,
    latency_required: bool = False,
) -> dict[str, Any]:
    consistency_thresholds_final = (
        dict(consistency_thresholds)
        if consistency_thresholds is not None
        else dict(DEFAULT_CONSISTENCY_THRESHOLDS)
    )
    files_sorted = sorted(input_files, key=lambda p: p.stat().st_mtime, reverse=True)
    latest_by_strategy, previous_by_strategy, parse_errors = collect_strategy_snapshots(files_sorted)

    snapshot_rows = _evaluate_rows(
        sorted(latest_by_strategy.values(), key=lambda r: r["strategy"]), thresholds
    )
    risk_exceptions = build_risk_exceptions(snapshot_rows)

    consistency_checks = build_consistency_checks(
        latest_by_strategy=latest_by_strategy,
        previous_by_strategy=previous_by_strategy,
        thresholds=consistency_thresholds_final,
    )
    consistency_exceptions = [row for row in consistency_checks if row["status"] == "FAIL"]

    performance_report = normalize_optional_baseline_report(performance_result)
    performance_check = resolve_optional_report_check(
        performance_report,
        required=performance_required,
    )
    latency_report = normalize_optional_baseline_report(latency_result)
    latency_check = resolve_optional_report_check(
        latency_report,
        required=latency_required,
    )

    checklist = build_operating_checklist(
        snapshot_rows=snapshot_rows,
        parse_errors=parse_errors,
        consistency_checks=consistency_checks,
        consistency_exceptions=consistency_exceptions,
        risk_exceptions=risk_exceptions,
        change_log=change_log,
        rollback_marker=rollback_marker,
        regression_result=regression_result,
        performance_check=performance_check,
        latency_check=latency_check,
    )

    incomplete_tasks = build_incomplete_tasks(
        checklist,
        performance_required=performance_required,
        latency_required=latency_required,
        manual_items=MANUAL_CHECKLIST_ITEMS,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in files_sorted],
        "thresholds": thresholds,
        "consistency_thresholds": consistency_thresholds_final,
        "summary": build_report_summary(
            snapshot_rows=snapshot_rows,
            risk_exceptions=risk_exceptions,
            consistency_checks=consistency_checks,
            consistency_exceptions=consistency_exceptions,
            parse_errors=parse_errors,
        ),
        "kpi_snapshot": snapshot_rows,
        "risk_exceptions": risk_exceptions,
        "consistency_checks": consistency_checks,
        "consistency_exceptions": consistency_exceptions,
        "checklist": checklist,
        "manual_checklist_items": MANUAL_CHECKLIST_ITEMS,
        "incomplete_tasks": incomplete_tasks,
        "parse_errors": parse_errors,
        "regression": normalize_regression_report(regression_result),
        "change_log": normalize_change_log(change_log),
        "rollback_marker": normalize_rollback_marker(rollback_marker),
        "performance_baseline": performance_report,
        "latency_baseline": latency_report,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    return build_weekly_operating_markdown(report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate weekly operating KPI/risk audit.")
    for spec in build_weekly_operating_argument_specs():
        flags = spec["flags"]
        kwargs = {key: value for key, value in spec.items() if key not in {"flags", "dest"}}
        parser.add_argument(*flags, dest=spec["dest"], **kwargs)
    return parser


def _resolve_input_files(
    *,
    repo_root: Path,
    explicit_inputs: Sequence[str],
    results_dir: str,
    pattern: str,
) -> list[Path]:
    return resolve_input_files(
        repo_root=repo_root,
        explicit_inputs=explicit_inputs,
        results_dir=results_dir,
        pattern=pattern,
        discover_input_files=_discover_input_files,
    )


def _resolve_audit_paths(
    *,
    repo_root: Path,
    output_md: str,
    output_json: str,
    signoff_json: str,
    close_gate_output_md: str,
    close_gate_output_json: str,
) -> dict[str, Path]:
    return {
        "output_md": (repo_root / output_md).resolve(),
        "output_json": (repo_root / output_json).resolve(),
        "signoff_json": (repo_root / signoff_json).resolve(),
        "close_gate_output_md": (repo_root / close_gate_output_md).resolve(),
        "close_gate_output_json": (repo_root / close_gate_output_json).resolve(),
    }


def _run_regression_command(command: str, repo_root: Path) -> dict[str, Any] | None:
    return run_regression_command(command, repo_root=repo_root, runner=subprocess.run)


def _collect_issue_messages(
    report: dict[str, Any],
    *,
    regression_result: dict[str, Any] | None,
    require_performance: bool,
    require_latency: bool,
) -> list[str]:
    return collect_issue_messages(
        report,
        regression_result=regression_result,
        require_performance=require_performance,
        require_latency=require_latency,
    )


def _report_issue_messages(issue_messages: list[str], *, strict: bool) -> int | None:
    for message in issue_messages:
        print(message)
        if strict:
            return 2
    return None


def _handle_strict_close(
    *,
    strict_close: bool,
    signoff_json_path: Path,
    close_gate_md: Path,
    close_gate_json: Path,
) -> int | None:
    if not strict_close:
        return None
    close_ready, close_detail, signoff_payload = _evaluate_close_gate(signoff_json_path)
    _write_close_gate_report(
        signoff_json_path=signoff_json_path,
        close_gate_md=close_gate_md,
        close_gate_json=close_gate_json,
        close_ready=close_ready,
        close_detail=close_detail,
        signoff_payload=signoff_payload,
    )
    if not close_ready:
        print(f"Weekly operating audit: close gate not ready ({close_detail}).")
        return 2
    return None


def _run_close_gate_only(
    *,
    strict_close: bool,
    signoff_json_path: Path,
    close_gate_md: Path,
    close_gate_json: Path,
) -> int:
    close_ready, close_detail, signoff_payload = _evaluate_close_gate(signoff_json_path)
    _write_close_gate_report(
        signoff_json_path=signoff_json_path,
        close_gate_md=close_gate_md,
        close_gate_json=close_gate_json,
        close_ready=close_ready,
        close_detail=close_detail,
        signoff_payload=signoff_payload,
    )
    if close_ready:
        print("Weekly close gate: READY_FOR_CLOSE.")
        return 0
    print(f"Weekly close gate: not ready ({close_detail}).")
    return 2 if strict_close else 0


def _load_baseline_reports(
    *,
    repo_root: Path,
    performance_json: str,
    latency_json: str,
) -> dict[str, dict[str, Any]]:
    return {
        "performance": load_optional_report(
            (repo_root / performance_json).resolve(),
            missing_error="missing_performance_json",
        ),
        "latency": load_optional_report(
            (repo_root / latency_json).resolve(),
            missing_error="missing_latency_json",
        ),
    }


def _prepare_audit_run(*, repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    thresholds = _load_thresholds((repo_root / args.thresholds).resolve())
    consistency_thresholds = _load_consistency_thresholds(
        (repo_root / args.consistency_thresholds).resolve()
    )
    input_files = _resolve_input_files(
        repo_root=repo_root,
        explicit_inputs=args.inputs or [],
        results_dir=args.results_dir,
        pattern=args.pattern,
    )
    return {
        "thresholds": thresholds,
        "consistency_thresholds": consistency_thresholds,
        "input_files": input_files,
        "regression_result": _run_regression_command(args.regression_cmd, repo_root),
        "change_log": _collect_recent_changes(repo_root, max(args.change_log_days, 1)),
        "rollback_marker": _detect_latest_tag(repo_root),
        "baselines": _load_baseline_reports(
            repo_root=repo_root,
            performance_json=args.performance_json,
            latency_json=args.latency_json,
        ),
    }


def main() -> int:
    args = _build_parser().parse_args()

    repo_root = Path(".").resolve()
    paths = _resolve_audit_paths(
        repo_root=repo_root,
        output_md=args.output_md,
        output_json=args.output_json,
        signoff_json=args.signoff_json,
        close_gate_output_md=args.close_gate_output_md,
        close_gate_output_json=args.close_gate_output_json,
    )
    signoff_json_path = paths["signoff_json"]
    close_gate_md = paths["close_gate_output_md"]
    close_gate_json = paths["close_gate_output_json"]

    if args.close_gate_only:
        return _run_close_gate_only(
            strict_close=args.strict_close,
            signoff_json_path=signoff_json_path,
            close_gate_md=close_gate_md,
            close_gate_json=close_gate_json,
        )

    audit_run = _prepare_audit_run(repo_root=repo_root, args=args)
    input_files = audit_run["input_files"]

    if not input_files:
        print("Weekly operating audit: no input files found.")
        return 2 if (args.strict or args.strict_close) else 0

    report = _build_report(
        input_files,
        audit_run["thresholds"],
        consistency_thresholds=audit_run["consistency_thresholds"],
        regression_result=audit_run["regression_result"],
        change_log=audit_run["change_log"],
        rollback_marker=audit_run["rollback_marker"],
        performance_result=audit_run["baselines"]["performance"],
        performance_required=args.require_performance,
        latency_result=audit_run["baselines"]["latency"],
        latency_required=args.require_latency,
    )

    _write_markdown(paths["output_md"], _to_markdown(report))
    _write_json(paths["output_json"], report)

    issue_messages = _collect_issue_messages(
        report,
        regression_result=audit_run["regression_result"],
        require_performance=args.require_performance,
        require_latency=args.require_latency,
    )
    if (exit_code := _report_issue_messages(issue_messages, strict=args.strict)) is not None:
        return exit_code
    if (
        exit_code := _handle_strict_close(
            strict_close=args.strict_close,
            signoff_json_path=signoff_json_path,
            close_gate_md=close_gate_md,
            close_gate_json=close_gate_json,
        )
    ) is not None:
        return exit_code
    if not issue_messages:
        print("Weekly operating audit: no threshold exceptions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
