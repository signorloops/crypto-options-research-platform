#!/usr/bin/env python3
"""Generate weekly KPI snapshot and risk exception audit from backtest outputs."""

from __future__ import annotations

import argparse
import json
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
    load_json_object as _load_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)
from scripts.governance.weekly_close_gate_utils import (
    build_close_gate_summary,
    collect_open_labels,
)
from scripts.governance.weekly_git_utils import parse_recent_change_entries
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
    build_close_gate_action_items,
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
    build_close_gate_markdown,
    build_close_gate_pr_brief,
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

def _to_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = str(entry).strip()
        if text:
            items.append(text)
    return items

def _load_close_gate_snapshot(signoff_json_path: Path) -> tuple[dict[str, Any], str]:
    if not signoff_json_path.exists():
        return {}, "missing_signoff_json"
    try:
        payload = _load_json(signoff_json_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, "invalid_signoff_json"
    return payload, ""


def _evaluate_close_gate(signoff_json_path: Path) -> tuple[bool, str, dict[str, Any]]:
    payload, load_error = _load_close_gate_snapshot(signoff_json_path)
    if load_error:
        return False, load_error, {}
    status = str(payload.get("status", "")).strip().upper()
    if status == "READY_FOR_CLOSE":
        return True, status, payload
    if status:
        return False, f"status={status}", payload
    return False, "status=UNKNOWN", payload

def _build_close_gate_report(
    *,
    signoff_json_path: Path,
    close_ready: bool,
    close_detail: str,
    signoff_payload: dict[str, Any],
) -> dict[str, Any]:
    manual_missing = collect_open_labels(signoff_payload.get("manual_items"))
    role_signoffs_missing = collect_open_labels(signoff_payload.get("role_signoffs"))
    signoff_status = str(signoff_payload.get("status", "")).strip().upper()
    auto_blockers = _to_text_list(signoff_payload.get("auto_blockers"))
    pending_items = _to_text_list(signoff_payload.get("pending_items"))
    action_items = build_close_gate_action_items(
        signoff_status=signoff_status,
        auto_blockers=auto_blockers,
        manual_missing=manual_missing,
        role_signoffs_missing=role_signoffs_missing,
        close_ready=close_ready,
    )
    pr_brief = build_close_gate_pr_brief(
        close_ready=close_ready,
        close_detail=close_detail,
        signoff_status=signoff_status,
        auto_blockers=auto_blockers,
        pending_items=pending_items,
        action_items=action_items,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if close_ready else "FAIL",
        "gate": "READY_FOR_CLOSE",
        "reason": close_detail,
        "signoff_json": str(signoff_json_path),
        "signoff_status": signoff_status or "UNKNOWN",
        "auto_blockers": auto_blockers,
        "pending_items": pending_items,
        "manual_missing": manual_missing,
        "role_signoffs_missing": role_signoffs_missing,
        "action_items": action_items,
        "pr_brief": pr_brief,
        "summary": build_close_gate_summary(
            auto_blockers=auto_blockers,
            pending_items=pending_items,
            manual_missing=manual_missing,
            role_signoffs_missing=role_signoffs_missing,
        ),
    }


def _close_gate_to_markdown(report: dict[str, Any]) -> str:
    return build_close_gate_markdown(report)


def _write_close_gate_report(
    *,
    signoff_json_path: Path,
    close_gate_md: Path,
    close_gate_json: Path,
    close_ready: bool,
    close_detail: str,
    signoff_payload: dict[str, Any],
) -> None:
    close_report = _build_close_gate_report(
        signoff_json_path=signoff_json_path,
        close_ready=close_ready,
        close_detail=close_detail,
        signoff_payload=signoff_payload,
    )
    _write_markdown(close_gate_md, _close_gate_to_markdown(close_report))
    _write_json(close_gate_json, close_report)


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _load_thresholds(path: Path) -> dict[str, float]:
    return load_threshold_map(path, DEFAULT_THRESHOLDS, label="threshold")


def _load_consistency_thresholds(path: Path) -> dict[str, float]:
    return load_threshold_map(path, DEFAULT_CONSISTENCY_THRESHOLDS, label="consistency threshold")


def _is_shallow_repository(repo_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    return completed.stdout.strip().lower() == "true"


def _collect_recent_changes(repo_root: Path, since_days: int) -> dict[str, Any]:
    shallow = _is_shallow_repository(repo_root)
    completed = subprocess.run(
        [
            "git",
            "log",
            f"--since={since_days} days ago",
            "--pretty=format:%H%x09%ad%x09%s",
            "--date=short",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "executed": False,
            "since_days": since_days,
            "entries": [],
            "count": 0,
            "shallow": shallow,
            "error": completed.stderr.strip(),
        }

    entries = parse_recent_change_entries(completed.stdout)
    return {
        "executed": True,
        "since_days": since_days,
        "entries": entries,
        "count": len(entries),
        "shallow": shallow,
        "error": "",
    }


def _detect_latest_tag(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return {"executed": True, "tag": completed.stdout.strip(), "error": "", "source": "tag"}

    head_ref = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head_ref.returncode != 0:
        return {"executed": False, "tag": "", "error": completed.stderr.strip(), "source": ""}
    return {
        "executed": True,
        "tag": f"HEAD-{head_ref.stdout.strip()}",
        "error": "",
        "source": "commit",
    }


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
        return 2 if args.strict_close else 0

    thresholds_path = (repo_root / args.thresholds).resolve()
    thresholds = _load_thresholds(thresholds_path)
    consistency_thresholds_path = (repo_root / args.consistency_thresholds).resolve()
    consistency_thresholds = _load_consistency_thresholds(consistency_thresholds_path)
    input_files = _resolve_input_files(
        repo_root=repo_root,
        explicit_inputs=args.inputs or [],
        results_dir=args.results_dir,
        pattern=args.pattern,
    )

    if not input_files:
        print("Weekly operating audit: no input files found.")
        return 2 if (args.strict or args.strict_close) else 0

    regression_result = _run_regression_command(args.regression_cmd, repo_root)

    change_log = _collect_recent_changes(repo_root, max(args.change_log_days, 1))
    rollback_marker = _detect_latest_tag(repo_root)
    baselines = _load_baseline_reports(
        repo_root=repo_root,
        performance_json=args.performance_json,
        latency_json=args.latency_json,
    )

    report = _build_report(
        input_files,
        thresholds,
        consistency_thresholds=consistency_thresholds,
        regression_result=regression_result,
        change_log=change_log,
        rollback_marker=rollback_marker,
        performance_result=baselines["performance"],
        performance_required=args.require_performance,
        latency_result=baselines["latency"],
        latency_required=args.require_latency,
    )

    _write_markdown(paths["output_md"], _to_markdown(report))
    _write_json(paths["output_json"], report)

    issue_messages = _collect_issue_messages(
        report,
        regression_result=regression_result,
        require_performance=args.require_performance,
        require_latency=args.require_latency,
    )
    for message in issue_messages:
        print(message)
        if args.strict:
            return 2
    if args.strict_close:
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
    if not issue_messages:
        print("Weekly operating audit: no threshold exceptions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
