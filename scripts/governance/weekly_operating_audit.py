#!/usr/bin/env python3
"""Generate weekly KPI snapshot and risk exception audit from backtest outputs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

MANUAL_CHECKLIST_ITEMS: list[str] = [
    "灰度发布完成",
    "24h 观察完成",
    "是否触发回滚已决策",
    "收益归因表",
    "变更与回滚记录",
    "ADR",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


def _to_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = str(entry).strip()
        if text:
            items.append(text)
    return items


def _summarize_items(items: list[str], limit: int = 4) -> str:
    if not items:
        return "None"
    visible = items[:limit]
    summary = " / ".join(visible)
    remaining = len(items) - len(visible)
    if remaining > 0:
        return f"{summary} (+{remaining} more)"
    return summary


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
    manual_items = signoff_payload.get("manual_items")
    role_signoffs = signoff_payload.get("role_signoffs")
    manual_missing = [
        str(item.get("label", "")).strip()
        for item in (manual_items if isinstance(manual_items, list) else [])
        if isinstance(item, dict) and not bool(item.get("done"))
    ]
    role_signoffs_missing = [
        str(item.get("label", "")).strip()
        for item in (role_signoffs if isinstance(role_signoffs, list) else [])
        if isinstance(item, dict) and not bool(item.get("done"))
    ]
    manual_missing = [x for x in manual_missing if x]
    role_signoffs_missing = [x for x in role_signoffs_missing if x]
    signoff_status = str(signoff_payload.get("status", "")).strip().upper()
    auto_blockers = _to_text_list(signoff_payload.get("auto_blockers"))
    pending_items = _to_text_list(signoff_payload.get("pending_items"))
    action_items: list[str] = []
    if signoff_status == "AUTO_BLOCKED":
        action_items.append("Resolve auto blockers to clear AUTO_BLOCKED status.")
    if manual_missing:
        action_items.append("Complete remaining manual checks.")
    if role_signoffs_missing:
        action_items.append("Collect all role sign-offs (Research/Engineering/Risk).")
    if not action_items and not close_ready:
        action_items.append("Review sign-off payload and set status to READY_FOR_CLOSE.")
    pr_brief_lines = [
        "### Weekly Close Gate",
        f"- Status: {'PASS' if close_ready else 'FAIL'} (`{close_detail}`)",
        f"- Signoff status: `{signoff_status or 'UNKNOWN'}`",
        f"- Auto blockers: {_summarize_items(auto_blockers)}",
        f"- Pending items: {_summarize_items(pending_items)}",
        f"- Next actions: {_summarize_items(action_items, limit=3)}",
    ]
    pr_brief = "\n".join(pr_brief_lines)
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
        "summary": {
            "auto_blockers": len(auto_blockers),
            "pending_items": len(pending_items),
            "manual_missing": len(manual_missing),
            "role_signoffs_missing": len(role_signoffs_missing),
        },
    }


def _close_gate_to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly Close Gate")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Gate: `{report['gate']}`")
    lines.append(f"- Reason: `{report['reason']}`")
    lines.append(f"- Signoff Status: `{report['signoff_status']}`")
    lines.append(f"- Signoff JSON: `{report['signoff_json']}`")
    lines.append("")
    lines.append("## Auto Blockers")
    lines.append("")
    if report["auto_blockers"]:
        for item in report["auto_blockers"]:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## Pending Items")
    lines.append("")
    if report["pending_items"]:
        for item in report["pending_items"]:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## Missing Manual Checks")
    lines.append("")
    if report["manual_missing"]:
        for item in report["manual_missing"]:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## Missing Role Sign-offs")
    lines.append("")
    if report["role_signoffs_missing"]:
        for item in report["role_signoffs_missing"]:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## PR Brief (Copy/Paste)")
    lines.append("")
    lines.append("```markdown")
    lines.append(report["pr_brief"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _pick_first_numeric(maps: Sequence[dict[str, Any]], keys: Sequence[str]) -> float | None:
    for mapping in maps:
        for key in keys:
            if key in mapping:
                value = _to_float(mapping.get(key))
                if value is not None:
                    return value
    return None


def _pick_first_text(maps: Sequence[dict[str, Any]], keys: Sequence[str]) -> str | None:
    for mapping in maps:
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def _infer_experiment_id(source: Path) -> str:
    stem = source.stem.strip().replace(" ", "_")
    return f"AUTO-{stem}"


def _extract_strategy_rows(raw: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        summary = payload.get("summary")
        metrics = payload.get("metrics")
        risk = payload.get("risk")
        summary_map = summary if isinstance(summary, dict) else {}
        metrics_map = metrics if isinstance(metrics, dict) else {}
        risk_map = risk if isinstance(risk, dict) else {}
        candidates = [summary_map, metrics_map, risk_map, payload]

        pnl = _pick_first_numeric(candidates, ["total_pnl", "final_pnl", "pnl"])
        sharpe = _pick_first_numeric(candidates, ["sharpe_ratio", "sharpe", "deflated_sharpe"])
        max_dd_raw = _pick_first_numeric(candidates, ["max_drawdown", "max_dd"])
        max_dd_abs = abs(max_dd_raw) if max_dd_raw is not None else None
        var_breach_rate = _pick_first_numeric(
            candidates,
            [
                "var_breach_rate",
                "var_exception_rate",
                "var_breach_ratio",
                "var_breach",
            ],
        )
        fill_error = _pick_first_numeric(
            candidates,
            [
                "fill_calibration_error",
                "fill_error",
                "calibration_error",
            ],
        )
        experiment_id = _pick_first_text(
            candidates, ["experiment_id", "experiment", "exp_id"]
        ) or _infer_experiment_id(source)

        rows.append(
            {
                "strategy": str(strategy),
                "source_file": source.name,
                "pnl": pnl,
                "sharpe": sharpe,
                "max_drawdown_abs": max_dd_abs,
                "var_breach_rate": var_breach_rate,
                "fill_calibration_error": fill_error,
                "experiment_id": experiment_id,
            }
        )
    return rows


def _discover_input_files(results_dir: Path, pattern: str) -> list[Path]:
    candidates = sorted(results_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p for p in candidates if p.is_file()]


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _format_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def _load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS)

    raw = _load_json(path)
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in DEFAULT_THRESHOLDS:
        if key in raw:
            value = _to_float(raw[key])
            if value is None:
                raise ValueError(f"Invalid threshold value for '{key}'")
            thresholds[key] = float(value)
    return thresholds


def _load_consistency_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_CONSISTENCY_THRESHOLDS)

    raw = _load_json(path)
    thresholds = dict(DEFAULT_CONSISTENCY_THRESHOLDS)
    for key in DEFAULT_CONSISTENCY_THRESHOLDS:
        if key in raw:
            value = _to_float(raw[key])
            if value is None:
                raise ValueError(f"Invalid consistency threshold value for '{key}'")
            thresholds[key] = float(value)
    return thresholds


def _collect_recent_changes(repo_root: Path, since_days: int) -> dict[str, Any]:
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
            "error": completed.stderr.strip(),
        }

    entries = []
    for raw in completed.stdout.splitlines():
        parts = raw.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_hash, commit_date, subject = parts
        entries.append({"date": commit_date, "commit": commit_hash[:8], "subject": subject})

    return {
        "executed": True,
        "since_days": since_days,
        "entries": entries,
        "count": len(entries),
        "error": "",
    }


def _detect_latest_tag(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
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
) -> dict[str, Any]:
    latest_by_strategy: dict[str, dict[str, Any]] = {}
    previous_by_strategy: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []
    consistency_thresholds_final = (
        dict(consistency_thresholds)
        if consistency_thresholds is not None
        else dict(DEFAULT_CONSISTENCY_THRESHOLDS)
    )
    files_sorted = sorted(input_files, key=lambda p: p.stat().st_mtime, reverse=True)

    for path in files_sorted:
        try:
            raw = _load_json(path)
            rows = _extract_strategy_rows(raw, path)
        except Exception as exc:  # pragma: no cover - defensive parser boundary
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue

        for row in rows:
            strategy = row["strategy"]
            if strategy not in latest_by_strategy:
                latest_by_strategy[strategy] = row
            elif strategy not in previous_by_strategy:
                previous_by_strategy[strategy] = row

    snapshot_rows = _evaluate_rows(
        sorted(latest_by_strategy.values(), key=lambda r: r["strategy"]), thresholds
    )
    risk_exceptions = [
        {
            "strategy": row["strategy"],
            "source_file": row["source_file"],
            "breached_rules": row["breached_rules"],
        }
        for row in snapshot_rows
        if row["status"] == "FAIL"
    ]

    consistency_checks: list[dict[str, Any]] = []
    for strategy in sorted(set(latest_by_strategy).intersection(previous_by_strategy)):
        latest = latest_by_strategy[strategy]
        previous = previous_by_strategy[strategy]

        latest_pnl = latest.get("pnl")
        prev_pnl = previous.get("pnl")
        latest_sharpe = latest.get("sharpe")
        prev_sharpe = previous.get("sharpe")
        latest_dd = latest.get("max_drawdown_abs")
        prev_dd = previous.get("max_drawdown_abs")

        abs_pnl_diff = (
            abs(float(latest_pnl) - float(prev_pnl))
            if latest_pnl is not None and prev_pnl is not None
            else None
        )
        abs_sharpe_diff = (
            abs(float(latest_sharpe) - float(prev_sharpe))
            if latest_sharpe is not None and prev_sharpe is not None
            else None
        )
        abs_max_drawdown_diff = (
            abs(float(latest_dd) - float(prev_dd))
            if latest_dd is not None and prev_dd is not None
            else None
        )

        breaches: list[str] = []
        if (
            abs_pnl_diff is not None
            and abs_pnl_diff > consistency_thresholds_final["max_abs_pnl_diff"]
        ):
            breaches.append(f"abs_pnl_diff>{consistency_thresholds_final['max_abs_pnl_diff']}")
        if (
            abs_sharpe_diff is not None
            and abs_sharpe_diff > consistency_thresholds_final["max_abs_sharpe_diff"]
        ):
            breaches.append(
                f"abs_sharpe_diff>{consistency_thresholds_final['max_abs_sharpe_diff']}"
            )
        if (
            abs_max_drawdown_diff is not None
            and abs_max_drawdown_diff > consistency_thresholds_final["max_abs_max_drawdown_diff"]
        ):
            breaches.append(
                f"abs_max_drawdown_diff>{consistency_thresholds_final['max_abs_max_drawdown_diff']}"
            )

        consistency_checks.append(
            {
                "strategy": strategy,
                "latest_source_file": latest.get("source_file", ""),
                "previous_source_file": previous.get("source_file", ""),
                "abs_pnl_diff": abs_pnl_diff,
                "abs_sharpe_diff": abs_sharpe_diff,
                "abs_max_drawdown_diff": abs_max_drawdown_diff,
                "status": "FAIL" if breaches else "PASS",
                "breached_rules": "; ".join(breaches),
            }
        )

    consistency_exceptions = [row for row in consistency_checks if row["status"] == "FAIL"]
    consistency_baseline_available = len(consistency_checks) > 0

    checklist = {
        "kpi_snapshot_updated": bool(snapshot_rows),
        "experiment_ids_assigned": bool(snapshot_rows)
        and all(bool(row.get("experiment_id")) for row in snapshot_rows),
        "risk_thresholds_confirmed": True,
        "change_log_complete": bool(change_log and change_log.get("count", 0) > 0),
        "rollback_version_marked": bool(rollback_marker and rollback_marker.get("tag")),
        "minimum_regression_passed": (
            bool(regression_result["passed"])
            if regression_result is not None and regression_result.get("executed")
            else None
        ),
        "consistency_check_completed": (
            len(parse_errors) == 0
            and consistency_baseline_available
            and len(consistency_exceptions) == 0
        ),
        "risk_exception_report_output": True,
        "anomalies_attributed": len(risk_exceptions) == 0,
    }

    incomplete_tasks = []
    if not checklist["kpi_snapshot_updated"]:
        incomplete_tasks.append("KPI 快照更新")
    if not checklist["experiment_ids_assigned"]:
        incomplete_tasks.append("实验编号分配完成")
    if not checklist["change_log_complete"]:
        incomplete_tasks.append("变更记录完整")
    if not checklist["rollback_version_marked"]:
        incomplete_tasks.append("回滚版本已标记")
    if checklist["minimum_regression_passed"] is not True:
        incomplete_tasks.append("最小回归通过")
    if not checklist["consistency_check_completed"]:
        incomplete_tasks.append("一致性检查完成")
    if not checklist["anomalies_attributed"]:
        incomplete_tasks.append("异常项已归因")
    incomplete_tasks.extend(MANUAL_CHECKLIST_ITEMS)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in files_sorted],
        "thresholds": thresholds,
        "consistency_thresholds": consistency_thresholds_final,
        "summary": {
            "strategies": len(snapshot_rows),
            "exceptions": len(risk_exceptions),
            "consistency_pairs": len(consistency_checks),
            "consistency_exceptions": len(consistency_exceptions),
            "parse_errors": len(parse_errors),
        },
        "kpi_snapshot": snapshot_rows,
        "risk_exceptions": risk_exceptions,
        "consistency_checks": consistency_checks,
        "consistency_exceptions": consistency_exceptions,
        "checklist": checklist,
        "manual_checklist_items": MANUAL_CHECKLIST_ITEMS,
        "incomplete_tasks": incomplete_tasks,
        "parse_errors": parse_errors,
        "regression": (
            regression_result
            if regression_result is not None
            else {
                "executed": False,
                "command": "",
                "passed": None,
                "return_code": None,
                "output_tail": "",
            }
        ),
        "change_log": (
            change_log
            if change_log is not None
            else {"executed": False, "since_days": 0, "entries": [], "count": 0, "error": ""}
        ),
        "rollback_marker": (
            rollback_marker
            if rollback_marker is not None
            else {"executed": False, "tag": "", "error": "", "source": ""}
        ),
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly Operating Audit")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Input files: `{len(report['inputs'])}`")
    lines.append(f"- Strategies in snapshot: `{report['summary']['strategies']}`")
    lines.append(f"- Risk exceptions: `{report['summary']['exceptions']}`")
    lines.append(f"- Consistency pairs: `{report['summary']['consistency_pairs']}`")
    lines.append(f"- Consistency exceptions: `{report['summary']['consistency_exceptions']}`")
    lines.append("")
    lines.append("## Input Files")
    lines.append("")
    if report["inputs"]:
        for path in report["inputs"]:
            lines.append(f"- `{path}`")
    else:
        lines.append("_none_")
    lines.append("")
    lines.append("## KPI Snapshot")
    lines.append("")
    snapshot_rows = []
    for row in report["kpi_snapshot"]:
        snapshot_rows.append(
            {
                "strategy": row["strategy"],
                "source_file": row["source_file"],
                "experiment_id": row.get("experiment_id") or "n/a",
                "pnl": _fmt(row.get("pnl")),
                "sharpe": _fmt(row.get("sharpe"), digits=4),
                "max_drawdown_abs": _fmt(row.get("max_drawdown_abs"), digits=4),
                "var_breach_rate": _fmt(row.get("var_breach_rate"), digits=4),
                "fill_calibration_error": _fmt(row.get("fill_calibration_error"), digits=4),
                "status": row["status"],
            }
        )
    lines.append(
        _format_table(
            snapshot_rows,
            [
                "strategy",
                "source_file",
                "experiment_id",
                "pnl",
                "sharpe",
                "max_drawdown_abs",
                "var_breach_rate",
                "fill_calibration_error",
                "status",
            ],
        )
    )
    lines.append("")
    lines.append("## Risk Exceptions")
    lines.append("")
    lines.append(
        _format_table(
            report["risk_exceptions"],
            ["strategy", "source_file", "breached_rules"],
        )
    )
    lines.append("")
    lines.append("## Consistency Checks")
    lines.append("")
    consistency_rows = []
    for row in report["consistency_checks"]:
        consistency_rows.append(
            {
                "strategy": row["strategy"],
                "latest_source_file": row["latest_source_file"],
                "previous_source_file": row["previous_source_file"],
                "abs_pnl_diff": _fmt(row.get("abs_pnl_diff")),
                "abs_sharpe_diff": _fmt(row.get("abs_sharpe_diff"), digits=4),
                "abs_max_drawdown_diff": _fmt(row.get("abs_max_drawdown_diff"), digits=4),
                "status": row["status"],
                "breached_rules": row["breached_rules"],
            }
        )
    lines.append(
        _format_table(
            consistency_rows,
            [
                "strategy",
                "latest_source_file",
                "previous_source_file",
                "abs_pnl_diff",
                "abs_sharpe_diff",
                "abs_max_drawdown_diff",
                "status",
                "breached_rules",
            ],
        )
    )
    lines.append("")
    lines.append("## Checklist (Auto)")
    lines.append("")
    auto_items = [
        ("KPI 快照更新", report["checklist"]["kpi_snapshot_updated"]),
        ("实验编号分配完成", report["checklist"]["experiment_ids_assigned"]),
        ("风险门槛已确认", report["checklist"]["risk_thresholds_confirmed"]),
        ("变更记录完整", report["checklist"]["change_log_complete"]),
        ("回滚版本已标记", report["checklist"]["rollback_version_marked"]),
        ("最小回归通过", report["checklist"]["minimum_regression_passed"]),
        ("一致性检查完成", report["checklist"]["consistency_check_completed"]),
        ("风险例外报告输出", report["checklist"]["risk_exception_report_output"]),
        ("异常项已归因", report["checklist"]["anomalies_attributed"]),
    ]
    for label, done in auto_items:
        mark = "[x]" if done is True else "[ ]"
        if done is None:
            lines.append(f"- {mark} {label}（未执行自动检查）")
        else:
            lines.append(f"- {mark} {label}")
    lines.append("")
    lines.append("## Regression Check")
    lines.append("")
    regression = report["regression"]
    if regression.get("executed"):
        lines.append(f"- Command: `{regression['command']}`")
        lines.append(f"- Return code: `{regression['return_code']}`")
        lines.append(f"- Passed: `{regression['passed']}`")
        if regression.get("output_tail"):
            lines.append("")
            lines.append("```text")
            lines.append(regression["output_tail"])
            lines.append("```")
    else:
        lines.append("_not executed_")
    lines.append("")
    lines.append("## Change Log (Auto)")
    lines.append("")
    change_log = report["change_log"]
    if change_log.get("executed"):
        lines.append(f"- Window: last `{change_log['since_days']}` day(s)")
        lines.append("")
        lines.append(_format_table(change_log["entries"], ["date", "commit", "subject"]))
    else:
        lines.append("_not available_")
    lines.append("")
    lines.append("## Rollback Marker (Auto)")
    lines.append("")
    rollback_marker = report["rollback_marker"]
    if rollback_marker.get("executed") and rollback_marker.get("tag"):
        if rollback_marker.get("source") == "commit":
            lines.append(f"- Rollback baseline (commit): `{rollback_marker['tag']}`")
        else:
            lines.append(f"- Latest tag: `{rollback_marker['tag']}`")
    else:
        lines.append("_not available_")
    lines.append("")
    lines.append("## Checklist (Manual)")
    lines.append("")
    for label in report["manual_checklist_items"]:
        lines.append(f"- [ ] {label}")
    lines.append("")
    lines.append("## Incomplete Tasks")
    lines.append("")
    for task in report["incomplete_tasks"]:
        lines.append(f"- {task}")
    lines.append("")
    if report["parse_errors"]:
        lines.append("## Parse Errors")
        lines.append("")
        lines.append(_format_table(report["parse_errors"], ["file", "error"]))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly operating KPI/risk audit.")
    parser.add_argument("--results-dir", default="results", help="Directory for backtest outputs.")
    parser.add_argument("--pattern", default="backtest*.json", help="Glob pattern in results dir.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Optional explicit input JSON files. If set, results-dir/pattern is ignored.",
    )
    parser.add_argument(
        "--thresholds",
        default="config/weekly_operating_thresholds.json",
        help="Path to thresholds JSON. Uses defaults when file is missing.",
    )
    parser.add_argument(
        "--consistency-thresholds",
        default="config/consistency_thresholds.json",
        help="Path to consistency thresholds JSON. Uses defaults when file is missing.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/weekly-operating-audit.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/weekly-operating-audit.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when risk exceptions exist or no strategy rows can be extracted.",
    )
    parser.add_argument(
        "--strict-close",
        action="store_true",
        help="Exit non-zero unless weekly sign-off status is READY_FOR_CLOSE.",
    )
    parser.add_argument(
        "--signoff-json",
        default="artifacts/weekly-signoff-pack.json",
        help="Path to weekly sign-off JSON used by --strict-close.",
    )
    parser.add_argument(
        "--close-gate-only",
        action="store_true",
        help="Only validate close gate status from --signoff-json.",
    )
    parser.add_argument(
        "--close-gate-output-md",
        default="artifacts/weekly-close-gate.md",
        help="Output markdown path for close gate summary.",
    )
    parser.add_argument(
        "--close-gate-output-json",
        default="artifacts/weekly-close-gate.json",
        help="Output JSON path for close gate summary.",
    )
    parser.add_argument(
        "--regression-cmd",
        default="",
        help="Optional regression command to execute and include in the audit report.",
    )
    parser.add_argument(
        "--change-log-days",
        type=int,
        default=7,
        help="Look-back window (days) for auto-generated change log.",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    signoff_json_path = (repo_root / args.signoff_json).resolve()
    close_gate_md = (repo_root / args.close_gate_output_md).resolve()
    close_gate_json = (repo_root / args.close_gate_output_json).resolve()

    def _write_close_gate_report(
        close_ready: bool, close_detail: str, signoff_payload: dict[str, Any]
    ) -> None:
        close_report = _build_close_gate_report(
            signoff_json_path=signoff_json_path,
            close_ready=close_ready,
            close_detail=close_detail,
            signoff_payload=signoff_payload,
        )
        close_gate_md.parent.mkdir(parents=True, exist_ok=True)
        close_gate_json.parent.mkdir(parents=True, exist_ok=True)
        close_gate_md.write_text(_close_gate_to_markdown(close_report), encoding="utf-8")
        close_gate_json.write_text(
            json.dumps(close_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.close_gate_only:
        close_ready, close_detail, signoff_payload = _evaluate_close_gate(signoff_json_path)
        _write_close_gate_report(close_ready, close_detail, signoff_payload)
        if close_ready:
            print("Weekly close gate: READY_FOR_CLOSE.")
            return 0
        print(f"Weekly close gate: not ready ({close_detail}).")
        return 2 if args.strict_close else 0

    thresholds_path = (repo_root / args.thresholds).resolve()
    thresholds = _load_thresholds(thresholds_path)
    consistency_thresholds_path = (repo_root / args.consistency_thresholds).resolve()
    consistency_thresholds = _load_consistency_thresholds(consistency_thresholds_path)

    if args.inputs:
        input_files = [Path(p).resolve() for p in args.inputs]
    else:
        results_dir = (repo_root / args.results_dir).resolve()
        input_files = _discover_input_files(results_dir, args.pattern)

    if not input_files:
        print("Weekly operating audit: no input files found.")
        return 2 if (args.strict or args.strict_close) else 0

    regression_result: dict[str, Any] | None = None
    if args.regression_cmd.strip():
        regression_cmd = shlex.split(args.regression_cmd)
        if not regression_cmd:
            raise ValueError("Regression command is empty after parsing")
        completed = subprocess.run(
            regression_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
        output_lines = combined_output.splitlines()
        regression_result = {
            "executed": True,
            "command": args.regression_cmd,
            "passed": completed.returncode == 0,
            "return_code": completed.returncode,
            "output_tail": "\n".join(output_lines[-40:]),
        }

    change_log = _collect_recent_changes(repo_root, max(args.change_log_days, 1))
    rollback_marker = _detect_latest_tag(repo_root)

    report = _build_report(
        input_files,
        thresholds,
        consistency_thresholds=consistency_thresholds,
        regression_result=regression_result,
        change_log=change_log,
        rollback_marker=rollback_marker,
    )

    md_path = (repo_root / args.output_md).resolve()
    json_path = (repo_root / args.output_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    had_issue = False
    if report["summary"]["exceptions"] > 0:
        print(f"Weekly operating audit: {report['summary']['exceptions']} risk exception(s).")
        had_issue = True
        if args.strict:
            return 2
    if regression_result is not None and not regression_result["passed"]:
        print("Weekly operating audit: regression command failed.")
        had_issue = True
        if args.strict:
            return 2
    if report["summary"]["strategies"] == 0:
        print("Weekly operating audit: no strategy rows extracted.")
        had_issue = True
        if args.strict:
            return 2
    if report["summary"]["consistency_exceptions"] > 0:
        print(
            f"Weekly operating audit: {report['summary']['consistency_exceptions']} consistency exception(s)."
        )
        had_issue = True
        if args.strict:
            return 2
    if args.strict_close:
        close_ready, close_detail, signoff_payload = _evaluate_close_gate(signoff_json_path)
        _write_close_gate_report(close_ready, close_detail, signoff_payload)
        if not close_ready:
            print(f"Weekly operating audit: close gate not ready ({close_detail}).")
            return 2
    if not had_issue:
        print("Weekly operating audit: no threshold exceptions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
