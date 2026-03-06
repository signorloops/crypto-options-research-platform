#!/usr/bin/env python3
"""Markdown render helpers for weekly operating audit outputs."""

from __future__ import annotations

from typing import Any

from scripts.governance.report_utils import format_markdown_table as _format_table


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def summarize_items(items: list[str], limit: int = 4) -> str:
    if not items:
        return "None"
    visible = items[:limit]
    summary = " / ".join(visible)
    remaining = len(items) - len(visible)
    return f"{summary} (+{remaining} more)" if remaining > 0 else summary


def build_close_gate_pr_brief(
    *,
    close_ready: bool,
    close_detail: str,
    signoff_status: str,
    auto_blockers: list[str],
    pending_items: list[str],
    action_items: list[str],
) -> str:
    return "\n".join(
        [
            "### Weekly Close Gate",
            f"- Status: {'PASS' if close_ready else 'FAIL'} (`{close_detail}`)",
            f"- Signoff status: `{signoff_status or 'UNKNOWN'}`",
            f"- Auto blockers: {summarize_items(auto_blockers)}",
            f"- Pending items: {summarize_items(pending_items)}",
            f"- Next actions: {summarize_items(action_items, limit=3)}",
        ]
    )


def build_close_gate_markdown(report: dict[str, Any]) -> str:
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
    sections = [
        ("## Auto Blockers", report["auto_blockers"]),
        ("## Pending Items", report["pending_items"]),
        ("## Missing Manual Checks", report["manual_missing"]),
        ("## Missing Role Sign-offs", report["role_signoffs_missing"]),
    ]
    for title, items in sections:
        lines.append(title)
        lines.append("")
        if items:
            for item in items:
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


def build_weekly_operating_markdown(report: dict[str, Any]) -> str:
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
    snapshot_rows = [
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
        for row in report["kpi_snapshot"]
    ]
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
    consistency_rows = [
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
        for row in report["consistency_checks"]
    ]
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
        ("性能基线达标", report["checklist"]["performance_baseline_passed"]),
        ("延迟基线达标", report["checklist"]["latency_baseline_passed"]),
        ("一致性检查完成", report["checklist"]["consistency_check_completed"]),
        ("风险例外报告输出", report["checklist"]["risk_exception_report_output"]),
        ("异常项已归因", report["checklist"]["anomalies_attributed"]),
    ]
    for label, done in auto_items:
        mark = "[x]" if done is True else "[ ]"
        lines.append(f"- {mark} {label}" if done is not None else f"- {mark} {label}（未执行自动检查）")
    lines.append("")
    lines.append("## Regression Check")
    lines.append("")
    regression = report["regression"]
    if regression.get("executed"):
        lines.append(f"- Command: `{regression['command']}`")
        lines.append(f"- Return code: `{regression['return_code']}`")
        lines.append(f"- Passed: `{regression['passed']}`")
        if regression.get("output_tail"):
            lines.extend(["", "```text", regression["output_tail"], "```"])
    else:
        lines.append("_not executed_")
    lines.append("")
    lines.append("## Algorithm Performance Baseline")
    lines.append("")
    performance = report["performance_baseline"]
    if performance.get("executed"):
        perf_summary = performance.get("summary", {})
        lines.append(f"- All passed: `{perf_summary.get('all_passed')}`")
        lines.append(
            f"- Checks passed: `{perf_summary.get('checks_passed')}/{perf_summary.get('checks_total')}`"
        )
        metrics = performance.get("metrics", {})
        if isinstance(metrics.get("var_monte_carlo"), dict):
            lines.append(
                f"- VaR Monte Carlo P95 (ms): `{_fmt(_to_float(metrics['var_monte_carlo'].get('p95_ms')), 4)}`"
            )
        if isinstance(metrics.get("backtest_engine"), dict):
            lines.append(
                f"- Backtest Engine P95 (ms): `{_fmt(_to_float(metrics['backtest_engine'].get('p95_ms')), 4)}`"
            )
    else:
        lines.append("_not available_")
        if performance.get("error"):
            lines.append(f"- Error: `{performance['error']}`")
    lines.append("")
    lines.append("## Latency Baseline")
    lines.append("")
    latency = report["latency_baseline"]
    if latency.get("executed"):
        latency_summary = latency.get("summary", {})
        lines.append(f"- All passed: `{latency_summary.get('all_passed')}`")
        lines.append(
            f"- Checks passed: `{latency_summary.get('checks_passed')}/{latency_summary.get('checks_total')}`"
        )
        benchmarks = latency.get("benchmarks", [])
        if isinstance(benchmarks, list) and benchmarks:
            benchmark_rows = [
                {
                    "name": row.get("name", ""),
                    "p95_ms": _fmt(_to_float(row.get("p95_ms")), 4),
                    "target_ms": _fmt(_to_float(row.get("target_ms")), 4),
                    "meets_target": row.get("meets_target"),
                }
                for row in benchmarks
            ]
            lines.extend(
                [
                    "",
                    _format_table(benchmark_rows, ["name", "p95_ms", "target_ms", "meets_target"]),
                ]
            )
    else:
        lines.append("_not available_")
        if latency.get("error"):
            lines.append(f"- Error: `{latency['error']}`")
    lines.append("")
    lines.append("## Change Log (Auto)")
    lines.append("")
    change_log = report["change_log"]
    if change_log.get("executed"):
        lines.append(f"- Window: last `{change_log['since_days']}` day(s)")
        if change_log.get("shallow"):
            lines.append("- Repository clone is shallow; change log may be incomplete.")
        lines.extend(["", _format_table(change_log["entries"], ["date", "commit", "subject"])])
    else:
        lines.append("_not available_")
    lines.append("")
    lines.append("## Rollback Marker (Auto)")
    lines.append("")
    rollback_marker = report["rollback_marker"]
    if rollback_marker.get("executed") and rollback_marker.get("tag"):
        prefix = "Rollback baseline (commit)" if rollback_marker.get("source") == "commit" else "Latest tag"
        lines.append(f"- {prefix}: `{rollback_marker['tag']}`")
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
        lines.append(f"- [ ] {task}")
    lines.append("")
    if report["parse_errors"]:
        lines.extend(
            [
                "## Parse Errors",
                "",
                _format_table(report["parse_errors"], ["file", "error"]),
                "",
            ]
        )
    return "\n".join(lines)
