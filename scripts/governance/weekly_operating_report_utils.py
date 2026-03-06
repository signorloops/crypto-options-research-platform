"""Shared report-assembly helpers for weekly operating audit."""

from __future__ import annotations

from typing import Any


def build_consistency_checks(
    *,
    latest_by_strategy: dict[str, dict[str, Any]],
    previous_by_strategy: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Compare latest and previous strategy snapshots against consistency thresholds."""
    checks: list[dict[str, Any]] = []
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
        if abs_pnl_diff is not None and abs_pnl_diff > thresholds["max_abs_pnl_diff"]:
            breaches.append(f"abs_pnl_diff>{thresholds['max_abs_pnl_diff']}")
        if abs_sharpe_diff is not None and abs_sharpe_diff > thresholds["max_abs_sharpe_diff"]:
            breaches.append(f"abs_sharpe_diff>{thresholds['max_abs_sharpe_diff']}")
        if (
            abs_max_drawdown_diff is not None
            and abs_max_drawdown_diff > thresholds["max_abs_max_drawdown_diff"]
        ):
            breaches.append(f"abs_max_drawdown_diff>{thresholds['max_abs_max_drawdown_diff']}")

        checks.append(
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
    return checks


def normalize_optional_baseline_report(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return the supplied baseline report or the default missing payload."""
    if result is not None:
        return dict(result)
    return {
        "executed": False,
        "summary": {"all_passed": None},
        "error": "",
        "path": "",
    }


def resolve_optional_report_check(report: dict[str, Any], *, required: bool) -> bool | None:
    """Return pass/fail/unknown for optional baseline reports."""
    all_passed = report.get("summary", {}).get("all_passed")
    if all_passed is None:
        return False if required else None
    return bool(all_passed)


def build_operating_checklist(
    *,
    snapshot_rows: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    consistency_checks: list[dict[str, Any]],
    consistency_exceptions: list[dict[str, Any]],
    risk_exceptions: list[dict[str, Any]],
    change_log: dict[str, Any] | None,
    rollback_marker: dict[str, Any] | None,
    regression_result: dict[str, Any] | None,
    performance_check: bool | None,
    latency_check: bool | None,
) -> dict[str, Any]:
    """Build the weekly audit checklist from shared report inputs."""
    consistency_baseline_available = len(consistency_checks) > 0
    rollback_marked = bool(
        rollback_marker
        and rollback_marker.get("source") == "tag"
        and rollback_marker.get("tag")
    )
    return {
        "kpi_snapshot_updated": bool(snapshot_rows),
        "experiment_ids_assigned": bool(snapshot_rows)
        and all(bool(row.get("experiment_id")) for row in snapshot_rows),
        "risk_thresholds_confirmed": True,
        "change_log_complete": bool(
            change_log
            and change_log.get("executed")
            and not change_log.get("shallow", False)
            and change_log.get("count", 0) > 0
        ),
        "rollback_version_marked": rollback_marked,
        "rollback_marker_from_tag": rollback_marked,
        "minimum_regression_passed": (
            bool(regression_result["passed"])
            if regression_result is not None and regression_result.get("executed")
            else None
        ),
        "performance_baseline_passed": performance_check,
        "latency_baseline_passed": latency_check,
        "consistency_check_completed": (
            len(parse_errors) == 0
            and consistency_baseline_available
            and len(consistency_exceptions) == 0
        ),
        "risk_exception_report_output": True,
        "anomalies_attributed": len(risk_exceptions) == 0,
    }


def build_report_summary(
    *,
    snapshot_rows: list[dict[str, Any]],
    risk_exceptions: list[dict[str, Any]],
    consistency_checks: list[dict[str, Any]],
    consistency_exceptions: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
) -> dict[str, int]:
    """Build compact top-line counts for the audit report."""
    return {
        "strategies": len(snapshot_rows),
        "exceptions": len(risk_exceptions),
        "consistency_pairs": len(consistency_checks),
        "consistency_exceptions": len(consistency_exceptions),
        "parse_errors": len(parse_errors),
    }
