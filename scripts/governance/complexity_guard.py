#!/usr/bin/env python3
"""
Complexity governance checker for weekly CI audits.

Computes structural complexity metrics over production Python modules and
compares them against configured thresholds.
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class Thresholds:
    max_python_files: int
    max_total_loc: int
    max_avg_loc_per_file: float
    max_file_loc: int
    soft_file_loc: int
    max_files_over_soft_loc: int
    max_function_loc: int
    soft_function_loc: int
    max_functions_over_soft_loc: int
    max_function_args: int
    max_methods_per_class: int
    max_classes_over_method_soft_limit: int
    soft_method_count_per_class: int


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _iter_python_files(
    root: Path, include_dirs: Sequence[str], exclude_dirs: Sequence[str]
) -> List[Path]:
    files: List[Path] = []
    exclude_set = set(exclude_dirs)

    for rel in include_dirs:
        base = root / rel
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if any(part in exclude_set for part in p.parts):
                continue
            files.append(p)

    # Stable order for deterministic reports.
    return sorted(set(files))


def _physical_loc(text: str) -> int:
    return len(text.splitlines())


def _code_loc(text: str) -> int:
    total = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        total += 1
    return total


def _func_args_count(node: ast.AST) -> int:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    args = node.args
    count = len(args.args) + len(args.kwonlyargs)
    if args.vararg:
        count += 1
    if args.kwarg:
        count += 1
    return count


def _function_length(node: ast.AST) -> int:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    if getattr(node, "end_lineno", None) is None:
        return 0
    return int(node.end_lineno - node.lineno + 1)


def _build_report(root: Path, config: Dict) -> Dict:
    scope = config["scope"]
    thresholds_cfg = config["thresholds"]
    thresholds = Thresholds(**thresholds_cfg)

    files = _iter_python_files(
        root=root,
        include_dirs=scope["include_dirs"],
        exclude_dirs=scope["exclude_dirs"],
    )
    if not files:
        raise RuntimeError("No Python files found in configured scope")

    file_rows: List[Dict] = []
    func_rows: List[Dict] = []
    class_rows: List[Dict] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(root))
        file_loc = _physical_loc(text)
        file_code_loc = _code_loc(text)
        file_rows.append(
            {
                "path": rel,
                "physical_loc": file_loc,
                "code_loc": file_code_loc,
            }
        )

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # If parsing fails, count the file but skip deep metrics for safety.
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_rows.append(
                    {
                        "path": rel,
                        "name": node.name,
                        "length": _function_length(node),
                        "args": _func_args_count(node),
                    }
                )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(
                    isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) for ch in node.body
                )
                class_rows.append(
                    {
                        "path": rel,
                        "name": node.name,
                        "method_count": int(method_count),
                    }
                )

    file_locs = [r["physical_loc"] for r in file_rows]
    avg_loc = statistics.mean(file_locs)
    max_file = max(file_rows, key=lambda x: x["physical_loc"])
    files_over_soft = [r for r in file_rows if r["physical_loc"] > thresholds.soft_file_loc]

    max_func = max(func_rows, key=lambda x: x["length"]) if func_rows else None
    funcs_over_soft = [r for r in func_rows if r["length"] > thresholds.soft_function_loc]
    max_args_func = max(func_rows, key=lambda x: x["args"]) if func_rows else None

    max_methods_class = max(class_rows, key=lambda x: x["method_count"]) if class_rows else None
    classes_over_soft = [
        r for r in class_rows if r["method_count"] > thresholds.soft_method_count_per_class
    ]

    metrics = {
        "python_files": len(file_rows),
        "total_loc": int(sum(file_locs)),
        "avg_loc_per_file": float(avg_loc),
        "max_file_loc": int(max_file["physical_loc"]),
        "files_over_soft_loc": len(files_over_soft),
        "max_function_loc": int(max_func["length"] if max_func else 0),
        "functions_over_soft_loc": len(funcs_over_soft),
        "max_function_args": int(max_args_func["args"] if max_args_func else 0),
        "max_methods_per_class": int(max_methods_class["method_count"] if max_methods_class else 0),
        "classes_over_method_soft_limit": len(classes_over_soft),
    }

    checks = [
        ("python_files", metrics["python_files"], thresholds.max_python_files),
        ("total_loc", metrics["total_loc"], thresholds.max_total_loc),
        ("avg_loc_per_file", metrics["avg_loc_per_file"], thresholds.max_avg_loc_per_file),
        ("max_file_loc", metrics["max_file_loc"], thresholds.max_file_loc),
        ("files_over_soft_loc", metrics["files_over_soft_loc"], thresholds.max_files_over_soft_loc),
        ("max_function_loc", metrics["max_function_loc"], thresholds.max_function_loc),
        (
            "functions_over_soft_loc",
            metrics["functions_over_soft_loc"],
            thresholds.max_functions_over_soft_loc,
        ),
        ("max_function_args", metrics["max_function_args"], thresholds.max_function_args),
        (
            "max_methods_per_class",
            metrics["max_methods_per_class"],
            thresholds.max_methods_per_class,
        ),
        (
            "classes_over_method_soft_limit",
            metrics["classes_over_method_soft_limit"],
            thresholds.max_classes_over_method_soft_limit,
        ),
    ]

    check_rows = []
    for name, value, limit in checks:
        status = "PASS" if value <= limit else "FAIL"
        check_rows.append(
            {
                "metric": name,
                "value": value,
                "threshold": limit,
                "status": status,
            }
        )

    return {
        "scope": scope,
        "thresholds": thresholds_cfg,
        "metrics": metrics,
        "checks": check_rows,
        "violations": [r for r in check_rows if r["status"] == "FAIL"],
        "top_files_by_loc": sorted(file_rows, key=lambda x: x["physical_loc"], reverse=True)[:15],
        "top_functions_by_loc": sorted(func_rows, key=lambda x: x["length"], reverse=True)[:15],
        "top_classes_by_methods": sorted(class_rows, key=lambda x: x["method_count"], reverse=True)[
            :15
        ],
    }


def _format_md_table(rows: Sequence[Dict], columns: Sequence[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _to_markdown(report: Dict) -> str:
    lines = []
    lines.append("# Weekly Complexity Governance Report")
    lines.append("")
    lines.append("## Summary Checks")
    lines.append("")
    lines.append(_format_md_table(report["checks"], ["metric", "value", "threshold", "status"]))
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    for key, value in report["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Top Files By LOC")
    lines.append("")
    lines.append(_format_md_table(report["top_files_by_loc"], ["path", "physical_loc", "code_loc"]))
    lines.append("")
    lines.append("## Top Functions By LOC")
    lines.append("")
    lines.append(
        _format_md_table(report["top_functions_by_loc"], ["path", "name", "length", "args"])
    )
    lines.append("")
    lines.append("## Top Classes By Method Count")
    lines.append("")
    lines.append(
        _format_md_table(report["top_classes_by_methods"], ["path", "name", "method_count"])
    )
    lines.append("")
    if report.get("regressions"):
        lines.append("## Regressions Vs Baseline")
        lines.append("")
        lines.append(
            _format_md_table(
                report["regressions"],
                ["metric", "value", "threshold", "baseline_value", "reason"],
            )
        )
        lines.append("")
    if report["violations"]:
        lines.append("## Violations")
        lines.append("")
        lines.append(
            _format_md_table(report["violations"], ["metric", "value", "threshold", "status"])
        )
    else:
        lines.append("## Violations")
        lines.append("")
        lines.append("No threshold violations.")
    lines.append("")
    return "\n".join(lines)


def _compute_regressions(report: Dict, baseline_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    regressions: List[Dict[str, Any]] = []
    for row in report.get("checks", []):
        if row.get("status") != "FAIL":
            continue

        metric = str(row.get("metric"))
        value = row.get("value")
        threshold = row.get("threshold")
        baseline_value = baseline_metrics.get(metric)
        if baseline_value is None:
            regressions.append(
                {
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "baseline_value": None,
                    "reason": "missing_baseline_metric",
                }
            )
            continue

        try:
            curr = float(value)
            limit = float(threshold)
            base = float(baseline_value)
        except (TypeError, ValueError):
            regressions.append(
                {
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "baseline_value": baseline_value,
                    "reason": "non_numeric_baseline_or_value",
                }
            )
            continue

        if base <= limit and curr > limit:
            reason = "new_violation"
        elif base > limit and curr > base:
            reason = "worsened_existing_violation"
        else:
            reason = ""

        if reason:
            regressions.append(
                {
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "baseline_value": baseline_value,
                    "reason": reason,
                }
            )
    return regressions


def _load_baseline_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    raise ValueError(f"Baseline report must contain a 'metrics' object: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complexity governance checks.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument(
        "--config",
        default="config/complexity_budget.json",
        help="Complexity budget config JSON path.",
    )
    parser.add_argument(
        "--report-md",
        default="artifacts/complexity-governance-report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/complexity-governance-report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any threshold is violated.",
    )
    parser.add_argument(
        "--baseline-json",
        default="",
        help="Optional baseline complexity report JSON for regression-only strict mode.",
    )
    parser.add_argument(
        "--strict-regression-only",
        action="store_true",
        help=(
            "When used with --strict, fail only on regressions versus --baseline-json, "
            "not on existing baseline violations."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    md_path = (root / args.report_md).resolve()
    json_path = (root / args.report_json).resolve()

    config = _load_config(config_path)
    report = _build_report(root, config)
    report["baseline"] = ""
    report["regressions"] = []

    baseline_metrics: Dict[str, Any] = {}
    if args.baseline_json:
        baseline_path = (root / args.baseline_json).resolve()
        baseline_metrics = _load_baseline_metrics(baseline_path)
        report["baseline"] = str(baseline_path)
        report["regressions"] = _compute_regressions(report, baseline_metrics)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.strict_regression_only and not args.baseline_json:
        print("Complexity check: --strict-regression-only requires --baseline-json.")
        return 2

    if args.strict and args.strict_regression_only:
        if report["regressions"]:
            print(
                "Complexity check: " f"{len(report['regressions'])} regression(s) versus baseline."
            )
            return 2
        if report["violations"]:
            print(
                "Complexity check: "
                f"{len(report['violations'])} existing violation(s), no regressions."
            )
            return 0
        print("Complexity check: all thresholds passed.")
        return 0

    if report["violations"]:
        print(f"Complexity check: {len(report['violations'])} violation(s) found.")
        if args.strict:
            return 2
    else:
        print("Complexity check: all thresholds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
