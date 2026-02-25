#!/usr/bin/env python3
"""Run daily regression commands and generate gate report artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_COMMANDS = [
    "python -m pytest -q tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py"
]


def _run_command(command: str, cwd: Path) -> dict[str, Any]:
    start = time.time()
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return {
            "command": command,
            "return_code": 127,
            "passed": False,
            "duration_sec": 0.0,
            "output_tail": str(exc),
        }

    try:
        completed = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
        return_code = completed.returncode
        combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
    except OSError as exc:
        return_code = 127
        combined_output = str(exc)

    lines = combined_output.splitlines()
    return {
        "command": command,
        "return_code": return_code,
        "passed": return_code == 0,
        "duration_sec": round(time.time() - start, 3),
        "output_tail": "\n".join(lines[-40:]),
    }


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [r for r in results if not r["passed"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "commands": len(results),
            "failed": len(failed),
            "all_passed": len(failed) == 0,
        },
        "results": results,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Daily Regression Gate")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Commands: `{report['summary']['commands']}`")
    lines.append(f"- Failed: `{report['summary']['failed']}`")
    lines.append(f"- All passed: `{report['summary']['all_passed']}`")
    lines.append("")
    for idx, row in enumerate(report["results"], start=1):
        lines.append(f"## Command {idx}")
        lines.append("")
        lines.append(f"- Command: `{row['command']}`")
        lines.append(f"- Return code: `{row['return_code']}`")
        lines.append(f"- Passed: `{row['passed']}`")
        lines.append(f"- Duration (s): `{row['duration_sec']}`")
        if row.get("output_tail"):
            lines.append("")
            lines.append("```text")
            lines.append(row["output_tail"])
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily regression gate commands.")
    parser.add_argument(
        "--cmd",
        action="append",
        dest="commands",
        help="Command to run. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/daily-regression-gate.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/daily-regression-gate.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any command fails.",
    )
    args = parser.parse_args()

    commands = args.commands if args.commands else list(DEFAULT_COMMANDS)
    cwd = Path(".").resolve()
    results = [_run_command(command, cwd=cwd) for command in commands]
    report = _build_report(results)

    md_path = Path(args.output_md).resolve()
    json_path = Path(args.output_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not report["summary"]["all_passed"]:
        print(f"Daily regression gate: {report['summary']['failed']} command(s) failed.")
        return 2 if args.strict else 0
    print("Daily regression gate: all commands passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
