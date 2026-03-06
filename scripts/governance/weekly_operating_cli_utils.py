"""CLI helpers for weekly operating audit orchestration."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


def resolve_input_files(
    *,
    repo_root: Path,
    explicit_inputs: Sequence[str],
    results_dir: str,
    pattern: str,
    discover_input_files: Callable[[Path, str], list[Path]],
) -> list[Path]:
    """Resolve audit inputs from explicit paths or discovery rules."""
    if explicit_inputs:
        return [Path(value).resolve() for value in explicit_inputs]
    return discover_input_files((repo_root / results_dir).resolve(), pattern)


def run_regression_command(
    command: str,
    *,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Optional[dict[str, Any]]:
    """Execute an optional regression command without shell interpolation."""
    if not command.strip():
        return None
    regression_cmd = shlex.split(command)
    if not regression_cmd:
        raise ValueError("Regression command is empty after parsing")
    completed = runner(
        regression_cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
    output_lines = combined_output.splitlines()
    return {
        "executed": True,
        "command": command,
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "output_tail": "\n".join(output_lines[-40:]),
    }


def collect_issue_messages(
    report: dict[str, Any],
    *,
    regression_result: Optional[dict[str, Any]],
    require_performance: bool,
    require_latency: bool,
) -> list[str]:
    """Return human-readable audit issue messages in report order."""
    messages: list[str] = []
    summary = report.get("summary", {})
    checklist = report.get("checklist", {})
    if summary.get("exceptions", 0) > 0:
        messages.append(
            f"Weekly operating audit: {summary['exceptions']} risk exception(s)."
        )
    if regression_result is not None and not regression_result.get("passed"):
        messages.append("Weekly operating audit: regression command failed.")
    if summary.get("strategies", 0) == 0:
        messages.append("Weekly operating audit: no strategy rows extracted.")
    if summary.get("consistency_exceptions", 0) > 0:
        messages.append(
            "Weekly operating audit: "
            f"{summary['consistency_exceptions']} consistency exception(s)."
        )
    if require_performance and checklist.get("performance_baseline_passed") is not True:
        messages.append("Weekly operating audit: performance baseline missing or failing.")
    if require_latency and checklist.get("latency_baseline_passed") is not True:
        messages.append("Weekly operating audit: latency baseline missing or failing.")
    return messages
