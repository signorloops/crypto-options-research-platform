"""CLI helpers for weekly operating audit orchestration."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Bound the regression command so a hung test run cannot block the audit forever.
SUBPROCESS_TIMEOUT_SEC = 60 * 60


def _as_text(value: object) -> str:
    """Best-effort conversion of captured subprocess output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


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
    timeout_sec: float = SUBPROCESS_TIMEOUT_SEC,
) -> Optional[dict[str, Any]]:
    """Execute an optional regression command without shell interpolation."""
    if not command.strip():
        return None
    try:
        regression_cmd = shlex.split(command)
    except ValueError as exc:
        # Malformed quoting must be recorded as a failed regression row instead of
        # crashing the whole audit run.
        return {
            "executed": True,
            "command": command,
            "passed": False,
            "return_code": 127,
            "output_tail": f"failed to parse regression command: {exc}",
        }
    if not regression_cmd:
        return {
            "executed": True,
            "command": command,
            "passed": False,
            "return_code": 127,
            "output_tail": "Regression command is empty after parsing",
        }
    try:
        completed = runner(
            regression_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        # A hung regression command must surface as a failed row, not a stalled audit.
        timed_out_output = "\n".join(
            part for part in (_as_text(exc.stdout), _as_text(exc.stderr)) if part
        )
        output_tail = (
            f"regression command timed out after {timeout_sec} seconds"
            + (f"\n{timed_out_output}" if timed_out_output else "")
        )
        return {
            "executed": True,
            "command": command,
            "passed": False,
            "return_code": 124,
            "output_tail": output_tail,
        }
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
