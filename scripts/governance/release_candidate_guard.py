#!/usr/bin/env python3
"""Verify metadata and sign-off evidence for release-candidate readiness."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.report_utils import (  # noqa: E402
    load_json_object as _load_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)


def _load_project_metadata(pyproject_path: Path) -> dict[str, Any]:
    with pyproject_path.open("rb") as fh:
        payload = tomllib.load(fh)
    project = payload.get("project", {})
    if not isinstance(project, dict):
        raise ValueError(f"Invalid [project] table in {pyproject_path}")
    return project


def _normalize_text_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def evaluate_release_candidate(*, pyproject_path: Path, signoff_path: Path) -> dict[str, Any]:
    project = _load_project_metadata(pyproject_path)
    signoff = _load_json(signoff_path)

    version = str(project.get("version", "")).strip()
    classifiers = _normalize_text_list(project.get("classifiers", []))
    signoff_status = str(signoff.get("status", "")).strip()
    auto_blockers = _normalize_text_list(signoff.get("auto_blockers", []))
    pending_items = _normalize_text_list(signoff.get("pending_items", []))

    failures: list[str] = []
    if not version.endswith("rc1"):
        failures.append(f"project version is not a release candidate: {version or 'missing'}")
    if "Development Status :: 3 - Alpha" in classifiers:
        failures.append("project metadata is still alpha")
    if "Development Status :: 4 - Beta" not in classifiers:
        failures.append("project metadata is missing beta classifier")
    if signoff_status != "READY_FOR_CLOSE":
        failures.append(
            f"weekly sign-off status is not READY_FOR_CLOSE: {signoff_status or 'missing'}"
        )
    if auto_blockers:
        failures.append("weekly sign-off has auto blockers: " + ", ".join(auto_blockers))
    if pending_items:
        failures.append("weekly sign-off has pending items: " + ", ".join(pending_items))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "metadata": {
            "version": version,
            "classifiers": classifiers,
        },
        "signoff": {
            "status": signoff_status,
            "auto_blockers": auto_blockers,
            "pending_items": pending_items,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Release Candidate Guard",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Passed: `{report['passed']}`",
        f"- Version: `{report['metadata']['version'] or 'missing'}`",
        f"- Sign-off status: `{report['signoff']['status'] or 'missing'}`",
        "",
        "## Failures",
        "",
    ]
    failures = report.get("failures", [])
    if failures:
        lines.extend(f"- [ ] {failure}" for failure in failures)
    else:
        lines.append("- [x] None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release-candidate readiness.")
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--signoff-json",
        default="artifacts/weekly-signoff-pack.json",
        help="Path to weekly sign-off JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/release-candidate-guard.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/release-candidate-guard.md",
        help="Path to Markdown report output.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the guard fails.",
    )
    args = parser.parse_args()

    report = evaluate_release_candidate(
        pyproject_path=Path(args.pyproject).resolve(),
        signoff_path=Path(args.signoff_json).resolve(),
    )
    _write_json(Path(args.output_json).resolve(), report)
    _write_markdown(Path(args.output_md).resolve(), _render_markdown(report))

    if report["passed"]:
        print("Release candidate guard passed.")
        return 0

    print("Release candidate guard failed:")
    for failure in report["failures"]:
        print(f"- {failure}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
