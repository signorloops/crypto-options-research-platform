#!/usr/bin/env python3
"""Create or reuse a local rollback tag for the current release candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.report_utils import write_json as _write_json

DEFAULT_TAG_PREFIX = "backup-release"
DEFAULT_MESSAGE = "Rollback baseline for release candidate"


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def _resolve_commit(root: Path, ref: str) -> str:
    completed = _run_git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    return completed.stdout.strip()


def _short_commit(commit: str) -> str:
    return commit[:8]


def _tags_pointing_at(root: Path, ref: str) -> list[str]:
    completed = _run_git(root, "tag", "--points-at", ref)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _tag_target_commit(root: Path, tag: str) -> str:
    completed = _run_git(root, "rev-list", "-n", "1", tag)
    return completed.stdout.strip()


def _build_default_tag(prefix: str, commit: str, now: datetime | None = None) -> str:
    ts = now or datetime.now(timezone.utc)
    return f"{prefix}-{ts.strftime('%Y%m%d-%H%M%S')}-{_short_commit(commit)}"


def _select_reusable_tag(tags: list[str], prefix: str) -> str:
    preferred = sorted(tag for tag in tags if tag.startswith(f"{prefix}-"))
    if preferred:
        return preferred[-1]
    return sorted(tags)[-1] if tags else ""


def _ensure_tag(
    root: Path,
    *,
    ref: str,
    tag: str | None,
    tag_prefix: str,
    message: str,
) -> dict[str, Any]:
    commit = _resolve_commit(root, ref)
    existing_tags = _tags_pointing_at(root, ref)
    requested_tag = tag.strip() if isinstance(tag, str) else ""
    reusable_tag = _select_reusable_tag(existing_tags, tag_prefix)
    target_tag = requested_tag or reusable_tag or _build_default_tag(tag_prefix, commit)

    tag_exists = False
    try:
        tag_commit = _tag_target_commit(root, target_tag)
        tag_exists = True
    except subprocess.CalledProcessError:
        tag_commit = ""

    if tag_exists and tag_commit != commit:
        raise ValueError(
            f"Tag '{target_tag}' points to a different commit ({_short_commit(tag_commit)})."
        )

    created = False
    reused_existing = tag_exists
    if not tag_exists:
        _run_git(root, "tag", "-a", target_tag, ref, "-m", message)
        created = True
        reused_existing = False

    return {
        "executed": True,
        "created": created,
        "reused_existing": reused_existing,
        "tag": target_tag,
        "ref": ref,
        "commit": commit,
        "message": message,
        "push_command": f"git push origin {target_tag}",
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or reuse a rollback tag for the release candidate."
    )
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--ref", default="HEAD", help="Git ref to tag. Defaults to HEAD.")
    parser.add_argument(
        "--tag-prefix",
        default=DEFAULT_TAG_PREFIX,
        help=f"Prefix for generated tags. Defaults to {DEFAULT_TAG_PREFIX}.",
    )
    parser.add_argument("--tag", default="", help="Explicit tag name to create or reuse.")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Annotated tag message.")
    parser.add_argument(
        "--output-json",
        default="artifacts/rollback-tag.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_json = Path(args.output_json).resolve()

    try:
        report = _ensure_tag(
            root,
            ref=args.ref,
            tag=args.tag,
            tag_prefix=args.tag_prefix.strip() or DEFAULT_TAG_PREFIX,
            message=args.message.strip() or DEFAULT_MESSAGE,
        )
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _write_json(output_json, report)
    status = "created" if report["created"] else "reused"
    print(
        f"Rollback tag {status}: {report['tag']} -> {_short_commit(report['commit'])}. "
        f"Push when needed with `{report['push_command']}`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
