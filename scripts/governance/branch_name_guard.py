#!/usr/bin/env python3
"""Fail when git branch names contain forbidden keywords."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _git_branches(root: Path) -> list[str]:
    cmd = ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes/origin"]
    proc = subprocess.run(cmd, cwd=root, check=True, text=True, capture_output=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard forbidden branch-name keywords.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    legacy_token = "".join(chr(code) for code in (99, 111, 100, 101, 120))
    parser.add_argument(
        "--forbidden",
        nargs="+",
        default=[legacy_token],
        help="Case-insensitive forbidden tokens for branch names.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    branches = _git_branches(root)
    forbidden = [token.lower() for token in args.forbidden]

    violations: list[str] = []
    for branch in branches:
        lowered = branch.lower()
        if any(token in lowered for token in forbidden):
            violations.append(branch)

    if violations:
        print("Branch name guard failed. Forbidden tokens found:")
        for item in violations:
            print(f"- {item}")
        return 2

    print("Branch name guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
