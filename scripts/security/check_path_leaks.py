#!/usr/bin/env python3
"""Fail when repository files contain absolute local paths."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "unix_user_home": re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[^\s'\"`]+"),
    "windows_user_home": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\Users\\[^\s'\"`]+"),
    "file_uri": re.compile(r"file:///[A-Za-z0-9._~/%+-][^\s'\"`]*"),
}


def repo_root() -> Path:
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
    )
    return Path(output.decode().strip())


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    rel_paths = [p for p in output.decode().split("\0") if p]
    return [root / rel for rel in rel_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect absolute local path leakage in repository files."
    )
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Scan all tracked files from git instead of the provided file list.",
    )
    parser.add_argument("files", nargs="*", help="Files passed from pre-commit.")
    return parser.parse_args()


def file_candidates(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.all_tracked:
        return tracked_files(root)

    if not args.files:
        return []

    resolved: list[Path] = []
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = root / p
        resolved.append(p)
    return resolved


def is_binary(raw: bytes) -> bool:
    return b"\0" in raw


def scan_file(path: Path, root: Path) -> list[str]:
    if not path.exists() or path.is_dir():
        return []

    raw = path.read_bytes()
    if is_binary(raw):
        return []

    findings: list[str] = []
    text = raw.decode("utf-8", errors="ignore")
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = str(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                col = match.start() + 1
                findings.append(f"{rel}:{line_no}:{col}: [{name}] {match.group(0)}")
    return findings


def main() -> int:
    args = parse_args()
    root = repo_root()
    candidates = file_candidates(args, root)
    findings: list[str] = []

    for path in candidates:
        findings.extend(scan_file(path, root))

    if findings:
        print("Absolute path leak check failed. Found:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print("Absolute path leak check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
