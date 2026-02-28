#!/usr/bin/env python3
"""Lightweight local markdown link checker."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")


def _iter_markdown_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*.md")):
                if candidate.is_file():
                    yield candidate
        elif path.is_file() and path.suffix.lower() == ".md":
            yield path


def _normalize_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    # Strip optional markdown title: path "title"
    if " " in target and not target.startswith("#"):
        target = target.split(" ", 1)[0].strip()
    return target


def _is_skipped(target: str) -> bool:
    if not target or target.startswith("#"):
        return True
    lower = target.lower()
    return lower.startswith(_SKIP_PREFIXES)


def _check_file(md_file: Path) -> list[str]:
    text = md_file.read_text(encoding="utf-8")
    errors: list[str] = []
    for match in _LINK_RE.finditer(text):
        target_raw = match.group(1)
        target = _normalize_target(target_raw)
        if _is_skipped(target):
            continue

        link_path = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not link_path:
            continue

        resolved = (md_file.parent / link_path).resolve()
        if not resolved.exists():
            line_no = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{md_file}:{line_no}: broken link -> {target_raw}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local markdown links.")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["README.md", "docs"],
        help="Markdown files or directories to check.",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    input_paths = [(repo_root / p).resolve() for p in args.paths]

    markdown_files = list(_iter_markdown_files(input_paths))
    if not markdown_files:
        print("No markdown files found for link checking.")
        return 0

    all_errors: list[str] = []
    for md_file in markdown_files:
        all_errors.extend(_check_file(md_file))

    if all_errors:
        print("Markdown link check failed:")
        for err in all_errors:
            print(f"- {err}")
        return 2

    print(f"Markdown link check passed ({len(markdown_files)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
