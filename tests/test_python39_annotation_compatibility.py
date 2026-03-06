"""Guard Python 3.9 compatibility for runtime-evaluated annotations."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCES = (
    list((REPO_ROOT / "core").rglob("*.py"))
    + list((REPO_ROOT / "data").rglob("*.py"))
    + list((REPO_ROOT / "research").rglob("*.py"))
    + list((REPO_ROOT / "strategies").rglob("*.py"))
    + list((REPO_ROOT / "execution").rglob("*.py"))
    + list((REPO_ROOT / "utils").rglob("*.py"))
    + list((REPO_ROOT / "scripts").rglob("*.py"))
    + list((REPO_ROOT / "tests").rglob("*.py"))
)
ANNOTATION_UNION_RE = re.compile(
    r"(^\s*def .*\s->\s*.*\|.*:)" r"|(^\s*\)\s*->\s*.*\|.*:)" r"|(^\s*[\w, ()]+\s*:\s*.*\|.*$)",
    re.MULTILINE,
)


def test_union_annotations_require_future_annotations_import():
    offenders: list[str] = []

    for path in PYTHON_SOURCES:
        text = path.read_text(encoding="utf-8")
        if "from __future__ import annotations" in text:
            continue
        if ANNOTATION_UNION_RE.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
