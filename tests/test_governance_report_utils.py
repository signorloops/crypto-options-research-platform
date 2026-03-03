"""Tests for shared governance report helper utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.governance.report_utils import (
    as_bool,
    discover_input_files,
    format_markdown_table,
    load_json_object,
    load_optional_json_object,
    write_json,
    write_markdown,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_json_object_requires_top_level_dict(tmp_path):
    path = tmp_path / "sample.json"
    _write(path, json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="Expected top-level object"):
        load_json_object(path)


def test_discover_input_files_sorted_by_mtime_desc(tmp_path):
    older = tmp_path / "results" / "a.json"
    newer = tmp_path / "results" / "b.json"
    _write(older, "{}")
    _write(newer, "{}")
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 10))

    found = discover_input_files(tmp_path / "results", "*.json")

    assert found == [newer, older]


def test_format_markdown_table_renders_rows():
    table = format_markdown_table(
        rows=[{"k": "v1"}, {"k": "v2"}],
        columns=["k"],
    )

    assert "| k |" in table
    assert "| v1 |" in table
    assert "| v2 |" in table


def test_load_optional_json_object_missing_returns_empty(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_optional_json_object(missing) == {}


def test_as_bool_normalizes_common_scalar_values():
    assert as_bool(True) is True
    assert as_bool(1) is True
    assert as_bool("yes") is True
    assert as_bool("false") is False
    assert as_bool(0) is False


def test_write_markdown_creates_parent_dirs(tmp_path):
    output = tmp_path / "nested" / "report.md"
    write_markdown(output, "# Demo")

    assert output.exists()
    assert output.read_text(encoding="utf-8") == "# Demo"


def test_write_json_uses_stable_utf8_format(tmp_path):
    output = tmp_path / "nested" / "report.json"
    write_json(output, {"msg": "中文", "n": 1})

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"msg": "中文", "n": 1}
