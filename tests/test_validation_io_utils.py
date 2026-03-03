"""Tests for validation script I/O helpers."""

from __future__ import annotations

import json

from validation_scripts.io_utils import load_json, write_json, write_text


def test_load_json_reads_object_payload(tmp_path):
    payload = {"a": 1, "b": "x"}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_json(str(path))

    assert loaded == payload


def test_write_json_creates_parent_dir_and_formats_output(tmp_path):
    path = tmp_path / "nested" / "payload.json"
    write_json(str(path), {"k": "v"})

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"k": "v"' in text


def test_write_text_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "note.md"
    write_text(str(path), "# hello\n")

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "# hello\n"
