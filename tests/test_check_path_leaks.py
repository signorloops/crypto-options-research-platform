"""Tests for the absolute-path leak pre-commit/CI checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "security" / "check_path_leaks.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("check_path_leaks_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_path_leaks module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _findings(module, tmp_path: Path, content: str) -> list[str]:
    target = tmp_path / "file.txt"
    target.write_text(content, encoding="utf-8")
    return module.scan_file(target, tmp_path)


def test_detects_plain_windows_path(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, 'x = "C:\\Users\\bob\\file.txt"\n')
    assert any("windows_user_home" in f for f in findings)


def test_detects_escaped_windows_path_in_json(tmp_path):
    """Doubled backslashes (the JSON/Python/YAML literal form) must be caught.

    Previously the pattern required exactly one backslash, so the escaped form
    — the way a Windows path actually appears in committed source — passed.
    """
    module = _load_module()
    content = '{"data_dir": "C:\\\\Users\\\\bob\\\\secret.txt"}\n'
    findings = _findings(module, tmp_path, content)
    assert any("windows_user_home" in f for f in findings)


def test_detects_forward_slash_windows_path(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, 'p = "C:/Users/bob/file"\n')
    assert any("windows_user_home" in f for f in findings)


def test_detects_non_c_drive_letter(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, "D:\\Users\\alice\\doc\n")
    assert any("windows_user_home" in f for f in findings)


def test_unix_paths_still_detected(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, 'log_dir = "/Users/bob/logs"\n')
    assert any("unix_user_home" in f for f in findings)


def test_no_false_positive_without_users_component(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, 'x = "C:\\\\Data\\\\file"\n')
    assert findings == []


def test_no_false_positive_on_unix_url_path(tmp_path):
    module = _load_module()
    findings = _findings(module, tmp_path, "see https://example.com/Users/bob\n")
    assert findings == []
