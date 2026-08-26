"""Tests for the local markdown link checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "docs" / "check_markdown_links.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_markdown_links_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_markdown_links module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_angle_bracket_target_with_spaces_kept_whole():
    """`[x](<docs/my file.md>)` must not be truncated to `docs/my`.

    Angle brackets exist precisely to allow spaces; the old code stripped the
    brackets and then applied the title-split heuristic, truncating the path.
    """
    module = _load_module()
    assert module._normalize_target("<docs/my file.md>") == "docs/my file.md"


def test_normalize_strips_title_after_plain_path():
    module = _load_module()
    assert module._normalize_target('docs/guide.md "Guide title"') == "docs/guide.md"


def test_normalize_anchor_untouched():
    module = _load_module()
    assert module._normalize_target("#section heading") == "#section heading"


def test_main_refuses_vacuous_pass(tmp_path, monkeypatch, capsys):
    """Zero markdown files found must be an error, not a silent pass.

    A renamed docs/ tree or typo'd --paths would otherwise turn the CI gate
    green while checking nothing.
    """
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_markdown_links.py", "--paths", str(tmp_path / "does_not_exist")],
    )
    exit_code = module.main()
    assert exit_code == 2
    assert "vacuously" in capsys.readouterr().out


def test_main_passes_with_real_files(tmp_path, monkeypatch):
    module = _load_module()
    (tmp_path / "README.md").write_text("[ok](./docs/a.md)\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("content\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_markdown_links.py", "--paths", str(tmp_path / "README.md")],
    )
    assert module.main() == 0


def test_main_flags_broken_local_link(tmp_path, monkeypatch):
    module = _load_module()
    (tmp_path / "README.md").write_text("[bad](./docs/missing.md)\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_markdown_links.py", "--paths", str(tmp_path / "README.md")],
    )
    assert module.main() == 2
