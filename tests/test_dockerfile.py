"""Regression tests for Docker build inputs."""

from pathlib import Path


def _dockerfile_lines() -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    return [line.strip() for line in dockerfile.splitlines() if line.strip()]


def test_dockerfile_only_copies_existing_dependency_manifests():
    repo_root = Path(__file__).resolve().parents[1]
    copy_lines = [
        line
        for line in _dockerfile_lines()
        if line.startswith("COPY ") and ("pyproject.toml" in line or "requirements.txt" in line)
    ]

    for line in copy_lines:
        parts = line.split()
        source = parts[1]
        assert (repo_root / source).exists(), f"Dockerfile copies missing file: {source}"


def test_dockerfile_copies_source_before_editable_install():
    lines = _dockerfile_lines()
    copy_source_index = next(i for i, line in enumerate(lines) if line == "COPY . .")
    install_index = next(i for i, line in enumerate(lines) if 'pip install -e "."' in line)

    assert copy_source_index < install_index
