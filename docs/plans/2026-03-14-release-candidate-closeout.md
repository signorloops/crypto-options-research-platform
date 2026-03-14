# Release Candidate Closeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the repository from a technically-green alpha snapshot to a release-candidate-ready state by promoting release metadata, enforcing a closeout gate around manual sign-off, and generating a final READY_FOR_CLOSE evidence pack.

**Architecture:** The repo's code paths are now largely stable, so the remaining work should stay narrow and governance-focused. First promote package metadata away from alpha so the source tree reflects the actual maturity level. Then add a dedicated release-candidate guard that fails if metadata or sign-off state is incomplete. Finally use the existing weekly governance scripts to generate a fresh `READY_FOR_CLOSE` package and publish the release evidence.

**Tech Stack:** Python 3.9+, pytest, GNU Make, existing governance scripts under `scripts/governance/`, Markdown docs, JSON artifact reports.

---

## Current Remaining Gaps

- `pyproject.toml` still declares `version = "0.1.0"` and `Development Status :: 3 - Alpha`, which conflicts with the current "near-release-candidate" state.
- `artifacts/weekly-signoff-pack.json` was last generated on `2026-03-06` and still reports `status = "PENDING_MANUAL_SIGNOFF"` with 6 pending items.
- The repo has strong freeze checks (`make algorithm-freeze-check`) but no single gate that also requires non-alpha metadata plus `READY_FOR_CLOSE` sign-off status.
- Local `master` is ahead of `origin/master` by 1 commit; `tasks/lessons.md` and `uv.lock` are untracked and should remain out of scope unless intentionally included.

### Task 1: Promote package metadata from alpha to release candidate

**Files:**
- Create: `tests/test_release_metadata.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
from pathlib import Path

import tomllib


def test_project_metadata_reflects_release_candidate_state():
    pyproject = Path("pyproject.toml")
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["version"] == "0.2.0rc1"
    classifiers = set(project["classifiers"])
    assert "Development Status :: 3 - Alpha" not in classifiers
    assert "Development Status :: 4 - Beta" in classifiers
```

**Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_release_metadata.py
```

Expected: FAIL because the current metadata still says `0.1.0` and `Alpha`.

**Step 3: Write minimal implementation**

Update `pyproject.toml`:

```toml
[project]
version = "0.2.0rc1"
classifiers = [
    "Development Status :: 4 - Beta",
    ...
]
```

Update `README.md` to describe the repository as a release-candidate-track research platform instead of a raw alpha snapshot. Keep the edit narrow: release wording only, no structural README rewrite.

**Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_release_metadata.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml README.md tests/test_release_metadata.py
git commit -m "chore: promote package metadata to release candidate"
```

### Task 2: Add a release-candidate guard that requires closeout evidence

**Files:**
- Create: `scripts/governance/release_candidate_guard.py`
- Create: `tests/test_release_candidate_guard.py`
- Modify: `Makefile`
- Modify: `tests/test_make_governance_targets.py`

**Step 1: Write the failing test**

```python
from scripts.governance.release_candidate_guard import evaluate_release_candidate


def test_evaluate_release_candidate_fails_on_alpha_metadata(tmp_path):
    report = evaluate_release_candidate(
        pyproject_path=tmp_path / "pyproject.toml",
        signoff_path=tmp_path / "weekly-signoff-pack.json",
    )
    assert report["passed"] is False
    assert "project metadata is still alpha" in report["failures"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_release_candidate_guard.py
```

Expected: FAIL because the guard module does not exist yet.

**Step 3: Write minimal implementation**

Implement `scripts/governance/release_candidate_guard.py` with one pure function plus CLI:

```python
def evaluate_release_candidate(*, pyproject_path: Path, signoff_path: Path) -> dict[str, Any]:
    ...
```

Guard rules:

- fail when `project.version` does not end with `rc1`
- fail when `Development Status :: 3 - Alpha` is present
- fail when sign-off report status is not `READY_FOR_CLOSE`
- fail when sign-off report contains any `auto_blockers` or `pending_items`

Emit:

- `artifacts/release-candidate-guard.json`
- `artifacts/release-candidate-guard.md`

Add Make target:

```make
release-candidate-check:
	$(PYTHON) scripts/governance/release_candidate_guard.py \
		--pyproject pyproject.toml \
		--signoff-json artifacts/weekly-signoff-pack.json \
		--output-json artifacts/release-candidate-guard.json \
		--output-md artifacts/release-candidate-guard.md \
		--strict
```

**Step 4: Run targeted tests and dry-run Make assertions**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_release_metadata.py \
  tests/test_release_candidate_guard.py \
  tests/test_make_governance_targets.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/governance/release_candidate_guard.py tests/test_release_candidate_guard.py Makefile tests/test_make_governance_targets.py
git commit -m "feat: add release candidate readiness guard"
```

### Task 3: Close the manual sign-off gap and regenerate release artifacts

**Files:**
- Modify: `docs/governance-operations.md`
- Modify: `artifacts/weekly-manual-status.json`
- Modify: `artifacts/weekly-signoff-pack.json`
- Modify: `artifacts/weekly-signoff-pack.md`
- Modify: `artifacts/release-candidate-guard.json`
- Modify: `artifacts/release-candidate-guard.md`

**Step 1: Add the missing operational instructions to docs**

Document the exact closeout sequence in `docs/governance-operations.md`:

```bash
make weekly-manual-update MANUAL_ARGS="--check gray_release_completed=true --check observation_24h_completed=true --check adr_signed=true --signoff research=<name> --signoff engineering=<name> --signoff risk=<name>"
make weekly-signoff-pack
make release-candidate-check
```

Include one note: signer names must be real owners, not placeholders.

**Step 2: Verify the current state is still blocked before applying manual closeout**

Run:

```bash
./.venv/bin/python scripts/governance/release_candidate_guard.py \
  --pyproject pyproject.toml \
  --signoff-json artifacts/weekly-signoff-pack.json \
  --strict
```

Expected: non-zero exit with failures describing pending sign-off items if Task 1 has already landed but sign-off is still incomplete.

**Step 3: Apply explicit manual closeout inputs**

Run with real signers:

```bash
make weekly-manual-update MANUAL_ARGS="--check gray_release_completed=true --check observation_24h_completed=true --check adr_signed=true --signoff research=<research-owner> --signoff engineering=<engineering-owner> --signoff risk=<risk-owner>"
make weekly-signoff-pack
make release-candidate-check
```

Expected:

- `artifacts/weekly-signoff-pack.json` reports `status = "READY_FOR_CLOSE"`
- `artifacts/release-candidate-guard.json` reports `"passed": true`

**Step 4: Run the freeze gate plus the new release-candidate gate**

Run:

```bash
make algorithm-freeze-check
make release-candidate-check
```

Expected: both commands exit 0.

**Step 5: Commit**

```bash
git add docs/governance-operations.md artifacts/weekly-manual-status.json artifacts/weekly-signoff-pack.json artifacts/weekly-signoff-pack.md artifacts/release-candidate-guard.json artifacts/release-candidate-guard.md
git commit -m "docs: close weekly signoff and publish release evidence"
```

### Task 4: Publish the release-candidate snapshot cleanly

**Files:**
- Modify only if needed: `.gitignore`
- No code changes expected otherwise

**Step 1: Confirm the worktree contains only intentional release-candidate changes**

Run:

```bash
git status --short
```

Expected: no accidental files beyond the release-candidate work and any explicitly preserved local notes.

**Step 2: Push the branch and publish the evidence**

Run:

```bash
git push origin master
```

If the team wants an explicit release baseline tag, run:

```bash
make prepare-rollback-tag
git push origin --tags
```

**Step 3: Record the published verification set**

Capture these commands in the release note or PR body:

```bash
make algorithm-freeze-check
make weekly-signoff-pack
make release-candidate-check
```

Expected: all three succeeded on the published commit.

**Step 4: Verify the remote-facing state**

Run:

```bash
git status --short --branch
```

Expected: working tree clean or only intentionally untracked local files; branch no longer ahead after push.

**Step 5: Commit only if doc or ignore adjustments were needed**

```bash
git add .gitignore
git commit -m "chore: ignore local release leftovers"
```

Skip this step if no file changed.
