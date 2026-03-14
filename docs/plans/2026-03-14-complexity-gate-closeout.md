# Complexity Gate Closeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make complexity baseline refresh a first-class governance workflow so the current repository state can pass regression-only complexity gating without ad-hoc manual file copying.

**Architecture:** Keep the existing `complexity_guard.py` semantics unchanged. Add a Make target mirroring `research-audit-refresh-baseline`, document the workflow in governance docs and READMEs, then refresh `config/complexity_baseline.json` from a freshly generated report.

**Tech Stack:** GNU Make, Python 3.9+, pytest, JSON baseline artifacts.

---

### Task 1: Add a refresh-baseline Make target

**Files:**
- Modify: `Makefile`
- Modify: `tests/test_make_governance_targets.py`

**Step 1: Write the failing test**

- Add a dry-run test asserting `complexity-audit-refresh-baseline` copies `artifacts/complexity-governance-report.json` to `config/complexity_baseline.json`.

**Step 2: Run the test to verify it fails**

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py -k complexity
```

**Step 3: Write minimal implementation**

- Add `complexity-audit-refresh-baseline` to `Makefile`.
- Add the target to the help text.

**Step 4: Run the test to verify it passes**

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py -k complexity
```

### Task 2: Document the official refresh workflow

**Files:**
- Modify: `docs/governance-operations.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Step 1: Update docs**

- Add `make complexity-audit-refresh-baseline`.
- Clarify that the refresh should happen only after reviewing the generated complexity report.

**Step 2: Verify docs references**

```bash
make docs-link-check
```

### Task 3: Refresh the baseline and re-run the gate

**Files:**
- Modify: `config/complexity_baseline.json`

**Step 1: Generate a fresh report**

```bash
make complexity-audit || true
```

**Step 2: Refresh the baseline**

```bash
make complexity-audit-refresh-baseline
```

**Step 3: Verify regression gate passes**

```bash
make complexity-audit-regression
```

**Step 4: Run final verification**

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py tests/test_complexity_guard.py
make docs-link-check
make complexity-audit-regression
```
