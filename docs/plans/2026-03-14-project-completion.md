# Project Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring the repo from alpha-with-known blockers to a release-candidate state by clearing the Python 3.9 test blocker, adding a reproducible Notebook 01 validation path, and aligning governance docs with runnable entrypoints.

**Architecture:** The work stays narrow and evidence-driven. First unblock `research/backtest/arena.py` so the declared Python floor (`>=3.9`) is actually honored and the existing backtest/reporting suites can run. Then replace the current ad hoc notebook verification with a deterministic script + artifact flow, and finally close the docs/Makefile drift so the documented governance commands are real commands.

**Tech Stack:** Python 3.9+, pytest, pandas, numpy, matplotlib, GNU Make, optional notebook extras (`nbclient` / `jupyter`) for final notebook refresh only.

---

## Current Blockers

- `./.venv/bin/python -m pytest -q -m "not integration"` fails during collection because `research/backtest/arena.py` uses runtime-evaluated `|` type hints without `from __future__ import annotations`.
- `tests/test_hawkes_comparison.py` and `tests/test_strategy_arena.py` are blocked by that import-time failure.
- `tasks/todo.md` still lists Notebook 01 rerun validation and the Python 3.9 compatibility issue as open work.
- `docs/governance-operations.md` documents `make research-audit*` commands, but `make research-audit` currently fails with `No rule to make target 'research-audit'`.
- The default dev environment does not currently include notebook execution packages (`nbformat`, `nbclient`, `jupyter` are missing in `.venv`), so notebook validation is not reproducible from the default install.

### Task 1: Restore Python 3.9-safe imports for `StrategyArena`

**Files:**
- Create: `tests/test_arena_python39_import.py`
- Modify: `research/backtest/arena.py`
- Test: `tests/test_arena_python39_import.py`
- Test: `tests/test_python39_annotation_compatibility.py`
- Test: `tests/test_strategy_arena.py`
- Test: `tests/test_hawkes_comparison.py`

**Step 1: Write the failing test**

```python
from importlib import import_module


def test_arena_module_imports_without_runtime_annotation_errors():
    module = import_module("research.backtest.arena")
    assert hasattr(module, "StrategyArena")
```

**Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_arena_python39_import.py
```

Expected: `ERROR` or `FAIL` with a `TypeError` complaining about `unsupported operand type(s) for |`.

**Step 3: Write minimal implementation**

Add postponed annotation evaluation at the top of `research/backtest/arena.py` and keep the existing type hints intact unless any additional import-time incompatibility remains.

```python
from __future__ import annotations

"""Strategy arena for fair backtest comparison across multiple strategies."""
```

If import still fails after that change, replace only the remaining problematic runtime unions with `typing.Union[...]` / `typing.Optional[...]` equivalents instead of broad refactors.

**Step 4: Run tests to verify the blocker is removed**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_arena_python39_import.py \
  tests/test_python39_annotation_compatibility.py \
  tests/test_strategy_arena.py \
  tests/test_hawkes_comparison.py
```

Expected: all four test files pass on Python 3.9.

**Step 5: Commit**

```bash
git add tests/test_arena_python39_import.py research/backtest/arena.py
git commit -m "fix: restore python39-safe arena imports"
```

### Task 2: Re-open the non-integration regression path

**Files:**
- Modify only if needed after the red/green run:
  `research/backtest/arena.py`
  `research/backtest/hawkes_comparison.py`
  `tests/test_strategy_arena.py`
  `tests/test_hawkes_comparison.py`
- Test: `tests/`

**Step 1: Use the existing suite as the failing test**

Run:

```bash
./.venv/bin/python -m pytest -q -m "not integration"
```

Expected after Task 1: the suite should get past collection. If a new assertion-level failure appears, treat that exact failing test as the red case and do not broaden the scope.

**Step 2: Apply the smallest production fix**

Only patch the concrete behavior exercised by the newly failing test. Do not refactor unrelated arena/reporting code while stabilizing the suite.

Likely touch points if a follow-on failure appears:

```python
# research/backtest/arena.py
# research/backtest/hawkes_comparison.py
# keep changes local to the failing path only
```

**Step 3: Re-run the specific failing test first**

Run the exact node that failed, for example:

```bash
./.venv/bin/python -m pytest -q tests/test_strategy_arena.py::test_scorecard_paths_and_report_generation
```

Expected: PASS.

**Step 4: Re-run the full non-integration suite**

Run:

```bash
./.venv/bin/python -m pytest -q -m "not integration"
```

Expected: PASS, with only intentional skips such as optional ML dependencies.

**Step 5: Commit**

```bash
git add research/backtest/arena.py research/backtest/hawkes_comparison.py tests
git commit -m "test: stabilize non-integration regression suite"
```

### Task 3: Add a reproducible Notebook 01 validation path

**Files:**
- Create: `scripts/backtest/validate_market_simulation_demo.py`
- Create: `tests/test_market_simulation_demo_validation.py`
- Modify: `Makefile`
- Modify: `tasks/todo.md`
- Optional manual refresh after script passes: `notebooks/01_market_simulation_demo.ipynb`

**Step 1: Write the failing test**

Create a deterministic smoke test around a callable validation function.

```python
from pathlib import Path

from scripts.backtest.validate_market_simulation_demo import run_validation


def test_run_validation_writes_metrics_and_plot(tmp_path: Path):
    report = run_validation(output_dir=tmp_path, days=2, seed=42)

    assert report["naive"]["trade_count"] > 0
    assert report["avellaneda_stoikov"]["trade_count"] > 0
    assert (tmp_path / "strategy_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
```

**Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_market_simulation_demo_validation.py
```

Expected: `ModuleNotFoundError` or failing assertions because the script/function does not exist yet.

**Step 3: Write minimal implementation**

Implement a script that recreates the notebook's core flow without requiring Jupyter:

- generate synthetic market data with `CompleteMarketSimulator(seed=42)`
- run both `NaiveMarketMaker` and `AvellanedaStoikov` through `BacktestEngine`
- fail fast if either strategy produces zero trades or empty `pnl_series`
- write:
  - `strategy_comparison.png`
  - `metrics.json`
  - optional `summary.md`

Suggested callable shape:

```python
def run_validation(output_dir: Path, days: int = 30, seed: int = 42) -> dict[str, dict[str, float]]:
    ...
```

Add a make target:

```make
notebook-01-validate:
	$(PYTHON) scripts/backtest/validate_market_simulation_demo.py \
		--output-dir artifacts/notebooks/01_market_simulation_demo
```

**Step 4: Run the new validation path**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_market_simulation_demo_validation.py
make notebook-01-validate
```

Expected:

- the pytest file passes
- `artifacts/notebooks/01_market_simulation_demo/strategy_comparison.png` exists
- `artifacts/notebooks/01_market_simulation_demo/metrics.json` exists
- both strategy summaries show non-zero trade counts

**Step 5: Refresh the notebook and close the TODO**

If notebook extras are available, install the minimal extra set and rerun the actual notebook once:

```bash
./.venv/bin/python -m pip install -e ".[dev,notebook]"
./.venv/bin/python -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/01_market_simulation_demo.ipynb \
  --output 01_market_simulation_demo.ipynb
```

Then update `tasks/todo.md` to mark Notebook 01 validation complete and record the artifact directory used for evidence.

**Step 6: Commit**

```bash
git add scripts/backtest/validate_market_simulation_demo.py tests/test_market_simulation_demo_validation.py Makefile tasks/todo.md notebooks/01_market_simulation_demo.ipynb
git commit -m "feat: add reproducible notebook 01 validation"
```

### Task 4: Align governance docs with runnable research-audit entrypoints

**Files:**
- Modify: `Makefile`
- Modify: `docs/governance-operations.md`
- Modify: `tests/test_make_governance_targets.py`

**Step 1: Write the failing tests**

Add dry-run assertions for the missing targets.

```python
def test_research_audit_target_is_available():
    stdout = _make_dry_run("research-audit")
    assert "validation_scripts/iv_surface_stability_report.py" in stdout
    assert "validation_scripts/research_audit_snapshot.py" in stdout


def test_research_audit_compare_target_is_available():
    stdout = _make_dry_run("research-audit-compare")
    assert "validation_scripts/research_audit_compare.py" in stdout
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py
```

Expected: FAIL because `make -n research-audit` currently exits non-zero.

**Step 3: Write minimal implementation**

Add Make targets that match the documentation and existing scripts:

```make
research-audit:
	$(PYTHON) validation_scripts/iv_surface_stability_report.py \
		--output-md artifacts/iv-surface-stability-report.md \
		--output-json artifacts/iv-surface-stability-report.json
	$(PYTHON) validation_scripts/pricing_model_zoo_benchmark.py \
		--seed 42 \
		--n-per-bucket 1 \
		--save-quotes-json artifacts/pricing-model-zoo-quotes.json \
		--output-json artifacts/pricing-model-zoo-benchmark.json \
		--output-md artifacts/pricing-model-zoo-benchmark.md \
		--strict
	$(PYTHON) validation_scripts/rough_jump_experiment.py --seed 42 > artifacts/rough-jump-experiment.txt
	$(PYTHON) validation_scripts/inverse_power_validation.py \
		--output-md artifacts/inverse-power-validation-report.md \
		--output-json artifacts/inverse-power-validation-report.json
	$(PYTHON) validation_scripts/research_audit_snapshot.py \
		--output-json artifacts/research-audit-snapshot.json
	$(MAKE) research-audit-compare
	$(PYTHON) validation_scripts/research_audit_weekly_summary.py \
		--output-md artifacts/research-audit-weekly-summary.md

research-audit-compare:
	$(PYTHON) validation_scripts/research_audit_compare.py \
		--baseline-json validation_scripts/fixtures/research_audit_snapshot_baseline.json \
		--current-json artifacts/research-audit-snapshot.json \
		--output-json artifacts/research-audit-drift-report.json \
		--output-md artifacts/research-audit-drift-report.md

research-audit-refresh-baseline:
	cp artifacts/research-audit-snapshot.json validation_scripts/fixtures/research_audit_snapshot_baseline.json
```

If product intent is to avoid new orchestration targets, shrink `docs/governance-operations.md` instead of adding targets, but choose one truth and make tests reflect it.

**Step 4: Run tests and docs checks**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py
make docs-link-check
make research-audit
```

Expected:

- Make target tests pass
- docs link check stays green
- research-audit artifacts are regenerated under `artifacts/`

**Step 5: Commit**

```bash
git add Makefile docs/governance-operations.md tests/test_make_governance_targets.py artifacts
git commit -m "build: align research audit docs and make targets"
```

### Task 5: Final closeout verification and release-readiness pass

**Files:**
- Modify only if verification uncovers drift:
  `README.md`
  `tasks/todo.md`
  `docs/governance-operations.md`
  `docs/plans/algorithm-freeze-checklist.md`

**Step 1: Run the full verification set**

Run:

```bash
./.venv/bin/python -m pytest -q -m "not integration"
make docs-link-check
make research-audit
make notebook-01-validate
make algorithm-freeze-check
```

Expected: all commands exit `0`.

**Step 2: Verify open-work markers are actually closed**

Check:

- `tasks/todo.md` has no remaining items tied to the Python 3.9 / Notebook 01 blockers.
- `weekly-signoff-pack` is still correctly marked manual if human signoff is intentionally pending.
- README / docs do not promise commands that still do not exist.

**Step 3: If `algorithm-freeze-check` fails, fix the exact gate only**

Do not reopen feature work. Only patch the failing verification path and rerun the same command until green.

**Step 4: Commit**

```bash
git add README.md tasks/todo.md docs
git commit -m "chore: close completion blockers and refresh verification docs"
```

## Completion Definition

Treat the project as effectively complete only when all of the following are true:

- `./.venv/bin/python -m pytest -q -m "not integration"` passes on Python 3.9
- `tests/test_hawkes_comparison.py` and `tests/test_strategy_arena.py` are green
- `make notebook-01-validate` produces fresh plot + metrics artifacts
- `tasks/todo.md` no longer lists the current known blockers
- `make research-audit` is either implemented and green, or the docs no longer promise it
- `make algorithm-freeze-check` passes end-to-end
