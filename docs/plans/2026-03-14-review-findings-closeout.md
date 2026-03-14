# Review Findings Closeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the four project-level review findings by hardening manual sign-offs, making the release-candidate gate recompute current evidence, and restoring real backtest metrics for fill rate and spread capture.

**Architecture:** The fixes stay local to the existing governance and backtest pipelines. Governance changes should make placeholder approvals impossible and ensure the `release-candidate-check` entrypoint regenerates the evidence it relies on. Backtest changes should compute metrics at the source in `BacktestEngine`, then propagate them into `StrategyArena` and governance reports without inventing placeholder values.

**Tech Stack:** Python 3.9+, pytest, GNU Make, existing governance scripts, pandas, numpy.

---

### Task 1: Reject placeholder sign-offs in governance flow

**Files:**
- Modify: `scripts/governance/manual_status_utils.py`
- Modify: `scripts/governance/weekly_manual_status_update.py`
- Modify: `scripts/governance/weekly_signoff_pack.py`
- Modify: `tests/test_weekly_manual_status_update.py`
- Modify: `tests/test_weekly_signoff_pack.py`

**Step 1: Write the failing tests**

- Add a test that `weekly_manual_status_update` rejects `research_owner`.
- Add a test that `_build_report()` does not treat placeholder signers as complete sign-offs.

**Step 2: Run tests to verify they fail**

```bash
./.venv/bin/python -m pytest -q tests/test_weekly_manual_status_update.py tests/test_weekly_signoff_pack.py
```

**Step 3: Write minimal implementation**

- Centralize placeholder detection in `manual_status_utils.py`
- Reject reserved placeholder names in `_parse_signoff_assignment()`
- Treat placeholder signers as incomplete in `weekly_signoff_pack.py`

**Step 4: Run tests to verify they pass**

```bash
./.venv/bin/python -m pytest -q tests/test_weekly_manual_status_update.py tests/test_weekly_signoff_pack.py
```

### Task 2: Make `release-candidate-check` recompute current evidence

**Files:**
- Modify: `Makefile`
- Modify: `tests/test_make_governance_targets.py`

**Step 1: Write the failing test**

- Extend the Make dry-run tests so `release-candidate-check` must call `algorithm-freeze-check` and `weekly-close-gate` before `release_candidate_guard.py`.

**Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py
```

**Step 3: Write minimal implementation**

- Change `release-candidate-check` in `Makefile` to run:
  1. `make algorithm-freeze-check`
  2. `make weekly-close-gate`
  3. `release_candidate_guard.py`

**Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest -q tests/test_make_governance_targets.py tests/test_release_candidate_guard.py
```

### Task 3: Compute real spread-capture and quote-count metrics in backtest engine

**Files:**
- Modify: `research/backtest/fill_model.py`
- Modify: `research/backtest/engine.py`
- Modify: `tests/test_strategy_arena.py`
- Create or modify: engine/fill-model tests covering spread capture output

**Step 1: Write the failing tests**

- Add a test that `BacktestEngine` result fields expose non-zero `total_spread_captured` / `avg_spread_captured_bps` when fills occur at quoted spreads.
- Add a test that result payload carries quote count needed for fill-rate calculation.

**Step 2: Run tests to verify they fail**

```bash
./.venv/bin/python -m pytest -q tests/test_strategy_arena.py tests/test_backtest_engine.py
```

**Step 3: Write minimal implementation**

- Track spread-capture totals in `RealisticFillSimulator`
- Track quote count in `BacktestEngine` results
- Populate `total_spread_captured` / `avg_spread_captured_bps` from actual fill activity instead of constants

**Step 4: Run tests to verify they pass**

```bash
./.venv/bin/python -m pytest -q tests/test_strategy_arena.py tests/test_backtest_engine.py
```

### Task 4: Remove synthetic metrics from `StrategyArena` and re-verify gates

**Files:**
- Modify: `research/backtest/arena.py`
- Modify: `tests/test_strategy_arena.py`
- Modify: `tests/test_weekly_pnl_attribution.py` if attribution output changes

**Step 1: Write the failing test**

- Add a test that scorecards derive `fill_rate` from result data rather than returning a hardcoded constant.

**Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest -q tests/test_strategy_arena.py
```

**Step 3: Write minimal implementation**

- Replace the hardcoded `0.3` fill rate with a computed value from the backtest result

**Step 4: Run focused tests and then full verification**

```bash
./.venv/bin/python -m pytest -q \
  tests/test_weekly_manual_status_update.py \
  tests/test_weekly_signoff_pack.py \
  tests/test_make_governance_targets.py \
  tests/test_release_candidate_guard.py \
  tests/test_strategy_arena.py \
  tests/test_weekly_pnl_attribution.py

make release-candidate-check
make algorithm-freeze-check
```
