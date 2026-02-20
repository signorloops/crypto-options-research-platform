# Algorithm Upgrades Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement six algorithmic upgrades across fill simulation, regime detection, adaptive calibration, risk revaluation, volatility surface modeling, and Hawkes strategy control; then update algorithm documentation.

**Architecture:** The implementation extends existing modules in-place with backward-compatible defaults. New behavior is enabled through additional config fields and optional method paths so current strategy wiring and tests keep working while enabling advanced modes. Each upgrade gets focused unit tests before implementation changes.

**Tech Stack:** Python 3.9+, numpy, pandas, scipy, hmmlearn, pytest.

---

### Task 1: Conditional Fill Probability Model

**Files:**
- Modify: `research/backtest/engine.py`
- Test: `tests/test_backtest.py`

**Step 1: Write failing tests**
- Add tests asserting:
1) deeper queue lowers fill probability
2) more favorable quote competitiveness increases fill probability
3) elevated volatility lowers fill probability

**Step 2: Run targeted test**
- Run: `pytest -q tests/test_backtest.py::TestRealisticFillSimulator`
- Expected: FAIL on missing probability behavior

**Step 3: Minimal implementation**
- Add logistic-style `fill_probability` function with features:
1) queue depth
2) quote competitiveness
3) latency penalty
4) orderbook imbalance
5) realized short-horizon vol proxy from `recent_trades`
- Use computed probability in `_create_fill`.

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_backtest.py::TestRealisticFillSimulator`
- Expected: PASS

---

### Task 2: Sticky HMM Regime Detector With Variance-Based State Mapping

**Files:**
- Modify: `research/signals/regime_detector.py`
- Test: `tests/test_regime_detector.py`

**Step 1: Write failing tests**
- Add tests asserting:
1) state ordering uses variance/covariance risk metric (not mean return)
2) sticky regime logic prevents rapid switching on low-confidence changes

**Step 2: Run targeted test**
- Run: `pytest -q tests/test_regime_detector.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- Add config:
1) `regime_persistence_min_samples`
2) `switch_hysteresis`
3) `min_confidence_for_switch`
- Compute per-state risk score from covariance matrix diagonal/traces.
- Apply sticky switch filter in `update`.

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_regime_detector.py`
- Expected: PASS

---

### Task 3: Online Parameter Calibration for AS and Integrated Strategy

**Files:**
- Modify: `strategies/market_making/avellaneda_stoikov.py`
- Modify: `strategies/market_making/integrated_strategy.py`
- Test: `tests/test_strategies.py`
- Test: `tests/test_integrated_strategy.py`

**Step 1: Write failing tests**
- Add tests asserting:
1) AS strategy updates `sigma`/`k` from rolling market features
2) Integrated strategy updates effective sigma/skew with bounded ranges

**Step 2: Run targeted tests**
- Run: `pytest -q tests/test_strategies.py tests/test_integrated_strategy.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- Add lightweight online calibrator logic:
1) rolling volatility to sigma
2) trade arrival intensity to k
3) inventory utilization to skew factor
- Keep defaults backward-compatible and clamped.

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_strategies.py tests/test_integrated_strategy.py`
- Expected: PASS

---

### Task 4: Full Revaluation Path in Monte Carlo VaR

**Files:**
- Modify: `research/risk/var.py`
- Test: `tests/test_risk.py`

**Step 1: Write failing tests**
- Add tests asserting:
1) MC VaR supports full revaluation when option contract fields are available
2) result remains finite and method remains `monte_carlo`

**Step 2: Run targeted tests**
- Run: `pytest -q tests/test_risk.py::TestVaRCalculator`
- Expected: FAIL

**Step 3: Minimal implementation**
- Add optional revaluation branch in `monte_carlo_var`:
1) reconstruct shocked spot and vol
2) call `InverseOptionPricer.calculate_price`
3) compute pathwise repriced PnL instead of Greeks approximation
- Keep existing Greeks path as fallback.

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_risk.py::TestVaRCalculator`
- Expected: PASS

---

### Task 5: Global SSVI-Like Arbitrage-Aware Surface Mode

**Files:**
- Modify: `research/volatility/implied.py`
- Test: `tests/test_volatility.py`

**Step 1: Write failing tests**
- Add tests asserting:
1) new `ssvi` interpolation mode returns bounded vols
2) total variance across maturities is non-decreasing at fixed moneyness

**Step 2: Run targeted tests**
- Run: `pytest -q tests/test_volatility.py::TestVolatilitySurface`
- Expected: FAIL

**Step 3: Minimal implementation**
- Add global SSVI parameter fit/cache:
1) fit ATM total variance term-structure
2) fit global rho/eta/lambda with stability constraints
3) add `method="ssvi"` path in `get_volatility`

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_volatility.py::TestVolatilitySurface`
- Expected: PASS

---

### Task 6: Marked Bid/Ask Hawkes + Online MLE Control Signals

**Files:**
- Modify: `strategies/market_making/hawkes_mm.py`
- Test: `tests/test_hawkes_comparison.py` (add focused unit tests)

**Step 1: Write failing tests**
- Add tests asserting:
1) marked trade size affects buy/sell intensity asymmetry
2) online MLE update returns stable parameter estimates under synthetic flow
3) quote metadata exposes control signals used for spread/skew

**Step 2: Run targeted tests**
- Run: `pytest -q tests/test_hawkes_comparison.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- Extend monitor with marked event updates:
1) size-weighted excitation
2) buy/sell-specific intensity state
- Add optional MLE-based parameter update using scipy optimize fallback to moments.
- Add explicit control signal computation used in quote decisions.

**Step 4: Re-run tests**
- Run: `pytest -q tests/test_hawkes_comparison.py`
- Expected: PASS

---

### Task 7: Algorithm Documentation Update

**Files:**
- Modify: `ALGORITHMS.md`
- Modify: `docs/theory.md`
- Modify: `docs/architecture.md`

**Step 1: Write/adjust docs**
- Document for each upgrade:
1) math intuition
2) config knobs
3) runtime tradeoffs
4) validation metrics

**Step 2: Verify docs references**
- Run: `rg -n "fill probability|sticky|ssvi|full revaluation|marked hawkes" ALGORITHMS.md docs/theory.md docs/architecture.md`
- Expected: each concept appears in docs.

---

### Task 8: Final Verification

**Files:**
- Verify only

**Step 1: Run targeted suites**
- Run: `pytest -q tests/test_backtest.py tests/test_regime_detector.py tests/test_strategies.py tests/test_integrated_strategy.py tests/test_risk.py tests/test_volatility.py tests/test_hawkes_comparison.py`

**Step 2: Run full suite**
- Run: `pytest -q`

**Step 3: Sanity quality**
- Run: `make test`

