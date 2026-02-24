# Day10 Regression Acceptance Report

Date: 2026-02-24
Scope: Coin-margined options roadmap P0/P1 increment + governance baseline checks.

## Commands Executed

1. Core regression bundle (roadmap Day10):
```bash
./venv/bin/pytest --disable-warnings \
  tests/test_pricing_inverse.py \
  tests/test_volatility.py \
  tests/test_hawkes_comparison.py \
  tests/test_research_dashboard.py
```
Result: `91 passed, 4 warnings in 5.99s`

2. New module bundle:
```bash
./venv/bin/pytest --disable-warnings \
  tests/test_jump_risk_premia.py \
  tests/test_pricing_model_zoo.py \
  tests/test_rough_volatility.py
```
Result: `9 passed, 1 warning in 0.94s`

3. Full suite sanity:
```bash
./venv/bin/pytest -q
```
Result: collection blocked (18 errors), mainly environment/dependency and Python 3.8 typing-compat issues.

## Acceptance Decision

- P0/P1 target scope: PASS
- Full repository suite: NOT PASS (environment baseline not yet aligned)

## Blocking Findings (Full Suite)

1. Python 3.8 typing incompatibility in some modules (example: `tuple[...]` style annotations in risk module).
2. Missing optional runtime dependencies in current venv (`duckdb`, `redis`, `hmmlearn`).
3. Async marker/plugin mismatch in pytest config (`asyncio` marker not registered in current env).

## Immediate Remediation Plan

1. Align CI/local baseline to Python 3.11 and update typed annotations to 3.8-safe only if 3.8 must remain supported.
2. Split test matrix into:
   - core required checks (always-on)
   - optional integration checks gated by extras/dependencies.
3. Add dependency groups (`[test-core]`, `[test-integration]`) and document one-command setup.

## Notes

- During regression preparation, a Python 3.8 compatibility fix was applied in `data/streaming.py` for `asyncio.Task[...]` annotations.
- Dashboard test dependencies were installed locally for this validation (`fastapi`, `starlette`, `httpx`, `plotly`).
