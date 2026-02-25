# Roadmap Progress And Gaps

Date: 2026-02-25  
Scope: `docs/plans/2026-02-24-coin-margined-options-research-roadmap.md` and `docs/plans/2026-Q2-long-term-execution-roadmap.md`

## Completed In This Iteration

1. Added weekly operating audit automation:
   - `scripts/governance/weekly_operating_audit.py`
   - `config/weekly_operating_thresholds.json`
   - `.github/workflows/weekly-operating-audit.yml`
   - Missing experiment IDs now auto-filled as `AUTO-<result_file_stem>`
   - Supports optional in-process minimal regression command and records pass/fail in report
   - Auto-includes recent git change log and latest rollback baseline tag (if available)
   - Generates `weekly-adr-draft.md` for decision记录的初稿
2. Added local execution entrypoint:
   - `make weekly-operating-audit`
3. Added governance script regression tests:
   - `tests/test_complexity_guard.py`
   - `tests/test_weekly_operating_audit.py`
4. Fixed pandas forward/backward fill deprecation warning:
   - `data/quote_integration.py`
5. Synced plan docs with completion snapshot and outstanding manual tasks.

## Remaining Tasks (Not Fully Automated)

1. 灰度发布与 24h 观察。  
2. 收益归因表沉淀。  
3. ADR 决策沉淀与回滚决策记录。  
4. W2-W4 观测一致性里程碑（阈值联调、线上/离线一致性、日回归+周回放门禁）。

## Verification Executed

```bash
./venv/bin/pytest -q tests/test_complexity_guard.py tests/test_weekly_operating_audit.py
./venv/bin/pytest -q tests/test_pricing_inverse.py tests/test_volatility.py tests/test_jump_risk_premia.py tests/test_pricing_model_zoo.py tests/test_rough_volatility.py tests/test_research_dashboard.py tests/test_quanto_inverse.py
pytest -q
./venv/bin/python scripts/governance/complexity_guard.py --strict
./venv/bin/python scripts/governance/weekly_operating_audit.py --strict
```

Notes:

1. `pytest -q` (Python 3.13 baseline) passes full collection and execution.  
2. Legacy `./venv` (Python 3.8) still lacks several optional deps and typing compatibility for full-suite collection.
