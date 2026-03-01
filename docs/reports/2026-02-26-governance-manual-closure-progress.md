# Governance Manual-Closure Progress

Date: 2026-02-26  
Scope: `docs/plans/weekly-operating-checklist.md` + weekly governance automation scripts

## Completed In This Iteration

1. Added manual-closure aggregation script:
   - `scripts/governance/weekly_signoff_pack.py`
   - Aggregates audit/canary/decision/attribution outputs into one sign-off package.
   - Produces `AUTO_BLOCKED` / `PENDING_MANUAL_SIGNOFF` / `READY_FOR_CLOSE` statuses.
   - Supports `--strict` mode for release gate integration.
   - Auto-creates `artifacts/weekly-manual-status.json` template if missing.
2. Expanded Make entrypoints and orchestration:
   - Added `daily-regression`, `weekly-pnl-attribution`, `weekly-canary-checklist`, `weekly-decision-log`, `weekly-signoff-pack`.
   - `make weekly-operating-audit` now chains the full governance artifact pipeline.
3. Hardened weekly regression command execution:
   - `scripts/governance/weekly_operating_audit.py` now parses `--regression-cmd` via `shlex.split`.
   - Removed `shell=True` execution path.
4. Added/extended regression tests:
   - `tests/test_weekly_signoff_pack.py`
   - `tests/test_weekly_operating_audit.py` (regression command execution behavior)
5. Updated weekly checklist outputs:
   - `docs/plans/weekly-operating-checklist.md` now includes all generated governance artifacts and manual status template file.
6. Added consistency gate for weekly audit:
   - `scripts/governance/weekly_operating_audit.py` now supports:
     - recursive result discovery (`rglob`)
     - `--consistency-thresholds` (`config/consistency_thresholds.json`)
     - latest-vs-previous snapshot diffs (`abs_pnl_diff`, `abs_sharpe_diff`, `abs_max_drawdown_diff`)
     - strict non-zero exit on consistency exceptions
7. Extended weekly CI workflow from single-report to full governance chain:
   - `.github/workflows/weekly-operating-audit.yml`
   - now emits weekly operating audit + attribution + canary checklist + ADR draft + decision log + sign-off pack
   - uploads all generated governance artifacts for review/audit traceability
8. Prepared W4 branch-protection gating handoff:
   - `.github/workflows/daily-regression.yml` now triggers on all PRs (not path-filtered), so `daily-regression-gate` can be configured as required check.
   - updated `docs/branch-protection-checklist.md` with required-check list and verification commands.
9. Landed project slimming controls (storage + dependencies):
   - Added `scripts/maintenance/workspace_slimmer.py` and `tests/test_workspace_slimmer.py`.
   - Added Make targets: `workspace-slim-report`, `workspace-slim-clean`, `workspace-slim-clean-venv`.
   - Dependency profiles split in `pyproject.toml`:
     - core runtime remains default
     - optional extras: `accelerated`, `ml`, `notebook`, `full`
   - Updated install docs (`README.md`, `docs/quickstart.md`, `docs/GUIDE.md`) to reflect editable extras profiles.
   - Updated `.gitignore` to block generated governance/research outputs and cleaned accidental artifact lines.

## Remaining Manual Tasks (By Design)

1. Fill `artifacts/weekly-manual-status.json` with real-world manual confirmations/signatures.
2. Execute gray release and 24h observation in production.
3. Confirm PnL attribution口径 with production week data.
4. Complete final ADR sign-off and rollback decision sign-off.

## Verification Executed

```bash
venv/bin/python -m pytest -q tests/test_weekly_signoff_pack.py tests/test_weekly_operating_audit.py tests/test_weekly_canary_checklist.py tests/test_weekly_decision_log.py tests/test_weekly_pnl_attribution.py tests/test_weekly_adr_draft.py
venv/bin/python -m pytest -q tests/test_workspace_slimmer.py
make -n weekly-operating-audit
make -n weekly-signoff-pack
make help | rg "daily-regression|weekly-pnl-attribution|weekly-canary-checklist|weekly-decision-log|weekly-signoff-pack"
venv/bin/python -m pytest -q tests/test_weekly_operating_audit.py
make workspace-slim-report
```
