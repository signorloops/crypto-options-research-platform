# Complexity Governance

## Goal

Keep the platform extensible without turning core modules into unmaintainable monoliths.
Governance is enforced by CI gate checks plus weekly audits and explicit complexity budgets.

---

## Scope

Governed code scope (default):

- `core/`
- `data/`
- `research/`
- `strategies/`
- `execution/`
- `utils/`

Excluded from budget:

- `tests/`
- `notebooks/`
- `validation_scripts/`
- `docs/`
- `venv/`

Rationale: production complexity and runtime coupling are the primary risk.

---

## Budget Definition

Complexity budget lives in:

- [config/complexity_budget.json](../config/complexity_budget.json)

Current key limits:

1. `max_python_files`
2. `max_total_loc`
3. `max_avg_loc_per_file`
4. `max_file_loc`
5. `max_files_over_soft_loc` + `soft_file_loc`
6. `max_function_loc`
7. `max_functions_over_soft_loc` + `soft_function_loc`
8. `max_function_args`
9. `max_methods_per_class`
10. `max_classes_over_method_soft_limit` + `soft_method_count_per_class`

Budgets are intended to be tightened gradually, not relaxed casually.

---

## Automation

Weekly CI workflow:

- [complexity-governance.yml](../.github/workflows/complexity-governance.yml)
- [ci.yml](../.github/workflows/ci.yml)（changed-files lint/type 增量门禁）

Checker script:

- [complexity_guard.py](../scripts/governance/complexity_guard.py)

Workflow behavior:

1. Runs on `pull_request` and `push` for governed production-code paths.
2. Runs every week on Monday (UTC) and on manual dispatch.
3. Generates Markdown + JSON reports.
4. Uploads reports as workflow artifacts.
5. Fails workflow when threshold violations exist.
6. CI 增量门禁仅对 changed files 执行更严格 lint/type 检查，作为逐步收紧策略。

---

## Local Run

```bash
cd crypto-options-research-platform
make complexity-audit
```

Regression-only strict mode (fail only on new/worsened violations vs baseline):

```bash
cd crypto-options-research-platform
make complexity-audit-regression BASELINE_COMPLEXITY_JSON=artifacts/complexity-baseline.json
```

Outputs:

- `artifacts/complexity-governance-report.md`
- `artifacts/complexity-governance-report.json`

Baseline file:

- Reuse any historical `complexity-governance-report.json` as `--baseline-json`.
- `--strict-regression-only` keeps existing debt visible while blocking new or worsened debt.
- Repository baseline snapshot: [config/complexity_baseline.json](../config/complexity_baseline.json)

---

## Violation Policy

When weekly check fails:

1. Triage top offenders from report (`top_files_by_loc`, `top_functions_by_loc`, `top_classes_by_methods`).
2. Pick one remediation path:
   - Extract cohesive submodules.
   - Split oversized functions into pure helpers.
   - Move experimental logic behind explicit flags.
   - Reduce constructor/utility argument fan-out via typed config objects.
3. If temporary waiver is necessary:
   - Create a dated note in `docs/plans/`.
   - Include owner, reason, rollback date.
   - Update budget only with explicit justification.

No silent threshold bumps.

---

## Review Cadence

1. Weekly: automated report review.
2. Bi-weekly: tighten at least one soft threshold when the codebase is stable.
3. Before major releases: run strict complexity check as release gate.
