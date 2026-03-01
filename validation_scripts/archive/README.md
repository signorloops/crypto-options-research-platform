# validation_scripts archive

This directory keeps only archive notes.
Legacy one-off scripts were removed in the 2026-03 cleanup pass because they
were not wired to active workflows, Make targets, tests, or runtime paths.

Historical outputs are preserved in:
- `docs/archive/reports/inverse-pnl-validation-report-2026-02-08.md`
- `docs/archive/reports/second-round-math-audit-report.md`

If a historical check needs to be reintroduced, create a new script under
`validation_scripts/`, then wire it into tests/docs/workflows explicitly.
