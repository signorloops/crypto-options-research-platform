# Governance Gate Activation Progress

Date: 2026-02-28  
Scope: weekly governance manual-closure execution + `master` branch required checks activation

## Completed In This Iteration

1. Ran full weekly governance chain locally:
   - `make weekly-operating-audit`
   - Generated/updated:
     - `artifacts/weekly-operating-audit.{md,json}`
     - `artifacts/weekly-pnl-attribution.{md,json}`
     - `artifacts/weekly-canary-checklist.{md,json}`
     - `artifacts/weekly-adr-draft.md`
     - `artifacts/weekly-decision-log.{md,json}`
     - `artifacts/weekly-signoff-pack.{md,json}`
     - `artifacts/weekly-manual-status.json` (auto-created template)
2. Confirmed sign-off pack gate status (initial run):
   - Current status: `PENDING_MANUAL_SIGNOFF` (expected before production/manual confirmations).
3. Activated branch protection required checks on `master` via GitHub API:
   - `complexity-audit`
   - `security`
   - `test (3.9)`
   - `test (3.10)`
   - `test (3.11)`
   - `docker`
   - `daily-regression-gate`
4. Updated roadmap wording to remove stale note that required checks were still pending repository settings.
5. Re-ran daily regression gate locally:
   - `make daily-regression`
   - Result: all commands passed.
6. Added objective prefill step for manual closure:
   - Added `scripts/governance/weekly_manual_status_prefill.py` + tests.
   - Added Make target `weekly-manual-prefill` and chained it into `make weekly-operating-audit`.
   - Auto-prefill now sets:
     - `rollback_decision_recorded` when decision evidence exists.
     - `change_and_rollback_recorded` when rollback reference is present.
     - `pnl_attribution_confirmed` only when attribution has zero missing entries.
7. Reduced pending-item noise in sign-off aggregation:
   - `weekly_signoff_pack` now normalizes alias labels (e.g. `收益归因表` -> `收益归因表确认`)
   - skips already-completed mapped manual tasks from follow-up/incomplete lists.
   - Current pending count reduced from `10` to `7` (same gate status, clearer queue).
8. Added W3 online/offline consistency replay artifact:
   - Added `scripts/governance/online_offline_consistency_replay.py` + tests.
   - Added threshold config `config/online_offline_consistency_thresholds.json`.
   - Added Make target `weekly-consistency-replay` and chained it into `make weekly-operating-audit`.
   - Current replay status is `FAIL` on live snapshot (`online_alerts>0`, `online_max_abs_deviation_bps>300.0`), giving actionable W3 evidence instead of placeholder text.
9. Linked replay result into weekly sign-off gating:
   - `weekly_signoff_pack` now consumes `artifacts/online-offline-consistency-replay.json`.
   - `FAIL` replay status now triggers auto blocker `online_offline_replay_status=FAIL`.
   - `PENDING_DATA`/missing replay status now adds pending item `线上/线下一致性回放数据待联调` (without auto blocking).
   - Updated `Makefile` order so replay runs before sign-off pack in `weekly-operating-audit`.
   - Current sign-off status is now `AUTO_BLOCKED` due replay `FAIL`.
10. Added one-command strict close gate:
   - Added Make target `weekly-close-gate`.
   - Target runs full weekly chain then validates close gate via:
     - `python scripts/governance/weekly_operating_audit.py --close-gate-only --strict-close --signoff-json artifacts/weekly-signoff-pack.json`
   - On each run, emits close gate summary artifacts:
     - `artifacts/weekly-close-gate.md`
     - `artifacts/weekly-close-gate.json`
   - Summary artifact now includes PR-ready brief text (`PR Brief (Copy/Paste)` / `pr_brief`) for direct status updates.
   - Current execution returns non-zero as expected when sign-off is not `READY_FOR_CLOSE`.

## Remaining Manual Tasks (By Design)

1. 灰度发布并完成 24h 观察。  
2. 在 `artifacts/weekly-manual-status.json` 填写人工确认与签字。  
3. 完成收益归因确认、ADR 签字、回滚决策记录。  
4. 将本周线上证据补齐后重跑签字包并执行严格门禁，确认 `READY_FOR_CLOSE`：
   - `make weekly-close-gate`

## Verification Executed

```bash
make weekly-operating-audit
make daily-regression
/opt/homebrew/bin/python3.13 -m pytest -q tests/test_weekly_signoff_pack.py tests/test_weekly_manual_status_prefill.py tests/test_online_offline_consistency_replay.py
make weekly-manual-prefill
make weekly-consistency-replay
make weekly-operating-audit
python3 scripts/governance/weekly_signoff_pack.py --strict
make weekly-close-gate
gh api repos/signorloops/crypto-options-research-platform/branches/master/protection
gh run view 22407908033 --repo signorloops/crypto-options-research-platform --json jobs
gh run view 22512827310 --repo signorloops/crypto-options-research-platform --json jobs
gh api --method PATCH repos/signorloops/crypto-options-research-platform/branches/master/protection/required_status_checks \
  -F strict=true \
  -f 'contexts[]=complexity-audit' \
  -f 'contexts[]=security' \
  -f 'contexts[]=test (3.9)' \
  -f 'contexts[]=test (3.10)' \
  -f 'contexts[]=test (3.11)' \
  -f 'contexts[]=docker' \
  -f 'contexts[]=daily-regression-gate'
```

Note:
- Strict sign-off gate currently returns exit code `2` (`AUTO_BLOCKED`) because replay status is `FAIL` and because manual confirmations are not yet fully complete.
