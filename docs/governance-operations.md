# Governance Operations Handbook

本手册整合研究审计、复杂度治理与工作区瘦身流程，作为统一入口。

## 1. Research Audit

目标：将研究脚本输出转成可追踪、可守门产物，持续验证 IV 稳定性、jump premia、model-zoo 与基线漂移。

常用命令：

```bash
make research-audit
make research-audit-compare
make research-audit-refresh-baseline
```

主要产物：

- `artifacts/iv-surface-stability-report.*`
- `artifacts/pricing-model-zoo-benchmark.*`
- `artifacts/research-audit-snapshot.json`
- `artifacts/research-audit-drift-report.*`
- `artifacts/research-audit-weekly-summary.md`

关键基线文件：

- `validation_scripts/fixtures/model_zoo_quotes_seed42.json`
- `validation_scripts/fixtures/research_audit_snapshot_baseline.json`

## 2. Complexity Governance

目标：通过预算阈值与 CI 守门，控制生产代码复杂度增长。

配置文件：

- `config/complexity_budget.json`
- `config/complexity_baseline.json`

常用命令：

```bash
make complexity-audit
make complexity-audit-refresh-baseline
make complexity-audit-regression BASELINE_COMPLEXITY_JSON=artifacts/complexity-baseline.json
```

刷新流程：

1. 先运行 `make complexity-audit` 生成最新报告。
2. 审阅 `artifacts/complexity-governance-report.json` / `.md` 中的预算变化。
3. 仅在确认当前复杂度状态应当成为新的治理基线后，再执行 `make complexity-audit-refresh-baseline`。

## 3. Workspace Slimming

目标：清理缓存、产物与本地大文件，降低仓库工作区噪音。

常用命令：

```bash
make workspace-slim-report
make workspace-slim-clean
make workspace-slim-clean-venv
```

默认安装建议：

```bash
pip install -e ".[dev]"
```

## 4. 日常最小流程

```bash
pytest -q -m "not integration"
make docs-link-check
make complexity-audit
```

## 5. 算法冻结门禁

目标：当算法链路满足发布条件时，使用统一门禁判定“停止继续优化”。

```bash
make algorithm-freeze-check
```

详细冻结标准见：

- `docs/plans/algorithm-freeze-checklist.md`

## 6. Release Candidate Closeout

目标：在算法冻结通过后，补齐人工确认项和角色签字，并生成最终的 release-candidate 证据。

常用命令：

```bash
make weekly-manual-update MANUAL_ARGS="--check gray_release_completed=true --check observation_24h_completed=true --check adr_signed=true --signoff research=<name> --signoff engineering=<name> --signoff risk=<name>"
make weekly-signoff-pack
make release-candidate-check
```

注意：

- `signoff` 的值必须是真实责任人姓名，不能使用占位符。
