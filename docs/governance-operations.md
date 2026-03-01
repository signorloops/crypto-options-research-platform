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
make complexity-audit-regression BASELINE_COMPLEXITY_JSON=artifacts/complexity-baseline.json
```

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
