# 2026 Q1 数学验证与脚本退役摘要

本摘要整合了 2026 Q1 的数学验证历史产物与脚本退役说明，减少归档文档分散维护成本。

## 合并来源

1. `inverse-pnl-validation-report-2026-02-08.md`
2. `second-round-math-audit-report.md`
3. `validation-scripts-retirement-note.md`

## 关键结论

- 早期 inverse PnL 验证用于确认基础公式行为与数值边界。
- 第二轮数学审核用于记录公式核对、量纲检查与风险影响评估。
- 历史 one-off 验证脚本在 2026-03 被下线，原因是未接入活跃工作流、测试或运行路径。

## 当前策略

1. 历史结论保留在本摘要与 Git 历史中。
2. 需要重新启用验证时，应在 `validation_scripts/` 下新建脚本并接入 tests/docs/workflows。
