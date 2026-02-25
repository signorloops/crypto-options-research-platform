# Research Audit Guide

## 目标

`Research Audit` workflow 将研究脚本输出转成可追踪、可守门的标准产物，用于持续验证：

1. 近月 IV surface 稳定性与静态无套利
2. rough-vol + jumps 实验表现
3. model-zoo 基准排名与误差稳定性
4. 与基线快照的漂移对比

## 主要产物

运行后 artifacts 包含：

1. `iv-surface-stability-report.md`
2. `iv-surface-stability-report.json`
3. `rough-jump-experiment.txt`
4. `pricing-model-zoo-benchmark.txt`
5. `pricing-model-zoo-benchmark.json`
6. `pricing-model-zoo-benchmark.md`
7. `research-audit-snapshot.json`
8. `research-audit-drift-report.md`
9. `research-audit-drift-report.json`
10. `research-audit-weekly-summary.md`

## 质量门槛（默认）

1. IV 报告门槛：
- `fail_on_arbitrage=true`
- `min_short_max_jump_reduction=0.005`

2. model-zoo 门槛：
- `expected_best_model=bates`
- `max_best_rmse=120.0`

3. 漂移门槛（对比基线）：
- `max_best_rmse_increase_pct=25.0`
- `max_iv_reduction_drop_pct=30.0`
- `allow_best_model_change=false`

## 本地执行

```bash
# 生成全套研究审计产物并执行守门
make research-audit

# 仅做当前快照 vs 基线对比
make research-audit-compare

# 若确认模型升级是预期行为，可刷新基线
make research-audit-refresh-baseline
```

## 基线文件

1. model-zoo 固定样本：
- `validation_scripts/fixtures/model_zoo_quotes_seed42.json`

2. 研究审计基线快照：
- `validation_scripts/fixtures/research_audit_snapshot_baseline.json`

## 基线更新流程（建议）

1. 手动触发 `Research Audit Baseline Refresh` workflow。
2. 下载 artifact `proposed-research-audit-baseline`。
3. 审阅：
- `proposed-research-audit-snapshot-baseline.json`
- `proposed-baseline-diff.md`
4. 若确认变更是预期升级，再提交基线文件更新。

## 调参建议

1. 若频繁误报：
- 先看 `research-audit-drift-report.md` 中具体失败项。
- 仅放宽对应阈值，不要一次性整体放宽。

2. 若模型升级导致“预期变化”：
- 先在本地确认结果合理（误差下降、无套利通过）。
- 再执行 `make research-audit-refresh-baseline` 并提交基线更新。

3. 若 CI 波动导致偶发失败：
- 优先检查随机种子、输入样本是否固定。
- 保持 `quotes_json` 固定输入，不建议切回纯随机样本守门。
