# arXiv 前沿映射摘要（2026-02-25）

## 结论

当时优先级最高的三条落地方向：

1. Inverse/Quanto-Inverse-Power 定价扩展
2. Jump premia 分解与稳定性监控
3. Rough-vol + jumps 的快速校准路径

## 映射到仓库

- 定价层：`research/pricing/*`
- 信号层：`research/signals/jump_risk_premia.py`
- 审计层：`validation_scripts/*` 与 `Research Audit` 工作流

## 执行优先级（历史）

1. 先落地 inverse-power 与 quanto 扩展。
2. 再补 jump premia 分层指标。
3. 最后推进联合校准/深度无套利实验分支。

## 备注

该文档为历史决策摘要，详细论文条目与逐段分析已并入 Git 历史。
