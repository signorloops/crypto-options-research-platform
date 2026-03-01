# Implementation Plan: arXiv-Driven Inverse Options Upgrades (2026-02-25)

## Scope

将 `docs/reports/2026-02-25-arxiv-frontier-inverse-options.md` 中的高 ROI 方向，拆解为可逐步交付的工程任务。

## Phase 1 (P0): Inverse-Power / Quanto-Inverse-Power Pricing

### Goal

在现有 `research/pricing/inverse_options.py` 基础上，新增 inverse-power 产品族定价与 Greeks 能力，保持与当前 API 风格一致。

### Deliverables

1. 新增模块：`research/pricing/inverse_power_options.py`
2. 新增类型：`InversePowerQuote`, `InversePowerGreeks`
3. 新增函数：
- `calculate_price(...)`
- `calculate_greeks(...)`
- `calculate_price_and_greeks(...)`
4. 验证脚本：`validation_scripts/inverse_power_validation.py`
5. 测试：`tests/test_pricing_inverse_power.py`

### Acceptance Criteria

1. 与退化情形一致：`power=1` 时结果与现有 inverse pricing 在容差内一致。
2. Greeks 数值稳定：与有限差分结果偏差在阈值内。
3. 边界条件通过：`T->0`、深度 ITM/OTM、高波动下不产生 NaN/Inf。

## Phase 2 (P1): Jump Premia Signal Decomposition

### Goal

把 clustered jumps 相关因子纳入现有 jump signal 管线，提升策略解释力与可监控性。

### Deliverables

1. 扩展：`research/signals/jump_risk_premia.py`
- 正/负跳跃风险溢价分解
- regime/cluster 状态分层
2. 报告脚本：`validation_scripts/jump_premia_stability_report.py`
3. 测试：`tests/test_jump_premia_stability_report.py`

### Acceptance Criteria

1. 输出包含可追踪的时序指标（正跳、负跳、净溢价）。
2. 对固定样本运行结果可复现（固定 seed + fixture）。
3. 可接入 `Research Audit` workflow 作为可选门槛。

## Phase 3 (P1): Fast Calibration Path for Weekly Audit

### Goal

在 `iv_surface_stability_report` 增加快速校准路径（cache/surrogate），缩短每周审计时间。

### Deliverables

1. 扩展：`validation_scripts/iv_surface_stability_report.py`
- `--fast-calibration` 开关
- cache key 与命中统计输出
2. 输出新增：calibration latency 指标
3. 测试：`tests/test_iv_surface_stability_report.py` 增补 fast-path 用例

### Acceptance Criteria

1. 相同输入下，fast-path 与 baseline 结果偏差在容差内。
2. 审计耗时明显下降（记录在 artifact 中）。
3. 无套利门槛逻辑保持不变。

## Phase 4 (P2): Joint Calibration + Deep No-Arbitrage Baseline

### Goal

提升曲面校准一致性与稀疏样本鲁棒性。

### Deliverables

1. 联合目标：`variance_term_penalty` 注入 model-zoo/calibration 路径。
2. 新增 deep no-arb baseline（实验分支，不默认守门）。
3. 漂移报告扩展：term-structure consistency 指标。

### Acceptance Criteria

1. 不破坏现有 `Research Audit` 默认阈值稳定性。
2. 新指标可在 `research-audit-drift-report` 中可视化追踪。
3. 全部新增测试通过，且 CI 时长增长可控。

## Execution Order

1. Phase 1
2. Phase 3
3. Phase 2
4. Phase 4

## Risk Controls

1. 每 phase 独立提交，避免大批量耦合改动。
2. 保持 fixture-first：先固化输入样本，再调整模型实现。
3. 守门参数默认保守，实验路径通过显式 flag 开启。

## Progress Snapshot (2026-02-25)

1. Phase 1:
- `DONE` inverse-power 模块与测试（`research/pricing/inverse_power_options.py`, `tests/test_inverse_power_options.py`）。
- `DONE` inverse-power 一致性验证脚本与 make 入口（`validation_scripts/inverse_power_validation.py`, `make inverse-power-validate`）。
- `DONE` quanto-inverse-power 基线模块与测试（`research/pricing/quanto_inverse_power.py`, `tests/test_quanto_inverse_power.py`）。
- `PENDING` 将 `InversePowerQuote` 类型补齐并接入基准脚本输入模式。

2. Phase 2:
- `DONE` jump-premia 稳定性报告与 workflow 集成（含阈值门槛）。
- `PENDING` jump premia 的 regime/cluster 状态分层输出。

3. Phase 3:
- `DONE` `iv_surface_stability_report` fast path（`--fast-calibration` + cache + latency metrics）。
- `PENDING` 引入 surrogate/近似器（当前仍是 cache-first fast path）。

4. Phase 4:
- `PENDING` 全部未开始（joint calibration / deep no-arb baseline / term-structure drift）。
