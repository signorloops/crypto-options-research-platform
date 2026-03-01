# 周执行清单（精简模板）

## 0. 季度执行框架（Q2）

季度目标：

1. 稳定性：策略可用性 >= 99.5%，重大回滚 <= 1
2. 风险一致性：误报率较基线下降 >= 30%
3. 收益可解释：归因覆盖率 >= 90%
4. 研发效率：变更可追溯且可回滚

阶段节奏：

1. W1-W4（观测与一致性）：KPI 基线、日级告警、线上/离线一致性
2. W5-W8（治理与风险）：参数版本治理、审批回滚清单、漂移分级动作
3. W9-W12（归因与闭环）：归因流水线、灰度放量、自动回滚阈值与复盘

## 1. 自动化入口

```bash
make weekly-operating-audit
make weekly-close-gate
make live-deviation-snapshot
```

主要产物：

- `artifacts/weekly-operating-audit.*`
- `artifacts/weekly-pnl-attribution.*`
- `artifacts/weekly-canary-checklist.*`
- `artifacts/weekly-signoff-pack.*`
- `artifacts/online-offline-consistency-replay.*`
- `artifacts/weekly-close-gate.*`
- `artifacts/live-deviation-snapshot.*`

## 2. 周信息

- 周次：`YYYY-WW`
- 时间范围：`YYYY-MM-DD ~ YYYY-MM-DD`
- Owner：
- 本周目标（<=3 项）：
1. 
2. 
3. 

## 3. 每日节奏

### 周一（目标）

- KPI 基线更新
- 实验编号分配
- 风险阈值确认

### 周二（实现）

- 代码/参数调整
- 变更记录补齐（实验编号、窗口、配置、回滚版本）
- 最小回归通过

### 周三（验证）

- 一致性检查
- 风险例外报告
- 复杂度审计
- 异常归因

### 周四（灰度）

- 小流量灰度发布
- 24h 观察
- 是否回滚决策

### 周五（复盘）

- KPI、风险、归因、变更、ADR 五项交付
- 下周阻塞项整理

## 4. 周关闭门禁

发布前要求：

1. `weekly-close-gate` 通过
2. 一致性结果非 `FAIL`
3. 人工签字项完整

## 5. 最小记录模板

- 变更记录：实验编号 / 影响 / 回滚版本
- 风险记录：指标 / 阈值 / 实际值 / 动作
- 归因记录：spread / adverse / inventory / hedging
