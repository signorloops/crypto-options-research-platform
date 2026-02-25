# 2026-03 Complexity Reduction Sprint（2周）

Date: 2026-02-25  
Duration: 2 weeks (10 working days)  
Owner: TBD  
Type: 行为不变重构（no business logic expansion）

---

## 1. 目标与边界

目标：在不影响现有研究路线与发布节奏的前提下，降低架构复杂度、减少耦合、提高可测性。

硬边界：

1. 不新增交易策略行为。  
2. 不调整风控阈值口径。  
3. 不修改外部 API 契约（函数签名若变化，需提供兼容层）。  
4. 任何任务失败可单独回滚。  

---

## 2. 与现有计划兼容性

与以下文档并行执行，不替代原计划：

1. `docs/plans/2026-Q2-long-term-execution-roadmap.md`
2. `docs/plans/weekly-operating-checklist.md`
3. `docs/complexity-governance.md`

映射关系：

1. 对齐 Q2 W2-W4：支撑“线上/离线一致性框架 + 日回归/周回放门禁”。  
2. 对齐每周清单周三项：复杂度审计与一致性检查合并执行。  
3. 对齐治理门禁：所有重构必须通过 `complexity-audit` 与最小回归集。  

---

## 3. 优先级任务（按收益/风险排序）

### P0-1 拆除反向依赖（高收益 / 低风险）

- [x] 将 `StrategyComparison` 从 `strategies/base.py` 移至 `research/backtest/`  
- [x] 清除 `strategies -> research.backtest` 反向 import  
- [x] 增加边界测试，禁止该回归

DoD:

1. `strategies/base.py` 不再 import `research.backtest`。  
2. 回测相关测试通过。  

### P0-2 统一包懒加载（高收益 / 低风险）

- [x] 将 `research/pricing/__init__.py` 改为 lazy export  
- [x] 将 `research/volatility/__init__.py` 改为 lazy export  
- [x] 将 `research/hedging/__init__.py` 改为 lazy export  
- [x] 将 `research/signals/__init__.py` 与 `strategies/arbitrage/__init__.py` 改为 lazy export

DoD:

1. `import research.*` 不触发可选依赖错误。  
2. 相关测试通过，导入行为与 `__all__` 兼容。  

### P1-3 拆 `research/risk/var.py` 大函数（高收益 / 中风险）

- [x] 抽离 `monte_carlo_var` 内部块为纯函数（例如路径生成、重估、聚合）  
- [x] 新增同级 helper 方法并保留兼容入口  
- [x] 增加函数级单测覆盖

DoD:

1. `var.py` 外部接口不变。  
2. `monte_carlo_var` 复杂度下降，周度复杂度报告可见。  

### P1-4 拆回测成交路径（高收益 / 中风险）

- [x] 将 fill probability 估计与成交构造拆到 `research/backtest/fill_model.py`  
- [x] `engine.py` 仅保留调度与生命周期逻辑  
- [x] 保持结果一致性（同随机种子）

DoD:

1. 回测结果在容忍范围内一致。  
2. `research/backtest/engine.py` LOC 明显下降。  

### P2-5 拆 `research/volatility/implied.py`（中高收益 / 中风险）

- [ ] 将曲面拟合、查询、校验拆分子模块（例如 `surface_fit.py`, `surface_query.py`）  
- [ ] 保留 `VolatilitySurface` 门面类与向后兼容导入路径

DoD:

1. 外部调用路径可用。  
2. 波动率与定价相关回归测试全绿。  

### P2-6 CI 增加架构守门（中收益 / 低风险）

- [x] 新增 `tests/test_architecture_boundaries.py`（分层依赖断言）  
- [ ] 在 CI 中为 changed files 启用更严格 lint/type 检查（逐步收紧）  
- [ ] 将该检查纳入周三审计基线

DoD:

1. PR 能自动阻断新增反向依赖。  
2. CI 不显著拉长（目标增量 <= 20%）。  

---

## 4. 两周执行排期（可直接勾选）

### Week 1（Day 1-5）

- [x] Day 1: P0-1 反向依赖拆除 + 边界测试  
- [x] Day 2: P0-2 pricing/volatility lazy export  
- [x] Day 3: P0-2 hedging/signals/arbitrage lazy export  
- [x] Day 4: P1-3 var 拆分（第一批）  
- [x] Day 5: P1-3 var 拆分（收敛）+ 周回归

### Week 2（Day 6-10）

- [x] Day 6: P1-4 fill model 拆分（第一批）  
- [x] Day 7: P1-4 fill model 回归与性能对比  
- [ ] Day 8: P2-5 implied 模块拆分（第一批）  
- [ ] Day 9: P2-5 implied 收敛 + 全量相关测试  
- [ ] Day 10: P2-6 CI 架构守门 + 文档沉淀 + ADR

---

## 5. 每日与每周验证命令

日常最小验证：

```bash
make complexity-audit
venv/bin/python -m pytest -q \
  tests/test_pricing_inverse.py \
  tests/test_volatility.py \
  tests/test_hawkes_comparison.py \
  tests/test_research_dashboard.py \
  tests/test_weekly_operating_audit.py
```

周三/周五验证（与周清单对齐）：

```bash
make weekly-operating-audit
make complexity-audit
venv/bin/python -m pytest -q
```

---

## 6. 量化验收目标（Sprint End）

基于 `artifacts/complexity-governance-report.md`：

1. `total_loc`: 24019 -> <= 23000  
2. `max_file_loc`: 1104 -> <= 1000  
3. `files_over_soft_loc`: 6 -> <= 5  
4. `functions_over_soft_loc`: 22 -> <= 18  
5. `classes_over_method_soft_limit`: 3 -> <= 2  

说明：若业务功能扩展导致体量增加，不放宽阈值；需以“拆分抵消增量”达成目标。

---

## 7. 风险与回滚

1. 任务按 PR 粒度独立提交，失败可逐项回滚。  
2. 若任一核心回归失败，停止后续拆分，先恢复可发布状态。  
3. 禁止通过调宽 `config/complexity_budget.json` 掩盖问题。  

---

## 8. 交付物清单

1. 代码重构 PR（按任务编号）。  
2. 每周复杂度报告与周审计报告。  
3. 一份 ADR：记录拆分策略、风险、回滚点、后续欠账。  
