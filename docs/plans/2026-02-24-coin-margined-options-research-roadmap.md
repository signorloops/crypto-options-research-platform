# Coin-Margined Options Research Roadmap (2 Weeks)

**Date:** 2026-02-24  
**Goal:** 将 arXiv 前沿成果转化为可回测、可监控、可上线迭代的币本位期权研究能力。  
**Scope:** 定价、IV 面、跳跃风险溢价信号、模型对比基准。

---

## 0. 进度快照（2026-02-25）

- [x] P0-1 Inverse 短期限 IV/Skew 稳定化（实现 + 测试通过）
- [x] P0-2 Jump Risk Premia 信号化（实现 + 测试通过）
- [x] P0-3 Model Zoo 基准框架（实现 + 测试通过）
- [x] P1-4 Rough + Jumps 实验通道（实现 + 测试通过）
- [x] P1-5 CEX/DeFi 偏离监控（实现 + 测试通过）
- [x] P2 Quanto-Inverse 定价与对冲骨架（实现 + 测试通过）
- [x] 周度自动审计（KPI 快照/风险例外）脚本 + CI 工作流

仍需人工闭环（不可完全自动化）：
- [ ] 灰度发布与 24h 观察（已自动生成 canary checklist，待实盘执行）
- [ ] 收益归因表（spread/adverse/inventory/hedging，模板 + 回测输出字段已补齐，待生产周数据人工确认）
- [ ] 异常项归因与修复计划确认（若周审计出现风险例外）
- [ ] ADR 决策沉淀与回滚决策记录（已自动生成 decision log，待人工签字）

---

## 1. 优先级（论文 -> 工程）

### P0（本轮必须完成）

1. **Inverse 短期限 IV/Skew 稳定化**  
   论文：On the implied volatility of Inverse options under stochastic volatility models (arXiv:2401.00539)  
   目标：提升近月/ATM 反解稳定性，减少局部噪声导致的 skew 抖动。

2. **Clustered Jump Risk Premia 信号化**  
   论文：Jump risk premia in the presence of clustered jumps (arXiv:2510.21297)  
   目标：提取正/负跳跃风险溢价，接入策略/回测作为 alpha 或风控因子。

3. **Crypto Option Pricing Model Zoo 基准**  
   论文：Pricing options on the cryptocurrency futures contracts (arXiv:2506.14614)  
   目标：建立 BS/Merton/Kou/Heston/Bates 对比框架，用统一误差指标选模。

### P1（本轮完成核心骨架）

4. **Rough + Jumps 的 Inverse 定价实验通道**  
   论文：Crypto Inverse-Power Options and Fractional Stochastic Volatility (arXiv:2403.16006)  
   目标：在现有 rough volatility 模块基础上补齐“粗糙波动 + 跳跃”实验能力。

5. **CEX vs DeFi 报价偏离监控（研究态）**  
   论文：Pricing of wrapped Bitcoin and Ethereum on-chain options (arXiv:2512.20190)  
   目标：建立统一基准价与偏离统计，为跨市场策略做准备。

### P2（后续迭代）

6. **Quanto-Inverse 扩展与对冲分析**
   论文：Inverse and Quanto Inverse Options in a Black-Scholes World (arXiv:2107.12041)  
   目标：补齐产品覆盖与不完备市场下的对冲评估模块。

---

## 2. 两周排期（可直接执行）

### Week 1（建能力 + 出第一批可用结果）

1. **Day 1-2: P0-1 Inverse 短期限 IV/Skew 稳定化**
- 代码：
  - `research/volatility/implied.py`
  - `research/pricing/inverse_options.py`
- 测试：
  - `tests/test_volatility.py`
  - `tests/test_pricing_inverse.py`
- 交付：
  - 新增短期限正则/先验开关（默认兼容旧行为）
  - 近月 IV surface 平滑度与静态无套利检查报告（研究脚本）

2. **Day 3-4: P0-2 Jump Risk Premia 信号**
- 代码：
  - `research/signals/`（新增 `jump_risk_premia.py`）
  - `research/backtest/hawkes_comparison.py`（接入信号）
- 测试：
  - 新增 `tests/test_jump_risk_premia.py`
  - 扩展 `tests/test_hawkes_comparison.py`
- 交付：
  - 输出 `positive_jump_premium`, `negative_jump_premium`, `net_jump_premium`
  - 回测里可开关地启用该信号

3. **Day 5: P0-3 Model Zoo 基准框架**
- 代码：
  - `research/pricing/`（新增 `model_zoo.py`）
  - `validation_scripts/`（新增模型比较脚本）
- 测试：
  - 新增 `tests/test_pricing_model_zoo.py`
- 交付：
  - 支持至少 BS/Merton/Kou/Heston/Bates 的统一接口
  - 输出 RMSE/MAE/IV-error 排行

### Week 2（增强 + 接入监控）

4. **Day 6-7: P1-4 Rough + Jumps 实验通道**
- 代码：
  - `research/pricing/rough_volatility.py`
  - `research/pricing/`（新增/扩展 jump 过程配置）
- 测试：
  - `tests/test_rough_volatility.py`（新增）
- 交付：
  - 实验模式支持 co-jump 或 clustered jump 参数
  - 输出 MC 置信区间与耗时统计

5. **Day 8-9: P1-5 CEX/DeFi 偏离监控**
- 代码：
  - `execution/research_dashboard.py`
  - `research/backtest/engine.py`（如需读取额外报价源）
  - `data/`（补充报价接入适配器，研究态可先离线）
- 测试：
  - `tests/test_research_dashboard.py`
- 交付：
  - 价差热力图（按期限/Delta 分桶）
  - 偏离阈值告警（仅研究告警）

6. **Day 10: 回归验证与文档沉淀**
- 文档：
  - `docs/theory.md`
  - `docs/算法与模型深度讲解.md`
  - `ALGORITHMS.md`
- 验证命令：
  - `pytest -q tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py`
  - `pytest -q`

---

## 3. 验收标准（Definition of Done）

1. 近月 IV surface 数值稳定，关键 bucket 无异常尖刺。  
2. Jump premia 信号可被回测调用，且有单独指标输出。  
3. Model Zoo 可复现实验结论并给出模型排名。  
4. Rough + Jumps 实验可运行并输出置信区间。  
5. Dashboard 出现跨市场偏离监控页面或面板。  
6. 新增功能均有测试与文档，默认配置不破坏现有流水线。

---

## 4. 风险与降级策略

1. 若 clustered jump 参数不稳：先降级为“固定窗口估计 + 平滑”。  
2. 若模型 zoo 计算过慢：先以日内抽样和到期分层跑 nightly。  
3. 若 DeFi 数据质量不足：先仅保留 CEX 内部偏离监控框架，待数据源稳定后接入。
