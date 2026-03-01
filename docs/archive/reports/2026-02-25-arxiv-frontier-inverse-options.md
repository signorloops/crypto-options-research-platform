# arXiv 前沿清单：币本位期权可落地方向（2026-02-25）

## 结论先行

对当前仓库最直接可落地的论文是：

1. `arXiv:2403.16006`（Crypto Inverse-Power Options and Fractional Stochastic Volatility）
2. `arXiv:2510.21297`（Jump risk premia in the presence of clustered jumps）
3. `arXiv:2510.19126`（Efficient calibration under rough volatility with jumps）

这三篇可以分别加强：

- 币本位/quanto 逆幂期权定价层
- BTC 期权跳跃风险溢价因子与策略解释层
- 研究审计中的“快速校准+漂移守门”层

## 论文清单与项目映射

### 1) Crypto Inverse-Power Options and Fractional Stochastic Volatility
- 链接: https://arxiv.org/abs/2403.16006
- 版本时间: v3, 2025-06-16
- 为什么重要:
  - 直接覆盖 inverse options、quanto inverse options 与 power-type 扩展。
  - 显式强调 price-vol co-jumps + fractional volatility，对 crypto 数据更贴近。
- 可落地到本项目:
  - 在 `research/pricing/inverse_options.py` 基础上新增 `inverse_power`/`quanto_inverse_power` 定价与 Greeks。
  - 在 `validation_scripts` 增加 inverse-power 校准对比报告（与现有 model-zoo 并行）。

### 2) Jump risk premia in the presence of clustered jumps
- 链接: https://arxiv.org/abs/2510.21297
- 版本时间: 2025-10-24
- 为什么重要:
  - 针对 BTC options，建模正负跳跃簇与 jump premia，并展示了对 carry 与 delta-hedged 收益的预测性。
- 可落地到本项目:
  - 扩展 `research/signals/jump_risk_premia.py`：加入正/负 jump premia 分解与状态分层。
  - 在 `research-audit` 新增 “jump premia stability” 子报告，纳入每周漂移监控。

### 3) An Efficient Calibration Framework for Volatility Derivatives under Rough Volatility with Jumps
- 链接: https://arxiv.org/abs/2510.19126
- 版本时间: 2025-10-21
- 为什么重要:
  - 给出“积分预计算 + 小网络近似残差 + 全局到局部搜索”的高效校准框架。
- 可落地到本项目:
  - 在 `validation_scripts/iv_surface_stability_report.py` 增加“快速校准模式（cache + surrogate）”。
  - 缩短 `Research Audit` 运行时延，为 CI 周期留余量。

### 4) Joint calibration of the volatility surface and variance term structure
- 链接: https://arxiv.org/abs/2509.08096
- 版本时间: 2025-09-09
- 为什么重要:
  - 指出仅拟合 IV surface 会与 variance term structure 脱节，并给出联合目标函数。
- 可落地到本项目:
  - 在 model-zoo / 校准脚本加入 `variance_term_penalty`。
  - 为 `research_audit_compare.py` 增加 term-structure 一致性漂移指标。

### 5) Risk-Sensitive Option Market Making with Arbitrage-Free eSSVI Surfaces
- 链接: https://arxiv.org/abs/2510.04569
- 版本时间: 2025-10-06
- 为什么重要:
  - 将 eSSVI 无套利约束、做市控制与 CVaR 风险目标放入统一框架。
- 可落地到本项目:
  - 把当前 `iv_surface_stability_report` 的 no-arb 检查升级为可微约束层实验分支。
  - 在 `strategies/market_making` 侧增加 CVaR 约束参数化实验。

### 6) No-Arbitrage Deep Calibration for Volatility Smile and Skewness
- 链接: https://arxiv.org/abs/2310.16703
- 版本时间: 2023-10-25
- 为什么重要:
  - 深度校准中显式加入导数约束来满足无套利，适合稀疏、噪声 IV 数据。
- 可落地到本项目:
  - 新增深度校准 baseline，与当前规则化平滑方法并列比较。
  - 将 butterfly/calendar 违规率纳入模型选择目标。

## 建议执行顺序（按 ROI）

1. `P0`：先落地 `2403.16006` 的 inverse-power/quanto 扩展（直接增强币本位主线）。
2. `P1`：接入 `2510.21297` 的 jump premia 分解，补齐策略解释力。
3. `P1`：把 `2510.19126` 的快速校准思路放进 audit pipeline（优先提升 CI 速度）。
4. `P2`：联合校准（`2509.08096`）和深度无套利校准（`2310.16703`）作为增强分支。

## 备注

- 上述均为 arXiv 预印本，建议“先复现实证结论，再纳入默认生产守门阈值”。
- 当前仓库已经有 `rough_jump_experiment` 与 `iv_surface_stability_report`，可作为接入这些论文方法的最短路径。
