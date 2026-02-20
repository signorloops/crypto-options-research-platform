# CORP 理论手册

> 精简合并版：币本位基础 + 数学推导 + 代码导读 + 参数标定

---

## 第1章 币本位期权基础

### 1.1 Inverse PnL 公式

币本位（inverse）合约的核心特征：
- 价格 USD 报价
- 盈亏以币结算

**PnL 公式**：
```
PnL_inverse = size × (1/P_entry - 1/P_exit)
```

**代码实现**：
- `research/pricing/inverse_options.py:calculate_pnl()`
- `core/types.py:Position.unrealized_pnl(inverse=True)`

**关键结论**：
- PnL 对价格是**非线性**（1/S 结构）
- 多头 PnL 边际收益递减（凹函数）

### 1.2 Inverse 定价公式

**Inverse Call**：
```
C_inv = e^(-rT) × (1/K) × N(-d2) - (1/S) × N(-d1)
```

**Inverse Put**：
```
P_inv = (1/S) × N(d1) - e^(-rT) × (1/K) × N(d2)
```

其中 d1, d2 基于 K/S 结构（不是标准 S/K）。

**代码**：`research/pricing/inverse_options.py:InverseOptionPricer.calculate_price()`

### 1.3 Greeks 差异

Inverse Greeks 表达式与标准 BS 不同：

| Greek | 标准 BS | Inverse 差异 |
|-------|---------|--------------|
| Delta | N(d1) | 需改写含 1/S 因子 |
| Gamma | n(d1)/(Sσ√T) | 量纲变化 |
| Vega | 需缩放因子 | 注意 1% vs 单位 vol |

**代码**：`research/pricing/inverse_options.py:_calculate_greeks_from_d()`

---

## 第2章 代码架构导读

### 2.1 模块分层

```
┌────────────────────────────────────────┐
│  execution/         服务入口           │
│  trading_engine.py, risk_monitor.py   │
├────────────────────────────────────────┤
│  strategies/        策略实现           │
│  market_making/, arbitrage/           │
├────────────────────────────────────────┤
│  research/          研究工具           │
│  backtest/, pricing/, volatility/     │
├────────────────────────────────────────┤
│  data/              数据层             │
│  downloaders/, cache.py, streaming.py │
├────────────────────────────────────────┤
│  core/              核心层             │
│  types.py, validation/                │
└────────────────────────────────────────┘
```

### 2.2 关键调用链

**做市策略调用链**：
```
TradingEngine.run()
  └─> strategy.quote(market_state, position)
       ├─> _calculate_reservation_price()
       ├─> _calculate_optimal_spread()
       └─> _apply_risk_constraints()
            └─> circuit_breaker.check()
```

**回测调用链**：
```
BacktestEngine.run(market_data)
  ├─> _prepare_events()
  ├─> for event in events:
  │    ├─> strategy.quote()
  │    ├─> fill_simulator.simulate_fill()
  │    └─> _update_portfolio()
  └─> _compute_metrics()
```

### 2.3 核心类型定义

**位置**：`core/types.py`

| 类型 | 用途 |
|------|------|
| `MarketState` | 市场状态（价格、订单簿、时间） |
| `OrderBook` | 订单簿（bids/asks/spread） |
| `Position` | 持仓（size/entry_price） |
| `QuoteAction` | 策略输出（bid/ask/size） |
| `Greeks` | 希腊值（delta/gamma/theta/vega） |

---

## 第3章 策略数学推导

### 3.1 Avellaneda-Stoikov 模型

**核心假设**：
- 成交强度随距离 mid 指数衰减：λ(δ) = A × e^(-kδ)
- 风险厌恶系数 γ

**保留价公式**：
```
r(s,t) = s - q × γ × σ² × (T-t)
```

**最优半价差**：
```
δ = (1/2)γσ²(T-t) + (1/γ) × ln(1 + γ/k)
```

**代码**：`strategies/market_making/avellaneda_stoikov.py`（`AvellanedaStoikov.quote`）

### 3.2 库存控制目标函数

单周期目标函数：
```
J(θ) = E[R_spread] - λ_q × E[q²] - λ_tail × CVaR_α - λ_c × E[C_txn]
```

其中 θ = (γ, base_spread_bps, inventory_limit, skew_factor)

### 3.3 Hawkes 过程

**强度函数**：
```
λ(t) = μ + Σ α × e^(-β(t-t_i))
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| μ | 基线强度 | 0.1 |
| α | 激励因子 | 0.1-0.9 |
| β | 衰减率 | 0.8-1.0 |
| α/β | 分支比 | <1（稳定性条件）|

**长期平均强度**：
```
λ* = μ / (1 - α/β)
```

**动态价差调整**：
```
spread_t = spread_base × m_intensity × m_inventory

m_intensity = {
  aggressive_factor   if λ > high_threshold
  1.0                 if low ≤ λ ≤ high
  defensive_factor    if λ < low_threshold
}
```

**代码**：`strategies/market_making/hawkes_mm.py`（`HawkesMarketMaker.quote`）

### 3.4 逆向选择检测

**启发式规则**：
```
is_adverse = True if |ΔP| > threshold × σ and λ(t) > λ_avg
```

**强度-价差相关性**：
- 理论预期：ρ < 0（高强度 → 窄价差）
- 若 ρ > 0 需检查策略逻辑

---

## 第4章 参数标定

### 4.1 参数分层

| 层级 | 参数示例 | 来源 |
|------|----------|------|
| 市场结构 | σ, k, λ | 统计估计 |
| 策略控制 | base_spread_bps, γ, inventory_limit | 回测寻优 |
| 风控约束 | daily_loss_limit_pct, max_drawdown_pct | 业务决策 |
| 运行工程 | TTL, timeout, retry_interval | 工程经验 |

### 4.2 关键参数标定方法

**γ（风险厌恶）**：
```
γ ≤ (η × S) / (Q × σ² × τ)

其中：
- η: 期望的最大报价偏移比例（如 0.01）
- Q: 库存上限
- S: 标的价格
- σ: 波动率
- τ: 剩余期限
```

**base_spread_bps**：
```
base_spread ∈ [min_spread, max_spread]

优化目标：max Sharpe = E[return] / Std[return]
约束：max_drawdown < limit
```

**inventory_limit**：
- 基于风险预算反推
- 压力回测验证

**Hawkes 参数 (μ, α, β)**：
```
离线估计：
- 使用 3-6 个月历史数据
- 矩估计或 MLE

在线自适应（Adaptive）：
- 每 N 笔交易更新
- 参数边界保护
- 稳定性监控
```

### 4.3 回测验证流程

1. **基线对比**：vs Naive, vs AS
2. **统计检验**：Welch's t-test, p < 0.05
3. **压力测试**：极端行情、流动性枯竭
4. **参数敏感性**：单参数扫描

### 4.4 参数更新策略

| 参数 | 更新频率 | 方法 |
|------|----------|------|
| γ | 月/季 | 滚动回测优化 |
| σ | 日 | 实现波动率 |
| Hawkes μ,α,β | 实时/每笔 | 在线估计 |
| 风控阈值 | 月/季 | 压力测试反推 |

---

## 第5章 2026-02 算法升级补充

### 5.1 Conditional Fill Probability（成交模拟）

- 文件：`research/backtest/engine.py`
- 将固定 fill 概率升级为条件概率模型（logistic）
- 主要输入：queue depth、quote competitiveness、latency、orderbook imbalance、短窗 realized vol
- 目标：让 fill probability 对微观结构状态敏感，而不是常数近似

### 5.2 Sticky Regime Detector（状态切换稳定化）

- 文件：`research/signals/regime_detector.py`
- 状态映射从“均值收益排序”改为“方差/协方差风险排序”
- 新增 sticky 控制：`regime_persistence_min_samples`、`min_confidence_for_switch`、`switch_hysteresis`
- 目标：降低低置信度 regime flip 抖动

### 5.3 Online Calibration（AS + Integrated）

- 文件：`strategies/market_making/avellaneda_stoikov.py`
- 文件：`strategies/market_making/integrated_strategy.py`
- `enable_online_calibration=True` 时，滚动更新 sigma/k（AS）与 effective sigma/skew（Integrated）
- 控制量通过 quote metadata 暴露（如 `calibrated_sigma`）

### 5.4 Full Revaluation VaR（Monte Carlo）

- 文件：`research/risk/var.py`
- `monte_carlo_var` 在持仓表含合约字段时走 full revaluation 路径
- 使用 `InverseOptionPricer.calculate_price` 做路径级重估，而非仅 Greeks 近似
- 保留旧路径作为回退，兼容历史接口

### 5.5 SSVI Volatility Surface

- 文件：`research/volatility/implied.py`
- `VolatilitySurface.get_volatility(..., method="ssvi")`
- 全局拟合 `rho/eta/lambda`，并对 ATM total variance term-structure 做 non-decreasing 约束
- 目标：提升曲面稳定性与跨期限一致性（calendar no-arbitrage 方向）

### 5.6 Marked Hawkes + MLE

（即 marked hawkes 路径）

- 文件：`strategies/market_making/hawkes_mm.py`
- `add_trade(..., size=...)` 引入 marked excitation（size-weighted）
- `estimate_parameters_online(use_mle=True)` 支持在线 MLE，失败回退矩估计
- quote metadata 增加 `control_signals`（intensity、flow_imbalance、spread/skew 控制量）

---

## 第6章 核心代码位置速查

| 功能 | 文件 | 入口 |
|------|------|------|
| **定价** | | |
| Inverse PnL | `research/pricing/inverse_options.py` | calculate_pnl() |
| Inverse Greeks | `research/pricing/inverse_options.py` | _calculate_greeks_from_d() |
| **策略** | | |
| AS 模型 | `strategies/market_making/avellaneda_stoikov.py` | `AvellanedaStoikov.quote` |
| AS Adaptive | `strategies/market_making/avellaneda_stoikov.py` | `ASWithVolatilityAdaptation.quote` |
| Hawkes | `strategies/market_making/hawkes_mm.py` | `HawkesMarketMaker.quote` |
| Hawkes Adaptive | `strategies/market_making/hawkes_mm.py` | `AdaptiveHawkesMarketMaker.quote` |
| Naive | `strategies/market_making/naive.py` | `NaiveMarketMaker.quote` |
| Integrated | `strategies/market_making/integrated_strategy.py` | `IntegratedMarketMakingStrategy.quote` |
| **回测** | | |
| 引擎 | `research/backtest/engine.py` | run() |
| 对比 | `research/backtest/hawkes_comparison.py` | ComprehensiveHawkesComparison |
| **风控** | | |
| 熔断 | `research/risk/circuit_breaker.py` | check_risk_limits() |
| VaR | `research/risk/var.py` | parametric_var() |
| **数据** | | |
| Deribit | `data/downloaders/deribit.py` | DeribitClient |
| 缓存 | `data/cache.py` | DataCache |

---

## 附录：推荐阅读顺序

**第1遍（运行）**：
1. 本手册第1章（币本位基础）
2. 本手册第2章（架构导读）
3. 跑通 `notebooks/06_hawkes_backtest_comparison.ipynb`

**第2遍（理解）**：
1. 本手册第3章（数学推导）
2. 对照代码阅读策略实现
3. 跑完整的对比实验

**第3遍（深入）**：
1. 本手册第4章（参数标定）
2. 调参实验，观察结果变化
3. 尝试实现新的策略变体

---

*本手册由4本中文深度学习手册精简合并而成*
*原手册已归档至 docs/archive/*
