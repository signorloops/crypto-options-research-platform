---
name: CORP算法优化计划
overview: 对 CORP 项目的算法层（波动率模型、交易策略、回测引擎、风控系统）进行全面审查，识别数学错误、性能瓶颈和缺失功能，提出分优先级的改进方案。
todos:
  - id: fix-realized-vol
    content: "BUG-2: 修复 realized_volatility 缺少 /N 归一化，影响所有下游模块"
    status: completed
  - id: fix-backtest-costs
    content: "BUG-1: 回测引擎接入滑点 + 手续费 + adverse selection，并修复 Arena 属性名不匹配 (BUG-7)"
    status: completed
  - id: fix-max-drawdown
    content: "BUG-3: drawdown.min() 改为 .max()"
    status: completed
  - id: fix-vpin-threshold
    content: "BUG-4: VPIN 毒性阈值改为 0.35-0.45 或调整范围计算"
    status: completed
  - id: fix-hmm-states
    content: "BUG-5: HMM 训练后按波动率均值排序重映射 state"
    status: completed
  - id: fix-as-time
    content: "BUG-6: AS 策略用 state.timestamp 替换 time.time()"
    status: completed
  - id: fix-rough-vol
    content: "BUG-8: 修复 rough_volatility_signature 的 variogram 计算"
    status: completed
  - id: fix-hawkes-cv
    content: "BUG-9: Hawkes CV 公式改为 sqrt(var)/mean"
    status: completed
  - id: fix-greeks-usd
    content: "BUG-10: 修复 inverse options 的 USD 转换双重计算"
    status: completed
  - id: fix-crypto-periods
    content: 全局将默认 periods=252 改为 365（crypto）
    status: completed
  - id: perf-hawkes-o1
    content: "PERF-1: Hawkes 强度计算改为 O(1) 递推"
    status: completed
  - id: perf-garch-jit
    content: "PERF-2: EWMA/GARCH 用 lfilter/numba 替换 Python 循环"
    status: completed
  - id: upgrade-iv-solver
    content: 隐含波动率求解器升级为 Jaeckel Let's Be Rational
    status: completed
  - id: upgrade-var
    content: "VaR 模型升级: Cornish-Fisher / EVT + VaR 检查接入熔断器"
    status: completed
  - id: remove-lookahead
    content: 移除合成数据中 informed trade 的前瞻偏差
    status: completed
isProject: false
---

# CORP 算法层深度审查与优化计划

## 发现的关键 Bug 汇总（必须修复）

以下是会直接导致错误结果的 Bug，按严重程度排列：

### BUG-1: 回测引擎交易成本完全失效

回测结果中的 PnL **完全没有扣除手续费和滑点**，导致所有策略的盈利数据严重失真。

- [corp/research/backtest/engine.py](corp/research/backtest/engine.py): `RealisticFillSimulator.simulate_fill()` 以裸报价价格成交，零滑点
- `_apply_slippage()` 和 `_simulate_fill_simple()` 存在但**从未被调用**
- `_check_adverse_selection()` 计算了结果但**直接丢弃**，未影响成交
- [corp/research/backtest/arena.py](corp/research/backtest/arena.py): `transaction_cost_bps=2.0` 存储了但**未传递给 BacktestEngine**

### BUG-2: realized_volatility 缺少 /N 归一化

[corp/research/volatility/historical.py](corp/research/volatility/historical.py): `realized_volatility()` 计算 `sqrt(sum(r^2))` 而非 `sqrt(mean(r^2))`，导致波动率随样本量增大而增大。**所有下游依赖此函数的模块（策略、风控、定价）都受影响。**

```python
# 当前（错误）: vol 随 N 增长
rv = realized_variance(returns)  # sum(r^2)
vol = np.sqrt(rv) * np.sqrt(periods)

# 修复: 除以 N
vol = np.sqrt(rv / len(returns)) * np.sqrt(periods)
```

### BUG-3: Max Drawdown 计算反了

[corp/research/backtest/engine.py](corp/research/backtest/engine.py): `drawdown.min()` 取的是最小回撤而非最大回撤。应改为 `.max()`。

### BUG-4: VPIN 毒性阈值永远无法触发

[corp/research/microstructure/vpin.py](corp/research/microstructure/vpin.py): VPIN 范围为 [0, 0.5]（除以了 `2 * total_vol`），但 `get_high_toxicity_periods` 默认阈值为 0.6，永远不会触发告警。

### BUG-5: HMM 状态标签随机排列

[corp/research/signals/regime_detector.py](corp/research/signals/regime_detector.py) 和 [fast_regime_detector.py](corp/research/signals/fast_regime_detector.py): HMM 训练后 state 0/1/2 与 LOW/MEDIUM/HIGH 的对应关系是随机的，每次重新训练后可能交换。需要按波动率均值排序后重映射。

### BUG-6: Avellaneda-Stoikov 用墙钟时间计算 (T-t)

[corp/strategies/market_making/avellaneda_stoikov.py](corp/strategies/market_making/avellaneda_stoikov.py): 用 `time.time()` 而非 `state.timestamp` 计算剩余时间。在回测中 (T-t) 永远不会正确递减。

### BUG-7: Arena 引用不存在的属性

[corp/research/backtest/arena.py](corp/research/backtest/arena.py): `_calculate_scorecard` 引用 `result.total_pnl` 和 `result.avg_trade_pnl`，但 `BacktestResult` 定义的是 `total_pnl_crypto` 和 `avg_trade_pnl_crypto`，运行时会 `AttributeError`。

### BUG-8: 粗糙波动率特征函数计算错误

[corp/research/volatility/models.py](corp/research/volatility/models.py): `rough_volatility_signature` 用 `returns[::delta]`（子采样）而非 block 累加。计算出的 Hurst 指数无意义。

### BUG-9: Hawkes CV 公式错误

[corp/strategies/market_making/hawkes_mm.py](corp/strategies/market_making/hawkes_mm.py): `cv = var_iet / mean_iet`，但变异系数应为 `std / mean = sqrt(var) / mean`。

### BUG-10: Portfolio Greeks USD 转换可能双重计算 spot

[corp/research/risk/greeks.py](corp/research/risk/greeks.py): 当 `fx_rate == spot` 时，inverse option 的 delta 被乘以 spot^2 而非 spot。

---

## 性能优化

### PERF-1: Hawkes 强度计算 O(n) -> O(1)

[corp/strategies/market_making/hawkes_mm.py](corp/strategies/market_making/hawkes_mm.py) 和 [corp/data/generators/hawkes.py](corp/data/generators/hawkes.py): 每次计算遍历所有历史事件。Hawkes 核有递推公式：

```
A(n) = exp(-beta * (t_n - t_{n-1})) * (A(n-1) + alpha)
lambda(t) = mu + A(n)
```

可将 O(n) 降为 O(1)，总体 O(n^2) 降为 O(n)。

### PERF-2: EWMA / GARCH 纯 Python 循环

[corp/research/volatility/models.py](corp/research/volatility/models.py): EWMA 和 GARCH 的核心循环用纯 Python 实现，比 NumPy 向量化或 Numba JIT 慢 10-100x。GARCH MLE 优化中该循环被调用数百次。

建议: 用 `scipy.signal.lfilter` 做 EWMA，用 `@numba.jit(nopython=True)` 做 GARCH likelihood。

### PERF-3: XGBoost 每次报价做 20 次预测

[corp/strategies/market_making/xgboost_spread.py](corp/strategies/market_making/xgboost_spread.py): `_predict_spread` 对 20 个候选 spread 做网格搜索，每次 quote 调用 20 次 XGBoost predict。应改为直接回归最优 spread 的单模型。

### PERF-4: 快速策略缓存淘汰 O(n log n)

[corp/strategies/market_making/fast_integrated_strategy.py](corp/strategies/market_making/fast_integrated_strategy.py): Greeks 缓存溢出时对所有 key 排序。应用 `collections.OrderedDict` 实现 O(1) LRU。

### PERF-5: VPIN 逐笔循环创建 volume bucket

[corp/research/microstructure/vpin.py](corp/research/microstructure/vpin.py): 应用 `np.cumsum` + `np.searchsorted` 向量化。

### PERF-6: 回测引擎全量 DataFrame -> list of dicts

[corp/research/backtest/engine.py](corp/research/backtest/engine.py): `_prepare_events()` 把整个 DataFrame 转为 dict list，内存翻三倍。应用 `itertuples()` 或直接按索引访问。

---

## 模型/策略改进建议

### 波动率模型

| 当前                  | 建议                                 | 优先级 | 参考文献                                |
| ------------------- | ---------------------------------- | --- | ----------------------------------- |
| 默认 periods=252      | 改为 365（crypto 全年交易）                | P0  | -                                   |
| 无跳跃鲁棒估计器            | 添加 Bipower Variation / MedRV       | P2  | Barndorff-Nielsen & Shephard (2004) |
| 无微结构噪声处理            | 添加 Two-Scale / Realized Kernel 估计器 | P2  | Zhang, Mykland & Ait-Sahalia (2005) |
| 朴素 regime-switching | 实现 Hamilton filter（EM + 转移概率矩阵）    | P2  | Hamilton (1989)                     |
| 无 EGARCH            | 添加 EGARCH/GJR-GARCH 捕捉杠杆效应         | P2  | Nelson (1991)                       |


### 隐含波动率

| 当前                   | 建议                                 | 优先级 | 参考文献                       |
| -------------------- | ---------------------------------- | --- | -------------------------- |
| 二分法 + Newton-Raphson | 实现 Jaeckel "Let's Be Rational"     | P1  | Jaeckel (2017)             |
| IDW 插值（维度不一致）        | 实现 SVI 参数化（无套利曲面）                  | P1  | Gatheral & Jacquier (2014) |
| 无套利检查                | 添加 butterfly/calendar spread 无套利验证 | P2  | -                          |


### 做市策略

| 策略                 | 关键改进                                                     | 优先级 |
| ------------------ | -------------------------------------------------------- | --- |
| Avellaneda-Stoikov | 升级为 Gueant-Lehalle-Fernandez-Tapia (2013) 有界库存版本         | P1  |
| Hawkes MM          | 修复 CV 公式 + O(1) 递推 + 非对称 adverse selection 响应            | P1  |
| Integrated         | `_get_spread_multiplier` 改为乘法组合（而非 max）                  | P1  |
| Fast Integrated    | 统一 log/simple return 计算 + 修复 `circuit_state` 未绑定         | P1  |
| PPO Agent          | 网络扩大到 256 units + LSTM + 扩充状态空间到 20-30 特征 + 训练 1M+ steps | P2  |
| XGBoost            | 改为直接回归最优 spread + 修复训练/推理特征分布不一致                         | P2  |


### 套利策略

| 策略             | 关键改进                                    | 优先级 |
| -------------- | --------------------------------------- | --- |
| Cross-Exchange | 添加执行模拟 + 滑点模型 + 三角套利支持                  | P2  |
| Basis          | 动态 funding rate + 保证金/清算风险建模 + 反向合约对冲比率 | P2  |
| Conversion     | 加入 streaming 价格更新 + ETH staking yield   | P3  |
| Box Spread     | 添加 short box 检测 + 流动性过滤                 | P3  |


### 风控系统

| 当前                         | 建议                                                       | 优先级 |
| -------------------------- | -------------------------------------------------------- | --- |
| 高斯 VaR                     | 添加 Cornish-Fisher / EVT / Filtered Historical Simulation | P1  |
| Monte Carlo 无 spot-vol 相关性 | 加入杠杆效应相关矩阵                                               | P2  |
| VaR 检查未集成到熔断器              | 将 `check_var_limit` 接入 `_check_all_limits`               | P1  |
| 无 per-instrument 熔断        | 添加标的级别熔断器                                                | P2  |
| asyncio/threading 混用       | 统一为 asyncio 或用 ThreadPoolExecutor 桥接                     | P1  |


### 回测引擎

| 当前                      | 建议                                    | 优先级 |
| ----------------------- | ------------------------------------- | --- |
| 零交易成本                   | 接入滑点模型 + maker/taker fee              | P0  |
| 合成数据 informed trade 看未来 | 移除 `price_path.iloc[min(i+1,...)]` 前瞻 | P0  |
| 无 bootstrap CI          | 添加 Sharpe/Drawdown 的 bootstrap 置信区间   | P1  |
| 无 Deflated Sharpe Ratio | 多策略比较时校正多重检验                          | P2  |
| 252 天年化                 | 改为 365 天（crypto）                      | P1  |
| 全局 np.random.seed       | 改为隔离的 np.random.Generator             | P2  |


### 缺失的前沿算法（长期规划）

- **Deep Hedging** (Buehler et al., 2019) -- 已实现并增强（特征标准化、诊断接口）（`corp/research/hedging/deep_hedging.py`）
- **Optimal Execution with Market Impact** (Almgren-Chriss) -- 已实现并增强（参与率上限、成本分解）（`corp/research/execution/almgren_chriss.py`）
- **Rough Volatility Pricing** (Gatheral et al., 2018) -- 已实现并增强（antithetic + 置信区间）（`corp/research/pricing/rough_volatility.py`）
- **Online Bayesian Changepoint Detection** (Adams & MacKay, 2007) -- 已实现并增强（batch 更新 + Top-K 变点）（`corp/research/signals/bayesian_changepoint.py`）
- **Volatility Surface Arbitrage Detection** -- 已补充套利机会扫描报告（`corp/research/volatility/implied.py`）

---

## 建议实施路线

**第一阶段（1-3 天）: 修复致命 Bug**

- BUG-1 ~ BUG-10，尤其是回测交易成本和 realized_volatility 归一化

**第二阶段（1-2 周）: 核心性能和模型改进**

- PERF-1 ~ PERF-6 性能优化
- 隐含波动率求解器升级（Jaeckel）
- VaR 模型升级（Cornish-Fisher / EVT）
- AS 策略升级为有界库存版本
- 回测引擎接入交易成本

**第三阶段（2-4 周）: 策略增强**

- PPO 网络重构 + 扩充训练
- XGBoost 模型改造
- 套利策略执行模拟
- SVI 波动率曲面

**第四阶段（按需）: 前沿算法**

- Deep Hedging
- Rough Volatility
- Bayesian Changepoint Detection
