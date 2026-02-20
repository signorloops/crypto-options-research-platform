# Hawkes 做市策略回测对比实验

本文档详细介绍 Hawkes 做市策略的回测对比实验设计、实施方法和结果分析。

## 目录

1. [实验目标](#实验目标)
2. [对比策略](#对比策略)
3. [测试场景](#测试场景)
4. [评估指标](#评估指标)
5. [实验框架](#实验框架)
6. [使用方法](#使用方法)
7. [结果解读](#结果解读)
8. [参考文献](#参考文献)

---

## 实验目标

验证 Hawkes 做市策略相比现有策略的优势和劣势，具体目标包括：

1. **理论验证**: 在高聚类市场中验证 Hawkes 策略的理论优势
2. **基准对比**: 与经典 AS 模型和 ML 策略进行公平对比
3. **稳健性测试**: 评估策略在不同市场条件下的表现稳定性
4. **参数优化**: 识别最优参数配置和自适应机制效果

---

## 对比策略

| 策略 | 代码位置 | 核心特点 | 适用场景 |
|------|----------|----------|----------|
| **Hawkes MM (固定参数)** | `strategies/market_making/hawkes_mm.py` (`HawkesMarketMaker`) | 基于 Hawkes 过程预测订单流聚类 | 已知市场状态 |
| **Adaptive Hawkes MM** | `strategies/market_making/hawkes_mm.py` (`AdaptiveHawkesMarketMaker`) | 在线参数更新，自适应市场变化 | 变化市场环境 |
| **Avellaneda-Stoikov** | `strategies/market_making/avellaneda_stoikov.py` (`AvellanedaStoikov`) | 经典做市理论，风险厌恶最优 | 稳定波动率市场 |
| **AS + Adaptive Vol** | `strategies/market_making/avellaneda_stoikov.py` (`ASWithVolatilityAdaptation`) | AS + 实现波动率动态调整 | 波动率变化市场 |
| **XGBoost Spread** | `strategies/market_making/xgboost_spread.py` (`XGBoostSpreadStrategy`) | 机器学习预测最优价差 | 大数据训练可用 |
| **Naive (固定价差)** | `strategies/market_making/naive.py` | 简单基准，固定对称价差 | 基准对照 |

### 策略配置示例

```python
# Hawkes MM 配置
hawkes_config = HawkesMMConfig(
    base_spread_bps=20.0,
    min_spread_bps=5.0,
    max_spread_bps=100.0,
    hawkes_mu=0.1,      # 基线强度
    hawkes_alpha=0.4,   # 激励因子
    hawkes_beta=0.8,    # 衰减率
    high_intensity_threshold=0.3,
    low_intensity_threshold=0.05,
    aggressive_factor=0.5,
    defensive_factor=2.0,
)

# AS 模型配置
as_config = ASConfig(
    gamma=0.1,          # 风险厌恶系数
    sigma=0.5,          # 波动率
    k=1.5,              # 订单到达强度
    T=1.0,              # 交易期限
)
```

---

## 测试场景

### 场景 A: 合成 Hawkes 数据

使用 `data/generators/hawkes.py` 生成不同聚类程度的交易流：

| 场景 | α (激励因子) | β (衰减率) | 分支比 | 聚类程度 | 预期结果 |
|------|--------------|------------|--------|----------|----------|
| **低聚类** | 0.1 | 0.8 | 0.125 | 接近泊松 | Hawkes ≈ AS |
| **中聚类** | 0.4 | 0.8 | 0.500 | 典型市场 | Hawkes 略优 |
| **高聚类** | 0.7 | 0.8 | 0.875 | 高活跃期 | Hawkes 显著优势 |
| **临界状态** | 0.9 | 1.0 | 0.900 | 接近不稳定 | Adaptive 优势 |

**生成代码：**
```python
from research.backtest.hawkes_comparison import ScenarioGenerator

gen = ScenarioGenerator(base_price=50000.0)
scenarios = gen.generate_hawkes_scenarios(seed_offset=0)
```

### 场景 B: 真实历史数据

| 期间 | 时间范围 | 市场特征 | 数据源 |
|------|----------|----------|--------|
| **低波动期** | 2023年8月 | 稳定市场，低聚类 | Deribit 历史数据 |
| **高波动期** | 2024年3月 | 活跃市场，高聚类 | Deribit 历史数据 |
| **极端事件** | 特定日期 | 异常市场条件 | 事件驱动分析 |

### 场景 C: 压力测试

| 场景 | 描述 | 测试目标 |
|------|------|----------|
| **突发交易量** | 模拟新闻事件，交易量突然增加 | 策略响应速度 |
| **流动性枯竭** | 宽价差、少成交 | 极端条件稳定性 |
| **库存极限** | 持续单向成交 | 库存管理能力 |

**压力测试代码：**
```python
stress_scenarios = gen.generate_stress_scenarios()
# 包含: volume_spike, liquidity_drought, inventory_stress
```

---

## 评估指标

### 收益指标

| 指标 | 说明 | 计算公式 | 目标值 |
|------|------|----------|--------|
| **Total PnL** | 总收益 (USD) | 期末权益 - 期初权益 | > 0 |
| **Annualized Return** | 年化收益率 | (1 + 总收益率)^(365/天数) - 1 | > 0 |
| **Sharpe Ratio** | 夏普比率 | (收益率均值 - 无风险利率) / 标准差 × √252 | > 1 |
| **Sortino Ratio** | 索提诺比率 | (收益率均值 - 目标) / 下行标准差 × √252 | > 1 |
| **Calmar Ratio** | 卡尔马比率 | 年化收益率 / 最大回撤 | > 2 |

### 风险指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Max Drawdown** | 最大回撤 | < 10% |
| **Daily PnL Std** | 日收益标准差 | 尽可能低 |
| **VaR (95%)** | 风险价值 | 可控范围内 |
| **CVaR** | 条件风险价值 | 可控范围内 |

### 做市专项指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **Spread Capture** | 价差捕获 ($) | 成交价差总和 |
| **Adverse Selection Cost** | 逆向选择成本 | 有毒订单流损失 |
| **Inventory Cost** | 库存成本 | 持仓带来的成本 |
| **Fill Rate** | 成交率 | 成交次数 / 报价次数 |
| **Quote Update Frequency** | 报价更新频率 | 报价调整次数 / 总时间 |

### Hawkes 专项指标

| 指标 | 说明 | 预期值 |
|------|------|--------|
| **平均 Hawkes 强度 λ(t)** | 订单流强度 | 随市场变化 |
| **价差与强度相关系数** | 强度-价差关系 | 负值（高强度→窄价差） |
| **参数稳定性** | 自适应版本参数变化 | 平稳收敛 |
| **逆向选择检测准确率** | 检测效果 | > 60% |

---

## 实验框架

### 核心类设计

```python
# research/backtest/hawkes_comparison.py

class ScenarioGenerator:
    """生成不同市场场景的测试数据."""

    def generate_hawkes_scenarios(self) -> Dict[str, pd.DataFrame]:
        """生成不同聚类程度的合成数据."""

    def load_real_scenarios(self, date_ranges) -> Dict[str, pd.DataFrame]:
        """加载真实历史数据的不同期间."""

    def generate_stress_scenarios(self) -> Dict[str, pd.DataFrame]:
        """生成压力测试场景."""

class HawkesMetricsCollector:
    """收集 Hawkes 策略特有的指标."""

    def collect_intensity_spread_correlation(self) -> float:
        """计算强度与价差的相关系数."""

    def analyze_adverse_selection_detection(self) -> Dict:
        """分析逆向选择检测效果."""

    def track_parameter_stability(self) -> pd.DataFrame:
        """追踪自适应版本的参数变化."""

class ComprehensiveHawkesComparison:
    """综合对比框架."""

    def run_full_comparison(self, strategies, scenarios) -> pd.DataFrame:
        """运行完整对比实验."""

    def generate_report(self, output_path: str):
        """生成完整报告."""

    def plot_comparison(self, metrics: List[str]):
        """生成对比图表."""
```

---

## 使用方法

### 1. 快速开始

```python
from research.backtest.hawkes_comparison import (
    ScenarioGenerator,
    ComprehensiveHawkesComparison
)
from strategies.market_making.hawkes_mm import (
    HawkesMarketMaker, HawkesMMConfig
)
from strategies.market_making.avellaneda_stoikov import (
    AvellanedaStoikov, ASConfig
)

# 初始化策略
strategies = [
    HawkesMarketMaker(HawkesMMConfig()),
    AvellanedaStoikov(ASConfig()),
    # ... 其他策略
]

# 生成场景
gen = ScenarioGenerator()
scenarios = gen.generate_hawkes_scenarios()

# 运行对比
comparison = ComprehensiveHawkesComparison(
    initial_capital=100000.0,
    transaction_cost_bps=2.0
)
results = comparison.run_full_comparison(strategies, scenarios)

# 生成报告
report = comparison.generate_summary_report()
print(report)
```

### 2. 使用 Jupyter Notebook

```bash
# 启动 Jupyter
jupyter notebook notebooks/06_hawkes_backtest_comparison.ipynb
```

Notebook 包含：
- 交互式参数调整
- 实时可视化
- 结果导出

### 3. 运行单元测试

```bash
pytest tests/test_hawkes_comparison.py -v
```

---

## 结果解读

### 预期结果

#### Hawkes 应该胜出的场景

1. **高聚类合成数据** (分支比 > 0.6)
   - 原因: Hawkes 过程准确建模订单流聚类
   - 预期: Sharpe Ratio 比其他策略高 20%+

2. **真实市场高活跃期**
   - 原因: 实际市场存在订单流聚类
   - 预期: 总收益比 AS 模型高 10-30%

3. **突发交易量场景**
   - 原因: Hawkes 能检测强度突变
   - 预期: 响应更快，回撤更小

#### Hawkes 可能落后的场景

1. **低聚类/泊松市场**
   - 原因: Hawkes 退化为泊松，额外复杂度无收益
   - 预期: 与 Naive/AS 相当或略差

2. **极端低流动性**
   - 原因: 价差约束频繁触发，策略优势无法发挥
   - 预期: 所有策略表现都差

3. **无逆向选择的市场**
   - 原因: 逆向选择检测带来额外成本
   - 预期: 略逊于不检测的策略

### 关键发现解读

| 发现 | 解读 | 行动 |
|------|------|------|
| 强度-价差相关系数为负 | 策略正常工作，高强度时收窄价差 | 保持当前逻辑 |
| Adaptive 版本参数波动大 | 市场 regime 变化频繁或更新过于敏感 | 调整更新间隔 |
| 逆向选择准确率低 | 阈值设置不当或特征不足 | 优化检测算法 |
| 高聚类场景优势不明显 | Hawkes 参数与市场不匹配 | 重新标定参数 |

---

## 可视化方案

实验框架自动生成以下图表：

1. **PnL 轨迹对比** - 累计收益随时间变化
2. **回撤对比** - 回撤百分比面积图
3. **强度-价差关系** - 散点图 + 回归线
4. **库存变化** - 各策略持仓量对比
5. **综合评分雷达图** - 多维度策略评估

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 回测过慢 | 使用抽样数据，控制时间范围 |
| 参数过多 | 先做单参数敏感性分析 |
| 结果难以解释 | 添加详细的日志和可视化 |
| 真实数据不足 | 先用合成数据验证理论 |

---

## 参考文献

1. **Hawkes Process** - Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes.
2. **Avellaneda-Stoikov** - Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
3. **Market Microstructure** - Easley, D., et al. (2012). Flow toxicity and liquidity in a high-frequency world.

---

## 附录

### 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 实验框架 | `research/backtest/hawkes_comparison.py` | 核心代码 |
| Jupyter Notebook | `notebooks/06_hawkes_backtest_comparison.ipynb` | 交互式实验 |
| 报告模板 | `research/reports/hawkes_comparison_report.md` | 报告模板 |
| 单元测试 | `tests/test_hawkes_comparison.py` | 测试代码 |

### 相关策略文件

| 策略 | 路径 |
|------|------|
| Hawkes MM | `strategies/market_making/hawkes_mm.py` |
| AS | `strategies/market_making/avellaneda_stoikov.py` |
| XGBoost | `strategies/market_making/xgboost_spread.py` |
| Naive | `strategies/market_making/naive.py` |

---

*文档版本: 1.1*

*最后更新: 2026-02-19*
