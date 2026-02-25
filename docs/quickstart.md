# 快速开始指南

> 📚 **提示**: 如果你不确定从哪里开始，先阅读 [GUIDE.md](GUIDE.md) - 它根据你的背景提供了详细的学习路径推荐。

## 安装

### 环境要求
- Python 3.9+
- 4GB+ RAM
- 10GB+ 磁盘空间（数据缓存）

### 步骤 1: 克隆仓库
```bash
git clone <repository-url>
cd corp
```

### 步骤 2: 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 步骤 3: 安装依赖
```bash
pip install -e ".[dev]"
```

### 步骤 4: 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API keys（可选）
```

## 第一个程序

### 获取市场数据
```python
import asyncio
from data.downloaders.deribit import DeribitClient

async def main():
    client = DeribitClient()

    async with client:
        # 获取 BTC 期权合约列表
        contracts = await client.get_instruments("BTC", "option")
        print(f"找到 {len(contracts)} 个 BTC 期权合约")

        # 获取第一个合约的订单簿
        if contracts:
            symbol = contracts[0].symbol
            ob = await client.get_order_book(symbol)
            print(f"{symbol}: 买 {ob.best_bid} / 卖 {ob.best_ask}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 实时 WebSocket 流
```python
import asyncio
from data.downloaders.deribit import DeribitClient

async def on_trade(trade):
    print(f"成交: {trade.instrument} @ {trade.price} ({trade.side})")

async def main():
    client = DeribitClient()

    # 订阅 BTC 永续合约的实时交易
    await client.subscribe_trades(
        ["BTC-PERPETUAL"],
        callback=on_trade
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 波动率计算
```python
import numpy as np
from research.volatility import yang_zhang_volatility

# 模拟 OHLC 数据
open_p = np.array([50000, 51000, 50500, 52000])
high = np.array([52000, 53000, 52500, 54000])
low = np.array([49000, 50000, 49500, 51000])
close = np.array([51000, 50500, 52000, 53000])

# 计算 Yang-Zhang 波动率
vol = yang_zhang_volatility(open_p, high, low, close, annualize=True)
print(f"年化波动率: {vol*100:.2f}%")
```

### 简单回测
```python
from datetime import datetime, timedelta
from research.backtest.engine import BacktestEngine
from strategies.market_making.naive import NaiveMarketMaker

# 配置
config = {
    'start_date': datetime(2024, 1, 1),
    'end_date': datetime(2024, 1, 31),
    'instruments': ['BTC-PERPETUAL'],
    'initial_capital': 100000,
}

# 创建策略
strategy = NaiveMarketMaker(
    spread_bps=50,  # 50基点价差
    max_position=1.0
)

# 运行回测
engine = BacktestEngine(config)
results = engine.run(strategy)

print(f"最终权益: {results.final_equity:.2f}")
print(f"夏普比率: {results.sharpe_ratio:.2f}")
```

### Hawkes 策略对比实验
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

# 生成测试场景（合成 Hawkes 数据）
gen = ScenarioGenerator(base_price=50000.0)
scenarios = gen.generate_hawkes_scenarios()

# 初始化策略
strategies = [
    HawkesMarketMaker(HawkesMMConfig()),
    AvellanedaStoikov(ASConfig()),
]

# 运行对比实验
comparison = ComprehensiveHawkesComparison(
    initial_capital=100000.0,
    transaction_cost_bps=2.0
)
results = comparison.run_full_comparison(strategies, scenarios)

# 生成报告
report = comparison.generate_summary_report()
print(report)
```

#### 使用 Jupyter Notebook
```bash
# 启动交互式实验
jupyter notebook notebooks/06_hawkes_backtest_comparison.ipynb
```

## 常用命令

### 运行测试
```bash
# 默认测试（不包含外部 API 集成测试）
pytest tests/ -v -m "not integration"

# 特定模块
pytest tests/test_volatility.py -v

# 覆盖率报告
pytest tests/ -m "not integration" --cov=core --cov=strategies --cov=research

# 显式运行集成测试（访问交易所 API）
RUN_INTEGRATION_TESTS=1 pytest tests/ -v -m "integration"
```

### 代码格式化
```bash
black .
ruff check . --fix
mypy .
```

### 研究审计与基准
```bash
# 一键生成所有研究审计工件（推荐）
make research-audit

# 近月 IV surface 稳定性 + 静态无套利报告
python validation_scripts/iv_surface_stability_report.py

# 加质量门槛（CI 模式）
python validation_scripts/iv_surface_stability_report.py --fail-on-arbitrage --min-short-max-jump-reduction 0.005

# Rough volatility + jumps 实验对比
python validation_scripts/rough_jump_experiment.py --seed 42

# 定价模型 zoo 基准（固定样本，适合跨提交漂移比较）
python validation_scripts/pricing_model_zoo_benchmark.py --quotes-json validation_scripts/fixtures/model_zoo_quotes_seed42.json

# 同时输出机器可读 JSON
python validation_scripts/pricing_model_zoo_benchmark.py --quotes-json validation_scripts/fixtures/model_zoo_quotes_seed42.json --output-json artifacts/pricing-model-zoo-benchmark.json

# 同时输出 JSON + Markdown（便于 CI Summary/周报）
python validation_scripts/pricing_model_zoo_benchmark.py --quotes-json validation_scripts/fixtures/model_zoo_quotes_seed42.json --output-json artifacts/pricing-model-zoo-benchmark.json --output-md artifacts/pricing-model-zoo-benchmark.md

# 加质量门槛（期望最优模型 + RMSE 上限）
python validation_scripts/pricing_model_zoo_benchmark.py --quotes-json validation_scripts/fixtures/model_zoo_quotes_seed42.json --expected-best-model bates --max-best-rmse 120.0

# 若需要动态生成样本，也可使用 seed + bucket
python validation_scripts/pricing_model_zoo_benchmark.py --seed 42 --n-per-bucket 1

# inverse-power MC 基线与闭式 inverse 定价一致性验证
python validation_scripts/inverse_power_validation.py --n-paths 120000 --max-abs-error 0.0006
```

GitHub Actions:
- `Research Audit` workflow 每周一 UTC 自动运行，并可手动触发。
- `Research Audit Baseline Refresh` 可手动生成“候选基线 + 差异报告”artifact 供审阅。
- 手动触发时可调 `seed`、`n_per_bucket`、`quotes_json`、`expected_best_model`、`max_best_rmse`、`max_best_rmse_increase_pct`、`max_iv_reduction_drop_pct`、`allow_best_model_change`、`fail_on_arbitrage`、`min_short_max_jump_reduction`、`min_net_jump_premium_std`。
- 运行后可在 artifact 下载 `iv-surface-stability`（md/json）、`rough-jump`（txt）、`jump-premia-stability`（md/json）、`model-zoo`（txt/json/md）、`research-audit-snapshot.json`、`research-audit-drift-report`（md/json）、`research-audit-weekly-summary.md`。
- 如果你确认模型升级是预期行为，可本地执行 `make research-audit-refresh-baseline` 刷新基线。

### 数据管理
```bash
# 查看缓存信息
python -c "from data.cache import DataCache; print(DataCache().get_cache_info())"

# 清理过期缓存
# 手动删除 data/cache/ 下的旧文件
```

## 故障排除

### 问题: API 返回 429 Too Many Requests
**解决:** 降低请求频率，或添加 API key 提高限流

### 问题: WebSocket 连接断开
**解决:** 已实现自动重连，检查网络连接

### 问题: 内存不足
**解决:** 减少回测日期范围，或增加交换空间

### 问题: Pydantic 验证错误
**解决:** 检查输入数据格式，参考 `core/validation/schemas.py`

## 下一步

- 阅读 [架构文档](architecture.md) 了解系统设计
- 查看 [API 文档](api.md) 了解所有接口
- 探索 [示例 Notebook](../notebooks/)
- 了解 [Hawkes 策略对比实验](hawkes_comparison_experiment.md) 详细设计
- 按 [Branch Protection 清单](branch-protection-checklist.md) 配置主干合并守门
- 阅读 [Research Audit 指南](research-audit.md) 了解研究守门产物与阈值
- 阅读 [arXiv 前沿清单（币本位期权）](reports/2026-02-25-arxiv-frontier-inverse-options.md) 获取可落地论文路线图
- 参考 [arXiv 路线实施计划](plans/2026-02-25-inverse-options-arxiv-implementation-plan.md) 直接拆任务执行
