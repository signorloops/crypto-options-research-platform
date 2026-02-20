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
# 全部测试
pytest tests/ -v

# 特定模块
pytest tests/test_volatility.py -v

# 覆盖率报告
pytest tests/ --cov=core --cov=strategies --cov=research
```

### 代码格式化
```bash
black .
ruff check . --fix
mypy .
```

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
