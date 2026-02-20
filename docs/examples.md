# CORP 使用示例合集

> 合并文档：缓存使用 + Deribit Greeks + OKX 波动率

---

# 第1部分：缓存层使用示例

## 1.1 快速开始

### 使用集成数据管理器（推荐）

```python
import asyncio
from data.integrated_manager import IntegratedDataManager

async def main():
    async with IntegratedDataManager(
        duckdb_path="data/analytics.db",
        redis_host="localhost",
        redis_port=6379,
        enable_redis=True,
        enable_duckdb=True
    ) as manager:
        await process_data(manager)

asyncio.run(main())
```

## 1.2 Parquet 文件缓存

```python
from datetime import datetime, timedelta
from data.cache import DataCache

cache = DataCache()

# 存储数据
cache.store_market_data(
    exchange="deribit",
    instrument="BTC-PERPETUAL",
    data=df,
    date=datetime(2024, 1, 1)
)

# 读取数据
df = cache.get_market_data(
    exchange="deribit",
    instrument="BTC-PERPETUAL",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
```

## 1.3 DuckDB 分析查询

```python
# 复杂分析查询
result = manager.duckdb_cache.query_market_data(
    """
    SELECT
        date,
        AVG(close) as avg_close,
        STDDEV(close) as volatility
    FROM market_data
    WHERE instrument = 'BTC-PERPETUAL'
      AND date BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY date
    ORDER BY date
    """
)
```

---

# 第2部分：Deribit Greeks 与隐含波动率计算示例

## 2.1 获取期权数据并计算 Greeks

```python
import asyncio
from data.downloaders.deribit import DeribitClient

async def main():
    async with DeribitClient() as client:
        # 获取期权合约
        options = await client.get_instruments("BTC", "option")

        # 获取订单簿
        option = options[0]
        ob = await client.get_order_book(option.instrument_name)

        # 计算隐含波动率
        from research.volatility import implied_volatility

        iv = implied_volatility(
            market_price=ob.mark_price,
            S=option.underlying_price,
            K=option.strike,
            T=option.days_to_expiry / 365,
            r=0.05,
            is_call=option.option_type == "call"
        )
        print(f"隐含波动率: {iv:.2%}")

asyncio.run(main())
```

## 2.2 计算组合 Greeks

```python
from core.types import Greeks, Position

# 多个头寸的组合 Greeks
positions = [
    Position(instrument="BTC-24JAN24-50000-C", size=1.0, avg_entry_price=1000),
    Position(instrument="BTC-24JAN24-55000-C", size=-0.5, avg_entry_price=500),
]

# 汇总 Greeks
total_greeks = Greeks(
    delta=sum(p.size * g.delta for p, g in zip(positions, greeks_list)),
    gamma=sum(p.size * g.gamma for p, g in zip(positions, greeks_list)),
    theta=sum(p.size * g.theta for p, g in zip(positions, greeks_list)),
    vega=sum(p.size * g.vega for p, g in zip(positions, greeks_list)),
)

print(f"组合 Delta: {total_greeks.delta:.4f}")
```

---

# 第3部分：OKX 波动率计算示例

## 3.1 获取历史数据计算实现波动率

```python
import asyncio
from data.downloaders.okx import OKXClient
from research.volatility import yang_zhang_volatility

async def main():
    async with OKXClient() as client:
        # 获取历史K线
        klines = await client.get_klines(
            "BTC-USD-SWAP",
            bar="1H",
            limit=500
        )

        # 提取 OHLC
        opens = [k.open for k in klines]
        highs = [k.high for k in klines]
        lows = [k.low for k in klines]
        closes = [k.close for k in klines]

        # 计算 Yang-Zhang 波动率
        vol = yang_zhang_volatility(
            opens, highs, lows, closes,
            annualize=True
        )
        print(f"年化波动率: {vol:.2%}")

asyncio.run(main())
```

## 3.2 多时间窗口波动率对比

```python
from research.volatility import (
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility
)

# 不同估计方法对比
methods = {
    "Close-to-Close": realized_volatility(returns),
    "Parkinson": parkinson_volatility(highs, lows),
    "Garman-Klass": garman_klass_volatility(opens, highs, lows, closes),
    "Yang-Zhang": yang_zhang_volatility(opens, highs, lows, closes),
}

for name, vol in methods.items():
    print(f"{name}: {vol:.2%}")
```

---

# 附录：完整示例代码

完整代码见：
- `tests/test_volatility.py` - 波动率计算测试
- `tests/test_pricing.py` - 定价与 Greeks 测试
- `notebooks/` - 交互式示例

---

*本文档由3个示例文档合并而成*
