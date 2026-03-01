# API 快速参考（精简版）

## 1. 核心数据类型

定义位置：`core/types.py`

- `MarketState`: 市场状态快照
- `OrderBook`: 订单簿结构
- `Position`: 持仓状态
- `QuoteAction`: 策略输出报价
- `Greeks`: 风险敏感度

验证入口：`core/validation/schemas.py`, `core/validation/validators.py`

## 2. 数据层

- 下载器：`data/downloaders/deribit.py`, `data/downloaders/okx.py`
- 流式：`data/streaming.py`
- 缓存：`data/cache.py`, `data/duckdb_cache.py`, `data/redis_cache.py`
- 集成管理：`data/integrated_manager.py`

## 3. 研究层

- 定价：`research/pricing/*`
- 波动率：`research/volatility/*`
- 风险：`research/risk/*`
- 信号：`research/signals/*`
- 回测：`research/backtest/*`
- 对冲/执行：`research/hedging/*`, `research/execution/*`

## 4. 策略接口

统一抽象在：`strategies/base.py`

关键方法：

- `quote(market_state, position)`
- `reset()`
- `train(...)`（可选）
- `on_fill(...)`（可选）

策略实现：

- 做市：`strategies/market_making/*`
- 套利：`strategies/arbitrage/*`

## 5. 服务入口

统一入口：

```bash
python -m execution.service_runner
```

环境变量：

- `SERVICE_NAME=trading-engine|risk-monitor|market-data-collector`
- 端口变量：`TRADING_ENGINE_PORT`, `RISK_MONITOR_PORT`, `MARKET_DATA_COLLECTOR_PORT`

研究看板：

```bash
python -m execution.research_dashboard
```

## 6. 常用命令

```bash
pytest -q -m "not integration"
make docs-link-check
make complexity-audit
make weekly-operating-audit
```

## 7. 相关文档

- `README.md`
- `docs/architecture.md`
- `docs/theory.md`
- `docs/deployment.md`
- `docs/governance-operations.md`
