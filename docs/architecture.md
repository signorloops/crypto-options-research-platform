# 架构总览（精简版）

## 1. 分层结构

```text
data -> core -> research -> strategies -> execution
```

- `data/`: 下载、流式、缓存与重建
- `core/`: 类型、异常、验证、健康检查
- `research/`: 定价、波动率、风险、信号、回测
- `strategies/`: 做市与套利策略
- `execution/`: 统一服务入口与研究看板

## 2. 主链路

1. 从交易所获取行情并构建 `MarketState`。
2. 策略层输出 `QuoteAction`。
3. 回测/执行层处理成交与仓位更新。
4. 风险层计算 Greeks、VaR 与熔断信号。
5. 产出审计报告、周治理产物和可视化结果。

## 3. 关键模块

数据与缓存：

- `data/downloaders/*.py`
- `data/streaming.py`
- `data/cache.py`, `data/duckdb_cache.py`, `data/redis_cache.py`

回测与评测：

- `research/backtest/engine.py`
- `research/backtest/arena.py`
- `research/backtest/hawkes_comparison.py`

风险与风控：

- `research/risk/greeks.py`
- `research/risk/var.py`
- `research/risk/circuit_breaker.py`

## 4. 策略框架

统一基类：`strategies/base.py`

做市策略目录：`strategies/market_making/`
套利策略目录：`strategies/arbitrage/`

策略设计约束：

1. 接口统一，便于回测对比。
2. 风险约束前置。
3. 研究逻辑与执行逻辑解耦。

## 5. 运维与治理

统一服务入口：

```bash
python -m execution.service_runner
```

建议例行检查：

```bash
make check-service-entrypoint
make docs-link-check
make complexity-audit
make weekly-operating-audit
```

## 6. 相关文档

- `docs/quickstart.md`
- `docs/api.md`
- `docs/theory.md`
- `docs/deployment.md`
- `docs/governance-operations.md`
