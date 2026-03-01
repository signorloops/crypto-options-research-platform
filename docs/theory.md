# CORP 理论手册（精简版）

## 1. Inverse 期权核心

币本位合约核心特征：USD 报价、币本位结算。

PnL 形式：

```text
PnL_inverse = size * (1/P_entry - 1/P_exit)
```

定价与 Greeks 入口：

- `research/pricing/inverse_options.py`
- `research/pricing/inverse_power_options.py`
- `research/pricing/quanto_inverse_power.py`

## 2. 波动率与曲面

核心模块：

- 历史波动率：`research/volatility/historical.py`
- 条件波动率：`research/volatility/models.py`
- IV 求解与曲面：`research/volatility/implied.py`, `research/volatility/iv_solvers.py`

重点原则：

1. 短期限稳定性优先。
2. 无套利约束必须可检测。
3. 曲面质量以审计产物衡量，而非单次回测结果。

## 3. 策略与风险联动

策略层：

- 做市：`strategies/market_making/*`
- 套利：`strategies/arbitrage/*`

风险层：

- Greeks 聚合：`research/risk/greeks.py`
- VaR/CVaR：`research/risk/var.py`
- 熔断器：`research/risk/circuit_breaker.py`

回测层：

- `research/backtest/engine.py`
- `research/backtest/arena.py`
- `research/backtest/hawkes_comparison.py`

## 4. Hawkes 核心要点

强度函数：

```text
lambda(t) = mu + sum(alpha * exp(-beta * (t - t_i)))
```

工程关注点：

1. 聚类强度估计稳定性。
2. 强度驱动价差/偏斜控制。
3. 与库存风险约束协同，不单独追求成交率。

## 5. 最小验证流程

```bash
pytest -q -m "not integration"
make complexity-audit
make weekly-operating-audit
```

研究审计与治理命令见：

- `docs/governance-operations.md`
