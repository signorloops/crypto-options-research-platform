# API 文档

本文档聚焦当前主干实现（Deribit / OKX / 研究模型 / 策略 / 回测）。

## 1. 数据下载器

### 1.1 DeribitClient

```python
from data.downloaders.deribit import DeribitClient

client = DeribitClient()
```

常用方法：

- `get_instruments(currency="BTC", instrument_type="option")`
- `get_order_book(instrument, depth=10)`
- `get_tick(instrument)`
- `get_trades(instrument, start, end, limit=1000)`
- `get_ticker(instrument)`
- `get_option_greeks(instrument)`
- `get_option_iv(instrument)`

### 1.2 OKXClient

```python
from data.downloaders.okx import OKXClient

client = OKXClient()
```

常用方法：

- `get_instruments(instrument_type="OPTION")`
- `get_option_instruments(underlying="BTC-USD")`
- `get_order_book(instrument, depth=10)`
- `get_ticker(instrument)`
- `get_klines(instrument, interval="1H", start=None, end=None)`

## 2. 波动率与定价

### 2.1 历史波动率

```python
import numpy as np
from research.volatility.historical import realized_volatility, yang_zhang_volatility

returns = np.random.normal(0, 0.02, 200)
rv = realized_volatility(returns, annualize=True, periods=365)
```

### 2.2 条件波动率模型

```python
from research.volatility.models import ewma_volatility, garch_volatility, har_volatility

ewma = ewma_volatility(returns)
garch = garch_volatility(returns, omega=1e-6, alpha=0.1, beta=0.85)
```

### 2.3 隐含波动率

```python
from research.volatility.implied import implied_volatility

iv = implied_volatility(
    market_price=10.5,
    S=50000,
    K=50000,
    T=30/365,
    r=0.03,
    is_call=True,
    method="hybrid",
)
```

### 2.4 Inverse 定价

```python
from research.pricing.inverse_options import InverseOptionPricer

price = InverseOptionPricer.calculate_price(
    S=50000, K=52000, T=30/365, r=0.03, sigma=0.6, option_type="call"
)

greeks = InverseOptionPricer.calculate_greeks(
    S=50000, K=52000, T=30/365, r=0.03, sigma=0.6, option_type="call"
)
```

## 3. 风险模块

### 3.1 Greeks 风险分析

```python
from research.risk.greeks import GreeksRiskAnalyzer

analyzer = GreeksRiskAnalyzer(risk_free_rate=0.03)
```

### 3.2 VaR / CVaR

```python
from research.risk.var import VaRCalculator

calc = VaRCalculator(confidence_level=0.95)
res = calc.parametric_var(positions_df, returns_df)
```

其他方法：

- `historical_var`
- `filtered_historical_var`
- `cornish_fisher_var`
- `evt_var`
- `monte_carlo_var`

### 3.3 Circuit Breaker

```python
from research.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

cb = CircuitBreaker(CircuitBreakerConfig())
state = cb.check_risk_limits(portfolio_state)
can_trade, reason = cb.can_trade(action)
```

## 4. 策略接口

所有做市策略实现统一基类：

```python
from strategies.base import MarketMakingStrategy

# 必须实现
# quote(state, position) -> QuoteAction
# get_internal_state() -> dict
```

### 4.1 做市策略

- `strategies.market_making.naive.NaiveMarketMaker`
- `strategies.market_making.avellaneda_stoikov.AvellanedaStoikov`
- `strategies.market_making.hawkes_mm.HawkesMarketMaker`
- `strategies.market_making.hawkes_mm.AdaptiveHawkesMarketMaker`
- `strategies.market_making.integrated_strategy.IntegratedMarketMakingStrategy`
- `strategies.market_making.fast_integrated_strategy.FastIntegratedMarketMakingStrategy`
- `strategies.market_making.xgboost_spread.XGBoostSpreadStrategy`
- `strategies.market_making.ppo_agent.PPOMarketMaker`

### 4.2 套利策略

- `CrossExchangeArbitrage`
- `BasisArbitrage`
- `ConversionArbitrage`
- `OptionBoxArbitrage`

跨交易所套利示例：

```python
from strategies.arbitrage.cross_exchange import CrossExchangeArbitrage, ExchangeFees

arb = CrossExchangeArbitrage(min_spread_bps=80, min_profit_pct=0.2)
arb.set_exchange_fees("okx", ExchangeFees(0.001, 0.001))
arb.set_exchange_fees("deribit", ExchangeFees(0.0005, 0.0005))

arb.update_price("okx", "BTC", 50000.0)
arb.update_price("deribit", "BTC", 50300.0)
```

## 5. 回测与比较

### 5.1 回测引擎

```python
from research.backtest.engine import BacktestEngine
from strategies.market_making.naive import NaiveMarketMaker

engine = BacktestEngine(strategy=NaiveMarketMaker())
result = engine.run(market_data_df)
print(result.summary())
```

### 5.2 多策略对比

```python
from research.backtest.arena import StrategyArena

arena = StrategyArena(market_data_df, initial_capital=100000)
comparison_df = arena.run_tournament(strategies)
```

### 5.3 Hawkes 专项对比

```python
from research.backtest.hawkes_comparison import ScenarioGenerator, ComprehensiveHawkesComparison

gen = ScenarioGenerator()
scenarios = gen.generate_hawkes_scenarios()

cmp = ComprehensiveHawkesComparison(initial_capital=100000)
results = cmp.run_full_comparison(strategies, scenarios)
```

## 6. 数据验证

### 6.1 Pydantic Schema

```python
from core.validation.schemas import TickData, BacktestConfig
```

### 6.2 自定义验证器

```python
from core.validation.validators import validate_price, validate_instrument_name
```

## 7. 研究看板

```bash
uvicorn execution.research_dashboard:app --host 0.0.0.0 --port 8501
```

- `/`：HTML 看板
- `/api/files`：可用结果文件
- `/health`：健康检查

## 8. 相关文档

- `docs/theory.md`
- `docs/architecture.md`
