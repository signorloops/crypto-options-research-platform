# 项目全景图（Mermaid 学习版）

> 目标：用一套 Mermaid 图帮助你从“整体 -> 模块 -> 调用链 -> 测试”快速建立项目心智模型。  
> 范围：`corp/` 下核心代码与主流程，不包含 `venv/`、临时缓存与归档文档。

---

## 1) 全局分层总览

```mermaid
flowchart TB
    Data["Data Layer<br/>downloaders / streaming / cache"] --> Core["Core Layer<br/>types / validation / exceptions"]
    Core --> Research["Research Layer<br/>pricing / volatility / risk / signals / backtest"]
    Research --> Strategy["Strategy Layer<br/>market_making / arbitrage"]
    Strategy --> Execution["Execution Layer<br/>trading_engine / risk_monitor / dashboard"]
    Execution --> Eval["Evaluation & Ops<br/>tests / notebooks / validation_scripts"]

    Data -. "market snapshots" .-> Strategy
    Research -. "models & features" .-> Execution
    Eval -. "feedback loop" .-> Research
```

---

## 2) 目录结构图（核心）

```mermaid
flowchart LR
    corp["corp/"]

    corp --> core["core/"]
    core --> core_types["types.py"]
    core --> core_validation["validation/"]
    core --> core_exc["exceptions.py"]
    core --> core_health["health_server.py"]

    corp --> data["data/"]
    data --> data_dl["downloaders/<br/>deribit.py / okx.py"]
    data --> data_stream["streaming.py"]
    data --> data_cache["cache.py / duckdb_cache.py / redis_cache.py"]
    data --> data_gen["generators/<br/>synthetic.py / hawkes.py"]
    data --> data_ob["orderbook_reconstructor.py"]

    corp --> research["research/"]
    research --> r_pricing["pricing/<br/>inverse_options.py / rough_volatility.py"]
    research --> r_vol["volatility/<br/>historical.py / models.py / implied.py"]
    research --> r_risk["risk/<br/>greeks.py / var.py / circuit_breaker.py"]
    research --> r_signals["signals/<br/>regime_detector.py / fast_regime_detector.py"]
    research --> r_bt["backtest/<br/>engine.py / arena.py / hawkes_comparison.py"]
    research --> r_hedge["hedging/<br/>adaptive_delta.py / deep_hedging.py"]
    research --> r_exec["execution/<br/>almgren_chriss.py"]

    corp --> strategies["strategies/"]
    strategies --> s_mm["market_making/<br/>naive / AS / hawkes / integrated / ppo / xgboost"]
    strategies --> s_arb["arbitrage/<br/>cross_exchange / basis / conversion / option_box"]

    corp --> execution["execution/"]
    execution --> e_engine["trading_engine.py"]
    execution --> e_risk["risk_monitor.py"]
    execution --> e_dash["research_dashboard.py"]
    execution --> e_runner["service_runner.py"]

    corp --> tests["tests/"]
    corp --> docs["docs/"]
    corp --> notebooks["notebooks/"]
```

---

## 3) 交易与回测主调用链

```mermaid
flowchart TD
    A["Exchange APIs<br/>Deribit / OKX"] --> B["data.downloaders + data.streaming"]
    B --> C["MarketState build<br/>core.types"]
    C --> D["strategy.quote(...)"]

    D --> D1["market making<br/>naive / AS / hawkes / integrated"]
    D --> D2["arbitrage<br/>cross_exchange / basis / conversion / option_box"]

    D1 --> E["research.backtest.engine<br/>RealisticFillSimulator"]
    D2 --> E
    E --> F["Portfolio update<br/>Position / Fill / PnL"]
    F --> G["Risk checks<br/>circuit_breaker + var + greeks"]
    G --> H["Scorecards<br/>arena / hawkes_comparison"]
    H --> I["Output<br/>results + dashboard + reports"]
```

---

## 4) 算法模块关系图

```mermaid
flowchart LR
    pricing["pricing.inverse_options"] --> risk["risk.greeks / risk.var"]
    vol["volatility.historical / models / implied"] --> strategy_as["AS / Integrated"]
    signals["signals.regime_detector"] --> strategy_int["Integrated Strategy"]
    hedge["hedging.adaptive_delta"] --> strategy_int
    micro["microstructure.vpin / orderbook_features"] --> strategy_hawkes["Hawkes MM"]
    hawkes_gen["data.generators.hawkes"] --> strategy_hawkes
    strategy_as --> backtest["backtest.engine / arena"]
    strategy_hawkes --> backtest
    strategy_int --> backtest
    risk --> execution["execution.risk_monitor / trading_engine"]
```

---

## 5) 测试覆盖映射

```mermaid
flowchart TB
    tests["tests/"] --> t_mm["test_strategies.py<br/>test_integrated_strategy.py<br/>test_fast_integrated_strategy.py"]
    tests --> t_risk["test_risk.py<br/>test_circuit_breaker.py"]
    tests --> t_bt["test_backtest.py<br/>test_strategy_arena.py<br/>test_hawkes_comparison.py"]
    tests --> t_data["test_data_downloaders.py<br/>test_streaming.py<br/>test_cache*.py"]
    tests --> t_core["test_core_types.py<br/>test_validation.py"]
    tests --> t_vol["test_volatility.py<br/>test_volatility_models.py"]
    tests --> t_price["test_pricing_inverse.py<br/>test_pricing_boundary.py"]

    t_mm --> target_mm["strategies/market_making/*"]
    t_risk --> target_risk["research/risk/*"]
    t_bt --> target_bt["research/backtest/*"]
    t_data --> target_data["data/*"]
    t_core --> target_core["core/*"]
    t_vol --> target_vol["research/volatility/*"]
    t_price --> target_price["research/pricing/*"]
```

---

## 6) 学习顺序建议

```mermaid
flowchart LR
    S1["Step 1<br/>README + GUIDE"] --> S2["Step 2<br/>core/types.py + data/streaming.py"]
    S2 --> S3["Step 3<br/>research/backtest/engine.py"]
    S3 --> S4["Step 4<br/>AS + Hawkes + Integrated strategy"]
    S4 --> S5["Step 5<br/>risk.var + circuit_breaker + regime_detector"]
    S5 --> S6["Step 6<br/>tests + notebooks + dashboard"]
```

