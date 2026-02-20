"""Tests for execution service entrypoint default ports."""

import execution.market_data_collector as market_data_collector
import execution.risk_monitor as risk_monitor
import execution.trading_engine as trading_engine


def test_trading_engine_default_port(monkeypatch):
    captured = {}
    monkeypatch.delenv("TRADING_ENGINE_PORT", raising=False)
    monkeypatch.setattr(
        trading_engine,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        trading_engine.asyncio, "run", lambda value: captured.setdefault("value", value)
    )

    trading_engine.main()

    assert captured["value"] == ("trading-engine", 8080)


def test_risk_monitor_default_port(monkeypatch):
    captured = {}
    monkeypatch.delenv("RISK_MONITOR_PORT", raising=False)
    monkeypatch.setattr(
        risk_monitor,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        risk_monitor.asyncio, "run", lambda value: captured.setdefault("value", value)
    )

    risk_monitor.main()

    assert captured["value"] == ("risk-monitor", 8081)


def test_market_data_collector_default_port(monkeypatch):
    captured = {}
    monkeypatch.delenv("MARKET_DATA_COLLECTOR_PORT", raising=False)
    monkeypatch.setattr(
        market_data_collector,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        market_data_collector.asyncio,
        "run",
        lambda value: captured.setdefault("value", value),
    )

    market_data_collector.main()

    assert captured["value"] == ("market-data-collector", 8082)
