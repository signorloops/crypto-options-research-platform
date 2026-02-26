"""Tests for execution service entrypoint default ports."""

import importlib

import execution.service_runner as service_runner


def test_trading_engine_default_port(monkeypatch):
    captured = {}
    monkeypatch.setenv("SERVICE_NAME", "trading-engine")
    monkeypatch.delenv("TRADING_ENGINE_PORT", raising=False)
    monkeypatch.setattr(
        service_runner,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        service_runner.asyncio, "run", lambda value: captured.setdefault("value", value)
    )

    service_runner.main()

    assert captured["value"] == ("trading-engine", 8080)


def test_risk_monitor_default_port(monkeypatch):
    captured = {}
    monkeypatch.setenv("SERVICE_NAME", "risk-monitor")
    monkeypatch.delenv("RISK_MONITOR_PORT", raising=False)
    monkeypatch.setattr(
        service_runner,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        service_runner.asyncio, "run", lambda value: captured.setdefault("value", value)
    )

    service_runner.main()

    assert captured["value"] == ("risk-monitor", 8081)


def test_market_data_collector_default_port(monkeypatch):
    captured = {}
    monkeypatch.setenv("SERVICE_NAME", "market-data-collector")
    monkeypatch.delenv("MARKET_DATA_COLLECTOR_PORT", raising=False)
    monkeypatch.setattr(
        service_runner,
        "run_service",
        lambda service_name, default_port: (service_name, default_port),
    )
    monkeypatch.setattr(
        service_runner.asyncio,
        "run",
        lambda value: captured.setdefault("value", value),
    )

    service_runner.main()

    assert captured["value"] == ("market-data-collector", 8082)


def test_legacy_entrypoint_modules_resolve_from_execution_package():
    assert importlib.import_module("execution.trading_engine").main is not None
    assert importlib.import_module("execution.risk_monitor").main is not None
    assert importlib.import_module("execution.market_data_collector").main is not None
