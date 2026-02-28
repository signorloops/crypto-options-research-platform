"""Runtime entrypoints for deployment services."""

from __future__ import annotations
import asyncio
import os
import sys
from types import ModuleType
from execution.service_runner import run_service


def _install_legacy_entrypoint(
    module_name: str, service_name: str, port_env: str, default_port: int
) -> None:
    full_name = f"{__name__}.{module_name}"
    module = ModuleType(full_name)
    module.run_service = run_service
    module.asyncio = asyncio

    def main() -> None:
        port = int(os.getenv(port_env, str(default_port)))
        module.asyncio.run(module.run_service(service_name, default_port=port))

    module.main = main
    sys.modules[full_name] = module
    setattr(sys.modules[__name__], module_name, module)


_install_legacy_entrypoint("trading_engine", "trading-engine", "TRADING_ENGINE_PORT", 8080)
_install_legacy_entrypoint("risk_monitor", "risk-monitor", "RISK_MONITOR_PORT", 8081)
_install_legacy_entrypoint(
    "market_data_collector", "market-data-collector", "MARKET_DATA_COLLECTOR_PORT", 8082
)
