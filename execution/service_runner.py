"""Shared runtime loop and CLI entrypoint for deployment services."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from core.health_server import HealthServer, default_checks

logger = logging.getLogger(__name__)
SERVICE_DEFAULTS = {
    "trading-engine": ("TRADING_ENGINE_PORT", 8080),
    "risk-monitor": ("RISK_MONITOR_PORT", 8081),
    "market-data-collector": ("MARKET_DATA_COLLECTOR_PORT", 8082),
}
SERVICE_TYPE_TO_NAME = {
    "trading": "trading-engine",
    "risk": "risk-monitor",
    "market_data": "market-data-collector",
}


def _health_port(default_port: int) -> int:
    raw = os.getenv("HEALTH_PORT")
    if raw is None:
        return default_port
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid HEALTH_PORT value %r, fallback to %s", raw, default_port)
        return default_port


async def run_service(service_name: str, default_port: int = 8080) -> None:
    """Run long-lived service with health endpoints and graceful shutdown."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    default_checks()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Fallback for environments without signal handler support.
            pass

    host = os.getenv("HEALTH_HOST", "0.0.0.0")
    port = _health_port(default_port)
    async with HealthServer(host=host, port=port, service_name=service_name):
        logger.info("%s started", service_name)
        await stop_event.wait()

    logger.info("%s stopped", service_name)


def _resolve_service_name() -> str:
    if service_name := os.getenv("SERVICE_NAME"):
        return service_name
    return SERVICE_TYPE_TO_NAME.get(os.getenv("SERVICE_TYPE", "trading"), "trading-engine")


def main() -> None:
    service_name = _resolve_service_name()
    env_key, default_port = SERVICE_DEFAULTS.get(service_name, ("HEALTH_PORT", 8080))
    port = int(os.getenv(env_key, str(default_port)))
    asyncio.run(run_service(service_name, default_port=port))


if __name__ == "__main__":
    main()
