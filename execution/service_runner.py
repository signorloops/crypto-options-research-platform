"""Shared runtime loop for container service entrypoints."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from core.health_server import HealthServer, default_checks


logger = logging.getLogger(__name__)


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
