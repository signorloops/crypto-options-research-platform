"""Container entrypoint for trading engine service."""

import asyncio
import os

from execution.service_runner import run_service


def main() -> None:
    port = int(os.getenv("TRADING_ENGINE_PORT", "8080"))
    asyncio.run(run_service("trading-engine", default_port=port))


if __name__ == "__main__":
    main()

