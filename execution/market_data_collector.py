"""Container entrypoint for market data collector service."""

import asyncio
import os

from execution.service_runner import run_service


def main() -> None:
    port = int(os.getenv("MARKET_DATA_COLLECTOR_PORT", "8082"))
    asyncio.run(run_service("market-data-collector", default_port=port))


if __name__ == "__main__":
    main()
