"""Container entrypoint for risk monitor service."""

import asyncio
import os

from execution.service_runner import run_service


def main() -> None:
    port = int(os.getenv("RISK_MONITOR_PORT", "8081"))
    asyncio.run(run_service("risk-monitor", default_port=port))


if __name__ == "__main__":
    main()
