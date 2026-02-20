"""
Health check server for Kubernetes/Docker deployments.

Provides HTTP endpoints for liveness and readiness probes.
"""

import asyncio
import inspect
import logging
import os
import resource
import sys
import time
from contextlib import suppress
from typing import Awaitable, Callable, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

CheckFunc = Callable[[], Union[bool, Awaitable[bool]]]

# Global health check functions registry
_health_checks: Dict[str, CheckFunc] = {}
_readiness_checks: Dict[str, CheckFunc] = {}

HTTP_REQUESTS_TOTAL = Counter(
    "corp_http_requests_total",
    "Total HTTP requests handled by the health server",
    ("service", "method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "corp_http_request_duration_seconds",
    "HTTP request latency for the health server",
    ("service", "method", "path"),
)
HEALTH_CHECK_STATUS = Gauge(
    "corp_health_check_status",
    "Health/readiness check status (1=healthy, 0=unhealthy)",
    ("service", "kind", "check"),
)
SERVICE_UP = Gauge(
    "corp_service_up",
    "Whether the health server is running (1=up, 0=down)",
    ("service",),
)


def register_health_check(name: str, check_func: CheckFunc) -> None:
    """Register a health check function."""
    _health_checks[name] = check_func


def register_readiness_check(name: str, check_func: CheckFunc) -> None:
    """Register a readiness check function."""
    _readiness_checks[name] = check_func


async def _run_check(check_func: CheckFunc) -> bool:
    """Run sync/async check function uniformly."""
    result = check_func()
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


def _memory_usage_mb() -> float:
    """Get process RSS in MB across Linux/macOS."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _memory_limit_mb() -> float:
    """Resolve memory limit from env, defaulting to 4GiB."""
    raw = os.getenv("MEMORY_LIMIT_MB", "4096")
    try:
        return max(float(raw), 1.0)
    except ValueError:
        logger.warning("Invalid MEMORY_LIMIT_MB=%r, fallback to 4096", raw)
        return 4096.0


def _memory_healthy() -> bool:
    """Check memory usage against configured process limit."""
    usage_mb = _memory_usage_mb()
    limit_mb = _memory_limit_mb()
    threshold_ratio_raw = os.getenv("MEMORY_HEALTH_THRESHOLD", "0.95")
    try:
        threshold_ratio = min(max(float(threshold_ratio_raw), 0.1), 1.0)
    except ValueError:
        threshold_ratio = 0.95
    return usage_mb <= limit_mb * threshold_ratio


def _connection_target(
    url_env: str,
    host_env: str,
    port_env: str,
    default_port: int,
) -> Optional[Tuple[str, int]]:
    """Resolve host/port from URL first, then host+port env variables."""
    raw_url = os.getenv(url_env)
    if raw_url:
        parsed = urlparse(raw_url)
        if parsed.hostname:
            return parsed.hostname, parsed.port or default_port
        logger.warning("Unable to parse %s=%r", url_env, raw_url)

    host = os.getenv(host_env)
    if not host:
        return None

    port_raw = os.getenv(port_env, str(default_port))
    try:
        port = int(port_raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %s", port_env, port_raw, default_port)
        port = default_port
    return host, port


async def _tcp_connectivity_check(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    """Attempt TCP connection to verify service reachability."""
    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host=host, port=port),
            timeout=timeout_seconds,
        )
        return True
    except (ConnectionError, OSError, asyncio.TimeoutError):
        return False
    finally:
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


async def _redis_healthy() -> bool:
    target = _connection_target("REDIS_URL", "REDIS_HOST", "REDIS_PORT", 6379)
    if target is None:
        return True
    host, port = target
    return await _tcp_connectivity_check(host, port)


async def _database_healthy() -> bool:
    target = _connection_target("DB_URL", "DB_HOST", "DB_PORT", 5432)
    if target is None:
        return True
    host, port = target
    return await _tcp_connectivity_check(host, port)


def create_health_app(service_name: str = "trading-engine") -> FastAPI:
    """Create FastAPI app with health endpoints."""
    app = FastAPI(title=f"CORP {service_name} Health")

    @app.middleware("http")
    async def instrument_requests(request, call_next):
        path = request.url.path
        method = request.method
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(
                service=service_name,
                method=method,
                path=path,
                status=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                service=service_name,
                method=method,
                path=path,
            ).observe(duration)

    @app.get("/health", response_class=JSONResponse)
    async def health() -> Dict:
        """Liveness probe - basic service health."""
        checks: Dict[str, bool] = {}
        for name, func in _health_checks.items():
            try:
                checks[name] = await _run_check(func)
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                checks[name] = False
            HEALTH_CHECK_STATUS.labels(service_name, "health", name).set(1 if checks[name] else 0)

        overall_healthy = all(checks.values()) if checks else True
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": service_name,
            "checks": checks,
        }

    @app.get("/ready", response_class=JSONResponse)
    async def readiness() -> Dict:
        """Readiness probe - check if ready to serve traffic."""
        failed = []

        for name, check_func in _readiness_checks.items():
            try:
                if not await _run_check(check_func):
                    failed.append(name)
            except Exception as e:
                logger.error(f"Readiness check '{name}' failed: {e}")
                failed.append(name)
                HEALTH_CHECK_STATUS.labels(service_name, "readiness", name).set(0)
            else:
                HEALTH_CHECK_STATUS.labels(service_name, "readiness", name).set(1)

        if failed:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "not_ready", "failed_checks": failed},
            )

        return {
            "status": "ready",
            "service": service_name,
            "checks": list(_readiness_checks.keys()),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


class HealthServer:
    """Async health check server using FastAPI."""

    def __init__(self, host: str = None, port: int = None, service_name: str = "trading-engine"):
        self.host = host or os.getenv("HEALTH_HOST", "0.0.0.0")
        self.port = port or int(os.getenv("HEALTH_PORT", "8080"))
        self.service_name = service_name
        self.app = create_health_app(service_name=self.service_name)
        self._server: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the health server."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=os.getenv("HEALTH_LOG_LEVEL", "warning"),
            access_log=False,
        )
        server = uvicorn.Server(config)

        self._server = asyncio.create_task(server.serve())
        SERVICE_UP.labels(service=self.service_name).set(1)
        logger.info(f"Health server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the health server."""
        if self._server:
            self._server.cancel()
            try:
                await self._server
            except asyncio.CancelledError:
                pass
        SERVICE_UP.labels(service=self.service_name).set(0)
        logger.info("Health server stopped")

    async def __aenter__(self) -> "HealthServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def default_checks() -> None:
    """Register default health checks."""
    register_health_check("memory", _memory_healthy)
    register_health_check("redis", _redis_healthy)
    register_health_check("database", _database_healthy)
    register_readiness_check("redis", _redis_healthy)
    register_readiness_check("database", _database_healthy)
