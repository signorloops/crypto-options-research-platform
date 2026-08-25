"""Health check server with liveness/readiness/metrics endpoints."""

import asyncio
import ctypes
import ctypes.util
import inspect
import logging
import os
import resource
import sys
import time
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# uvicorn's uvicorn.config.STARTUP_FAILURE exit code; vendored here so the
# done-callback can recognise bind failures without importing uvicorn eagerly.
_UVICORN_STARTUP_FAILURE = 3

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


class _MachTaskBasicInfo(ctypes.Structure):
    """Subset of ``mach_task_basic_info`` used to read the resident size."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("suspend_count", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("deprecated", ctypes.c_int32),
        ("user_time", ctypes.c_uint32),
        ("system_time", ctypes.c_uint32),
    ]


_MACH_TASK_BASIC_INFO = 20
_MACH_TASK_BASIC_INFO_COUNT = ctypes.sizeof(_MachTaskBasicInfo) // 4
_mach_libc: Optional["ctypes.CDLL"] = None


def _load_mach_libc() -> Optional["ctypes.CDLL"]:
    """Load and configure libc for the Mach ``task_info`` call (macOS only)."""
    global _mach_libc
    if _mach_libc is not None:
        return _mach_libc
    if sys.platform != "darwin":
        return None
    try:
        name = ctypes.util.find_library("c") or "libc.dylib"
        libc = ctypes.CDLL(name)
        libc.mach_task_self.restype = ctypes.c_uint32
        libc.task_info.argtypes = [
            ctypes.c_uint32,  # target task port
            ctypes.c_uint32,  # flavor
            ctypes.POINTER(_MachTaskBasicInfo),
            ctypes.POINTER(ctypes.c_uint32),  # in/out count (natural_t items)
        ]
        libc.task_info.restype = ctypes.c_int
    except (OSError, ValueError, AttributeError):
        return None
    _mach_libc = libc
    return libc


def _read_mach_task_rss_bytes() -> Optional[int]:
    """Read current resident size via Mach ``task_info`` (macOS)."""
    libc = _load_mach_libc()
    if libc is None:
        return None
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(_MACH_TASK_BASIC_INFO_COUNT)
    try:
        result = libc.task_info(
            libc.mach_task_self(),
            ctypes.c_uint32(_MACH_TASK_BASIC_INFO),
            ctypes.byref(info),
            ctypes.byref(count),
        )
    except (OSError, ValueError, AttributeError):
        return None
    if result != 0:  # KERN_SUCCESS
        return None
    return int(info.resident_size)


def _read_proc_status_rss_kb() -> Optional[float]:
    """Read current RSS (``VmRSS``) in kB from ``/proc/self/status`` (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _memory_usage_mb() -> float:
    """Get the process's *current* RSS in MB across Linux/macOS.

    Peak RSS (``ru_maxrss``) must not drive the health check: a single
    transient memory spike would otherwise mark the service unhealthy forever,
    even after the memory has been released back to the OS. Current RSS is
    therefore read from ``/proc/self/status`` on Linux and from the Mach
    ``task_info`` call on macOS. ``ru_maxrss`` remains only as a last-resort
    fallback for platforms without either source, with the caveat that macOS
    reports it in bytes (peak) while Linux reports it in kilobytes.
    """
    rss_kb = _read_proc_status_rss_kb()
    if rss_kb is not None:
        return rss_kb / 1024.0

    rss_bytes = _read_mach_task_rss_bytes()
    if rss_bytes is not None:
        return rss_bytes / (1024.0 * 1024.0)

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


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


def _set_check_status_metric(service_name: str, kind: str, check_name: str, passed: bool) -> None:
    HEALTH_CHECK_STATUS.labels(service_name, kind, check_name).set(1 if passed else 0)


def _build_request_instrumenter(service_name: str):
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

    return instrument_requests


async def _collect_health_checks(service_name: str) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    for name, func in _health_checks.items():
        try:
            checks[name] = await _run_check(func)
        except Exception as exc:
            logger.error("Health check '%s' failed: %s", name, exc)
            checks[name] = False
        _set_check_status_metric(service_name, "health", name, checks[name])
    return checks


async def _collect_failed_readiness_checks(service_name: str) -> list[str]:
    failed: list[str] = []
    for name, check_func in _readiness_checks.items():
        try:
            passed = await _run_check(check_func)
        except Exception as exc:
            logger.error("Readiness check '%s' failed: %s", name, exc)
            passed = False
        _set_check_status_metric(service_name, "readiness", name, passed)
        if not passed:
            failed.append(name)
    return failed


def create_health_app(service_name: str = "trading-engine") -> FastAPI:
    """Create FastAPI app with health endpoints."""
    app = FastAPI(title=f"CORP {service_name} Health")
    app.middleware("http")(_build_request_instrumenter(service_name))

    @app.get("/health", response_class=JSONResponse)
    async def health() -> Dict:
        """Liveness probe - basic service health."""
        checks = await _collect_health_checks(service_name)
        overall_healthy = all(checks.values()) if checks else True
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": service_name,
            "checks": checks,
        }

    @app.get("/ready", response_class=JSONResponse)
    async def readiness() -> Dict:
        """Readiness probe - check if ready to serve traffic."""
        failed = await _collect_failed_readiness_checks(service_name)
        if failed: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "not_ready", "failed_checks": failed})
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
        self._uvicorn_server: Optional[Any] = None
        self._shutdown_event = asyncio.Event()
        self._startup_event = asyncio.Event()
        self._startup_error: Optional[BaseException] = None

    def _log_task_exception(self, task: "asyncio.Task") -> None:
        """Surface exceptions escaping the server task (they are otherwise never retrieved)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, SystemExit) and exc.code == _UVICORN_STARTUP_FAILURE:
            logger.error(
                "Health server failed to start on %s:%s (uvicorn startup failure, "
                "port may already be in use)",
                self.host,
                self.port,
            )
        else:
            logger.error("Health server task crashed: %r", exc, exc_info=exc)

    async def start(self) -> None:
        """Start the health server and wait until the port is actually bound.

        ``_startup_and_supervise`` runs uvicorn's ``startup()`` (which binds the
        socket) before entering the serve loop, so a bind failure — for example
        an already-taken port — raises ``SystemExit`` inside that coroutine,
        where it is converted into a ``RuntimeError`` instead of killing the
        process. ``start()`` blocks on ``_startup_event`` and only returns once
        binding succeeded; the done-callback makes any later crash visible.
        """
        self._startup_event.clear()
        self._startup_error = None
        self._server = asyncio.create_task(self._startup_and_supervise())
        self._server.add_done_callback(self._log_task_exception)

        await self._startup_event.wait()
        # Explicit annotation: the supervisor coroutine (or its done-callback)
        # reassigns ``self._startup_error`` while we wait, which mypy cannot see.
        startup_error: Optional[BaseException] = self._startup_error
        if startup_error is not None:
            with suppress(BaseException):
                await self._server
            raise startup_error

        SERVICE_UP.labels(service=self.service_name).set(1)
        logger.info(f"Health server started on {self.host}:{self.port}")

    async def _startup_and_supervise(self) -> None:
        """Bind the port first, then supervise uvicorn's main loop.

        Mirrors uvicorn's ``Server._serve`` preamble (config load + lifespan
        creation) and then calls ``startup()`` directly, so a bind failure
        surfaces as an exception in this coroutine rather than as a silent
        ``SystemExit`` inside an unsupervised task.
        """
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=os.getenv("HEALTH_LOG_LEVEL", "warning"),
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._uvicorn_server = server

        try:
            if not config.loaded:
                config.load()
            server.lifespan = config.lifespan_class(config)
            try:
                # Binds the socket. On failure uvicorn logs the error, shuts
                # the lifespan down and raises SystemExit(STARTUP_FAILURE).
                await server.startup()
            except (SystemExit, OSError) as exc:
                raise RuntimeError(
                    f"Health server failed to start on {self.host}:{self.port} "
                    f"(port may already be in use): {exc!r}"
                ) from exc

            if server.should_exit or not server.started:
                with suppress(Exception):
                    await server.lifespan.shutdown()
                raise RuntimeError(
                    f"Health server failed to start on {self.host}:{self.port}: "
                    "uvicorn exited during startup"
                )
        except Exception as exc:
            self._startup_error = (
                exc
                if isinstance(exc, RuntimeError)
                else RuntimeError(f"Health server failed: {exc!r}")
            )
            self._startup_event.set()
            raise

        self._startup_event.set()

        try:
            await server.main_loop()
        finally:
            with suppress(Exception):
                await server.shutdown()

    async def stop(self) -> None:
        """Stop the health server."""
        if self._server:
            server = self._uvicorn_server
            if server is not None:
                # Graceful exit: let uvicorn close connections and the socket.
                server.should_exit = True
            try:
                await asyncio.wait_for(self._server, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._server.cancel()
                with suppress(asyncio.CancelledError):
                    await self._server
            except Exception:
                # Shutdown errors are already logged by the done-callback.
                pass
        self._uvicorn_server = None
        SERVICE_UP.labels(service=self.service_name).set(0)
        logger.info("Health server stopped")

    async def __aenter__(self) -> "HealthServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


def default_checks() -> None:
    """Register default health checks."""
    register_health_check("memory", _memory_healthy)
    register_health_check("redis", _redis_healthy)
    register_health_check("database", _database_healthy)
    register_readiness_check("redis", _redis_healthy)
    register_readiness_check("database", _database_healthy)
