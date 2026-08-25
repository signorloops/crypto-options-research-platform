"""
Tests for health server endpoints and check execution behavior.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import core.health_server as health_server


def setup_function():
    """Reset global check registries for test isolation."""
    health_server._health_checks.clear()
    health_server._readiness_checks.clear()
    for key in (
        "REDIS_URL",
        "REDIS_HOST",
        "REDIS_PORT",
        "DB_URL",
        "DB_HOST",
        "DB_PORT",
    ):
        os.environ.pop(key, None)


def test_health_endpoint_handles_async_and_failing_checks():
    """Health endpoint should isolate failing checks and still respond."""

    async def async_ok() -> bool:
        return True

    def broken_check() -> bool:
        raise RuntimeError("boom")

    health_server.register_health_check("sync_ok", lambda: True)
    health_server.register_health_check("async_ok", async_ok)
    health_server.register_health_check("broken", broken_check)

    app = health_server.create_health_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["sync_ok"] is True
    assert payload["checks"]["async_ok"] is True
    assert payload["checks"]["broken"] is False


def test_ready_endpoint_returns_503_when_checks_fail():
    """Readiness endpoint should fail closed when any readiness check fails."""
    health_server.register_readiness_check("ready_ok", lambda: True)
    health_server.register_readiness_check("ready_fail", lambda: False)

    app = health_server.create_health_app()
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"]["status"] == "not_ready"
    assert "ready_fail" in payload["detail"]["failed_checks"]


def test_ready_endpoint_supports_async_checks():
    """Readiness endpoint should support coroutine-based checks."""

    async def async_ready() -> bool:
        return True

    health_server.register_readiness_check("async_ready", async_ready)

    app = health_server.create_health_app()
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "async_ready" in payload["checks"]


def test_metrics_endpoint_returns_prometheus_text():
    """Metrics endpoint should expose Prometheus text format."""
    app = health_server.create_health_app(service_name="test-service")
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "corp_http_requests_total" in response.text


def test_default_checks_include_real_connectivity_checks(monkeypatch):
    """Default checks should register memory/redis/database probes."""

    async def _always_true(*args, **kwargs):
        return True

    monkeypatch.setattr(health_server, "_tcp_connectivity_check", _always_true)
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("DB_HOST", "localhost")

    health_server.default_checks()

    assert {"memory", "redis", "database"} <= set(health_server._health_checks.keys())
    assert {"redis", "database"} <= set(health_server._readiness_checks.keys())

    app = health_server.create_health_app()
    with TestClient(app) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json()["checks"]["redis"] is True
    assert health_response.json()["checks"]["database"] is True
    assert ready_response.status_code == 200


def test_connection_target_prefers_url_and_fallback(monkeypatch):
    """Connection target should parse URL first then fallback to host/port env."""
    monkeypatch.setenv("REDIS_URL", "redis://cache.local:6380/0")
    assert health_server._connection_target("REDIS_URL", "REDIS_HOST", "REDIS_PORT", 6379) == (
        "cache.local",
        6380,
    )

    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "invalid")
    assert health_server._connection_target("REDIS_URL", "REDIS_HOST", "REDIS_PORT", 6379) == (
        "redis.internal",
        6379,
    )


def test_memory_healthy_respects_threshold(monkeypatch):
    """Memory check should fail when usage exceeds configured threshold."""
    monkeypatch.setattr(health_server, "_memory_usage_mb", lambda: 960.0)
    monkeypatch.setenv("MEMORY_LIMIT_MB", "1000")
    monkeypatch.setenv("MEMORY_HEALTH_THRESHOLD", "0.95")
    assert health_server._memory_healthy() is False

    monkeypatch.setattr(health_server, "_memory_usage_mb", lambda: 930.0)
    assert health_server._memory_healthy() is True


def test_memory_usage_reads_proc_status_before_mach(monkeypatch):
    """Current RSS should come from /proc/self/status (VmRSS, kB) when available."""
    monkeypatch.setattr(health_server, "_read_proc_status_rss_kb", lambda: 2048.0, raising=True)
    monkeypatch.setattr(health_server, "_read_mach_task_rss_bytes", lambda: 999 * 1024 * 1024)
    assert health_server._memory_usage_mb() == 2.0


def test_memory_usage_reads_mach_task_info_when_proc_missing(monkeypatch):
    """Without /proc, macOS Mach task_info resident size (bytes) should be used."""
    monkeypatch.setattr(health_server, "_read_proc_status_rss_kb", lambda: None)
    monkeypatch.setattr(health_server, "_read_mach_task_rss_bytes", lambda: 5 * 1024 * 1024)
    assert health_server._memory_usage_mb() == 5.0


def test_memory_usage_reports_current_rss_not_historic_peak(monkeypatch):
    """A transient spike must not keep the check unhealthy after memory is freed.

    ru_maxrss is peak RSS on macOS/maximum on Linux; the health check must use
    current RSS instead so a one-time spike does not permanently degrade the
    service.
    """
    monkeypatch.setattr(health_server, "_read_proc_status_rss_kb", lambda: None)
    monkeypatch.setattr(health_server, "_read_mach_task_rss_bytes", lambda: 512 * 1024 * 1024)

    class _Rusage:
        ru_maxrss = 4 * 1024 * 1024 * 1024  # historic 4GiB peak, in bytes

    monkeypatch.setattr(health_server.resource, "getrusage", lambda *_args, **_kwargs: _Rusage())

    monkeypatch.setenv("MEMORY_LIMIT_MB", "1024")
    monkeypatch.delenv("MEMORY_HEALTH_THRESHOLD", raising=False)

    # Current RSS (512MB) is well under the limit even though the historic
    # peak (4GiB) would have exceeded it.
    assert health_server._memory_usage_mb() == 512.0
    assert health_server._memory_healthy() is True


@pytest.mark.asyncio
async def test_tcp_connectivity_check_success_and_failure():
    """TCP connectivity helper should return True on open port and False otherwise."""

    async def _handler(reader, writer):
        writer.close()

    server = await asyncio.start_server(_handler, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()

    try:
        assert await health_server._tcp_connectivity_check(host, port, timeout_seconds=0.5) is True
    finally:
        server.close()
        await server.wait_closed()

    assert (
        await health_server._tcp_connectivity_check("127.0.0.1", port, timeout_seconds=0.1) is False
    )


@pytest.mark.asyncio
async def test_health_server_start_and_stop_with_uvicorn_stub(monkeypatch):
    """HealthServer should start and stop cleanly with cancellable server task."""

    class _FakeConfig:
        def __init__(self, app, host, port, log_level, access_log):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level
            self.access_log = access_log
            self.loaded = True
            self.lifespan_class = lambda config: _FakeLifespan()

    class _FakeLifespan:
        async def startup(self):
            return None

        async def shutdown(self):
            return None

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False
            self.lifespan = _FakeLifespan()

        async def startup(self):
            self.started = True

        async def shutdown(self):
            return None

        async def main_loop(self):
            while not self.should_exit:
                await asyncio.sleep(0.01)

    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(Config=_FakeConfig, Server=_FakeServer)
    )

    server = health_server.HealthServer(host="127.0.0.1", port=9999, service_name="svc")
    await server.start()
    assert server._server is not None
    assert not server._server.done()

    await server.stop()
    assert server._server.done()


@pytest.mark.asyncio
async def test_health_server_start_raises_when_port_taken(monkeypatch):
    """start() must raise when uvicorn cannot bind instead of reporting success."""

    class _FakeConfig:
        def __init__(self, app, host, port, log_level, access_log):
            self.loaded = True
            self.lifespan_class = lambda config: _FakeLifespan()

    class _FakeLifespan:
        async def startup(self):
            return None

        async def shutdown(self):
            return None

    class _TakenPortServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False
            self.lifespan = _FakeLifespan()

        async def startup(self):
            # Mirror uvicorn's behaviour on bind failure.
            raise SystemExit(3)

        async def shutdown(self):
            return None

        async def main_loop(self):
            await asyncio.sleep(3600)

    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(Config=_FakeConfig, Server=_TakenPortServer)
    )

    server = health_server.HealthServer(host="127.0.0.1", port=9998, service_name="svc")
    with pytest.raises(RuntimeError, match="failed to start"):
        await server.start()

    # SERVICE_UP must not claim the service is up after a failed start.
    assert server._server.done()


@pytest.mark.asyncio
async def test_health_server_task_exceptions_are_logged(monkeypatch, caplog):
    """Crashes in the serve loop should be logged by the done-callback."""

    class _CrashingServer:
        def __init__(self, config):
            self.started = False
            self.should_exit = False
            self.lifespan = _FakeLifespanStub()

        async def startup(self):
            self.started = True

        async def shutdown(self):
            return None

        async def main_loop(self):
            await asyncio.sleep(0)
            raise RuntimeError("boom in serve loop")

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(
            Config=lambda *a, **k: SimpleNamespace(
                loaded=True, lifespan_class=lambda cfg: _FakeLifespanStub()
            ),
            Server=_CrashingServer,
        ),
    )

    server = health_server.HealthServer(host="127.0.0.1", port=9997, service_name="svc")
    with caplog.at_level("ERROR", logger="core.health_server"):
        await server.start()
        with pytest.raises(RuntimeError, match="boom in serve loop"):
            await server._server

    assert any("crashed" in rec.message for rec in caplog.records)


class _FakeLifespanStub:
    async def startup(self):
        return None

    async def shutdown(self):
        return None


@pytest.mark.asyncio
async def test_connection_targets_and_readiness_defaults(monkeypatch):
    """Connectivity checks should be no-op healthy when endpoints are not configured."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)

    assert health_server._connection_target("REDIS_URL", "REDIS_HOST", "REDIS_PORT", 6379) is None
    assert await health_server._redis_healthy() is True
    assert await health_server._database_healthy() is True
