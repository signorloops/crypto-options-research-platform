"""
Tests for logging configuration.
"""
import json
import logging
import os
import tempfile

import pytest

from utils.logging_config import (
    JSONFormatter,
    StandardFormatter,
    debug,
    error,
    exception,
    get_logger,
    info,
    log_extra,
    setup_logging,
    warning,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging handlers after each test to ensure isolation."""
    yield
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.WARNING)


class TestJSONFormatter:
    """Test JSON formatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_sanitize_sensitive_data(self):
        """Test sensitive data is redacted."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.extra_data = {"api_key": "secret123", "password": "mypass"}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["extra"]["api_key"] == "***REDACTED***"
        assert data["extra"]["password"] == "***REDACTED***"

    def test_sanitize_nested_dict(self):
        """Test sanitization in nested dictionaries."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.extra_data = {
            "user": {"api_secret": "secret", "name": "test"},
            "token": "abc123"
        }

        output = formatter.format(record)
        data = json.loads(output)

        assert data["extra"]["user"]["api_secret"] == "***REDACTED***"
        assert data["extra"]["user"]["name"] == "test"
        assert data["extra"]["token"] == "***REDACTED***"


class TestStandardFormatter:
    """Test standard formatter."""

    def test_format(self):
        """Test standard format."""
        formatter = StandardFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        assert "INFO" in output
        assert "test" in output
        assert "Test message" in output


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_console_only(self, caplog):
        """Test setup with console handler only."""
        with caplog.at_level(logging.DEBUG):
            setup_logging(level="DEBUG", log_file=None, force=False)
            logger = logging.getLogger("test_console")
            logger.debug("Debug message")

        assert "Debug message" in caplog.text

    def test_setup_with_file(self):
        """Test setup with file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            setup_logging(level="INFO", log_file=log_file)

            logger = logging.getLogger("test_file")
            logger.info("File test message")

            # Check file exists and has content
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "File test message" in content

    def test_json_format(self):
        """Test JSON format logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.json.log")
            setup_logging(level="INFO", log_file=log_file, log_format="json")

            logger = logging.getLogger("test_json")
            logger.info("JSON test")

            with open(log_file) as f:
                lines = f.readlines()
                # Get the last JSON line (the test log message)
                data = json.loads(lines[-1].strip())
                assert data["message"] == "JSON test"

    def test_third_party_loggers(self):
        """Test third-party logger levels are set."""
        setup_logging()

        assert logging.getLogger("websockets").level == logging.WARNING
        assert logging.getLogger("aiohttp").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING


class TestLogExtra:
    """Test log_extra function."""

    def test_log_extra(self):
        """Test log_extra creates correct dict."""
        extra = log_extra(instrument="BTC", price=50000)
        assert extra == {"extra_data": {"instrument": "BTC", "price": 50000}}


class TestConvenienceFunctions:
    """Test convenience logging functions."""

    def test_debug(self, caplog):
        """Test debug function."""
        with caplog.at_level(logging.DEBUG):
            setup_logging(level="DEBUG", force=False)
            debug("Debug test", instrument="BTC")
        assert "Debug test" in caplog.text

    def test_info(self, caplog):
        """Test info function."""
        with caplog.at_level(logging.INFO):
            setup_logging(level="INFO", force=False)
            info("Info test", price=50000)
        assert "Info test" in caplog.text

    def test_warning(self, caplog):
        """Test warning function."""
        with caplog.at_level(logging.WARNING):
            setup_logging(level="WARNING", force=False)
            warning("Warning test")
        assert "Warning test" in caplog.text

    def test_error(self, caplog):
        """Test error function."""
        with caplog.at_level(logging.ERROR):
            setup_logging(level="ERROR", force=False)
            error("Error test", error_code=500)
        assert "Error test" in caplog.text

    def test_exception(self, caplog):
        """Test exception function."""
        with caplog.at_level(logging.ERROR):
            setup_logging(level="ERROR", force=False)
            try:
                raise ValueError("Test exception")
            except ValueError:
                exception("Exception occurred")

        assert "Exception occurred" in caplog.text
        assert "Test exception" in caplog.text


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger(self):
        """Test get_logger returns logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
