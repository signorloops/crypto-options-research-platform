.PHONY: help install install-dev test test-unit test-integration test-cov lint lint-fix format format-check type-check quality clean docs

# Detect virtual environment or use system Python
VENV_PYTHON := $(wildcard ./venv/bin/python) $(wildcard ./.venv/bin/python) $(wildcard ./env/bin/python)
ifeq ($(VENV_PYTHON),)
    PYTHON ?= $(shell which python3)
else
    PYTHON ?= $(firstword $(VENV_PYTHON))
endif
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BLACK := $(PYTHON) -m black
MYPY := $(PYTHON) -m mypy

SRC_DIRS := core data research strategies utils config execution tests

help:
	@echo "Available commands:"
	@echo "  install          Install package dependencies"
	@echo "  install-dev      Install package with development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  lint             Run linting checks (ruff)"
	@echo "  lint-fix         Run linting and auto-fix where possible"
	@echo "  format           Format code (black)"
	@echo "  format-check     Check code formatting"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  quality          Run format-check + lint + type-check"
	@echo "  clean            Clean build artifacts"
	@echo "  docs             Build documentation"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

test:
	$(PYTEST) -v -m "not integration"

test-unit:
	$(PYTEST) -v -m "not integration"

test-integration:
	RUN_INTEGRATION_TESTS=1 $(PYTEST) -v -m "integration"

test-cov:
	$(PYTEST) --cov=core --cov=data --cov=research --cov=strategies --cov=utils --cov=config --cov=execution --cov-report=term-missing --cov-report=html

lint:
	$(RUFF) check $(SRC_DIRS) verify_implementation.py

lint-fix:
	$(RUFF) check --fix $(SRC_DIRS) verify_implementation.py

format:
	$(BLACK) core data research strategies utils config tests execution verify_implementation.py

format-check:
	$(BLACK) --check core data research strategies utils config tests execution verify_implementation.py

type-check:
	$(MYPY) core data research strategies utils config execution tests

quality: format-check lint type-check
	@echo "All quality checks passed!"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Building documentation..."
	@echo "Docs would be built here"
