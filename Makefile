.PHONY: help install install-dev test test-unit test-integration test-cov lint lint-fix format format-check type-check quality clean docs research-audit research-audit-compare research-audit-refresh-baseline

# Detect virtual environment or use system Python
VENV_PYTHON := $(wildcard ./venv/bin/python) $(wildcard ./.venv/bin/python) $(wildcard ./.venv311/bin/python) $(wildcard ./env/bin/python)
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
	@echo "  research-audit   Generate IV stability/model-zoo/rough-jump research reports"
	@echo "  research-audit-compare Compare current audit snapshot against tracked baseline"
	@echo "  research-audit-refresh-baseline Refresh tracked baseline with current snapshot"
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

research-audit:
	mkdir -p artifacts
	$(PYTHON) validation_scripts/iv_surface_stability_report.py \
		--seed 42 \
		--fail-on-arbitrage \
		--min-short-max-jump-reduction 0.005 \
		--output-md artifacts/iv-surface-stability-report.md \
		--output-json artifacts/iv-surface-stability-report.json
	$(PYTHON) validation_scripts/rough_jump_experiment.py --seed 42 > artifacts/rough-jump-experiment.txt
	$(PYTHON) validation_scripts/jump_premia_stability_report.py \
		--seed 42 \
		--output-md artifacts/jump-premia-stability-report.md \
		--output-json artifacts/jump-premia-stability-report.json
	$(PYTHON) validation_scripts/pricing_model_zoo_benchmark.py \
		--quotes-json validation_scripts/fixtures/model_zoo_quotes_seed42.json \
		--expected-best-model bates \
		--max-best-rmse 120.0 \
		--output-json artifacts/pricing-model-zoo-benchmark.json \
		--output-md artifacts/pricing-model-zoo-benchmark.md \
		> artifacts/pricing-model-zoo-benchmark.txt
	$(PYTHON) validation_scripts/research_audit_snapshot.py \
		--iv-report-json artifacts/iv-surface-stability-report.json \
		--model-zoo-json artifacts/pricing-model-zoo-benchmark.json \
		--rough-jump-txt artifacts/rough-jump-experiment.txt \
		--output-json artifacts/research-audit-snapshot.json
	$(PYTHON) validation_scripts/research_audit_compare.py \
		--baseline-json validation_scripts/fixtures/research_audit_snapshot_baseline.json \
		--current-json artifacts/research-audit-snapshot.json \
		--max-best-rmse-increase-pct 25.0 \
		--max-iv-reduction-drop-pct 30.0 \
		--output-json artifacts/research-audit-drift-report.json \
		--output-md artifacts/research-audit-drift-report.md
	$(PYTHON) validation_scripts/research_audit_weekly_summary.py \
		--iv-report-json artifacts/iv-surface-stability-report.json \
		--model-zoo-json artifacts/pricing-model-zoo-benchmark.json \
		--drift-report-json artifacts/research-audit-drift-report.json \
		--output-md artifacts/research-audit-weekly-summary.md
	@echo "Research audit artifacts generated under artifacts/"

research-audit-compare:
	$(PYTHON) validation_scripts/research_audit_compare.py \
		--baseline-json validation_scripts/fixtures/research_audit_snapshot_baseline.json \
		--current-json artifacts/research-audit-snapshot.json \
		--max-best-rmse-increase-pct 25.0 \
		--max-iv-reduction-drop-pct 30.0 \
		--output-json artifacts/research-audit-drift-report.json \
		--output-md artifacts/research-audit-drift-report.md

research-audit-refresh-baseline:
	$(MAKE) research-audit
	cp artifacts/research-audit-snapshot.json validation_scripts/fixtures/research_audit_snapshot_baseline.json
	@echo "Baseline refreshed: validation_scripts/fixtures/research_audit_snapshot_baseline.json"

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
