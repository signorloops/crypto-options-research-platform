.PHONY: help install install-dev test test-unit test-integration test-cov lint lint-fix format format-check type-check quality complexity-audit daily-regression weekly-operating-audit weekly-pnl-attribution weekly-canary-checklist weekly-adr-draft clean docs

# Detect Python interpreter with project minimum version (3.9+).
PYTHON_CANDIDATES := ./venv/bin/python ./.venv/bin/python ./env/bin/python python3.13 python3.12 python3.11 python3.10 python3.9 python3 python
PYTHON ?= $(shell \
	for candidate in $(PYTHON_CANDIDATES); do \
		resolved="$$candidate"; \
		if [ ! -x "$$resolved" ]; then \
			resolved=$$(command -v "$$candidate" 2>/dev/null || true); \
		fi; \
		[ -n "$$resolved" ] || continue; \
		[ -x "$$resolved" ] || continue; \
		"$$resolved" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1 || continue; \
		echo "$$resolved"; \
		break; \
	done \
)
ifeq ($(strip $(PYTHON)),)
$(error Python >=3.9 not found. Create a 3.11+ environment and run `make install-dev`)
endif
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BLACK := $(PYTHON) -m black
MYPY := $(PYTHON) -m mypy
ADR_OWNER ?= TBD

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
	@echo "  complexity-audit Run strict complexity governance checks"
	@echo "  daily-regression Run daily regression gate report"
	@echo "  weekly-operating-audit Generate weekly KPI and risk exception report"
	@echo "  weekly-pnl-attribution Generate weekly PnL attribution report"
	@echo "  weekly-canary-checklist Generate weekly canary rollout checklist"
	@echo "  weekly-adr-draft Generate ADR draft from weekly audit JSON"
	@echo "  clean            Clean build artifacts"
	@echo "  docs             Build documentation"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

test:
	$(PYTEST) -v

test-unit:
	$(PYTEST) -v -m "not integration"

test-integration:
	$(PYTEST) -v -m "integration"

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

complexity-audit:
	$(PYTHON) scripts/governance/complexity_guard.py \
		--config config/complexity_budget.json \
		--report-md artifacts/complexity-governance-report.md \
		--report-json artifacts/complexity-governance-report.json \
		--strict

daily-regression:
	$(PYTHON) scripts/governance/daily_regression_gate.py \
		--cmd "$(PYTHON) -m pytest -q tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py" \
		--output-md artifacts/daily-regression-gate.md \
		--output-json artifacts/daily-regression-gate.json \
		--strict

weekly-operating-audit:
	$(PYTHON) scripts/governance/weekly_operating_audit.py \
		--thresholds config/weekly_operating_thresholds.json \
		--consistency-thresholds config/consistency_thresholds.json \
		--output-md artifacts/weekly-operating-audit.md \
		--output-json artifacts/weekly-operating-audit.json \
		--regression-cmd "$(PYTHON) -m pytest -q tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py" \
		--strict
	$(PYTHON) scripts/governance/weekly_pnl_attribution.py \
		--output-md artifacts/weekly-pnl-attribution.md \
		--output-json artifacts/weekly-pnl-attribution.json
	$(PYTHON) scripts/governance/weekly_canary_checklist.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--attribution-json artifacts/weekly-pnl-attribution.json \
		--output-md artifacts/weekly-canary-checklist.md \
		--output-json artifacts/weekly-canary-checklist.json
	$(PYTHON) scripts/governance/weekly_adr_draft.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--output-md artifacts/weekly-adr-draft.md \
		--owner "$(ADR_OWNER)"

weekly-pnl-attribution:
	$(PYTHON) scripts/governance/weekly_pnl_attribution.py \
		--output-md artifacts/weekly-pnl-attribution.md \
		--output-json artifacts/weekly-pnl-attribution.json

weekly-canary-checklist:
	$(PYTHON) scripts/governance/weekly_canary_checklist.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--attribution-json artifacts/weekly-pnl-attribution.json \
		--output-md artifacts/weekly-canary-checklist.md \
		--output-json artifacts/weekly-canary-checklist.json

weekly-adr-draft:
	$(PYTHON) scripts/governance/weekly_adr_draft.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--output-md artifacts/weekly-adr-draft.md \
		--owner "$(ADR_OWNER)"

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
