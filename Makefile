.PHONY: help install install-dev install-dev-full workspace-slim-report workspace-slim-clean workspace-slim-clean-venv test test-unit test-integration test-cov lint lint-fix format format-check type-check quality branch-name-guard check-service-entrypoint docs-link-check complexity-audit daily-regression weekly-operating-audit weekly-close-gate weekly-pnl-attribution weekly-canary-checklist weekly-decision-log weekly-manual-prefill weekly-signoff-pack weekly-consistency-replay weekly-adr-draft clean docs

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
	@echo "  install-dev-full Install dev + heavy optional stacks (ml/notebook/accelerated)"
	@echo "  workspace-slim-report Dry-run workspace bloat cleanup plan"
	@echo "  workspace-slim-clean Clean safe generated/cache files"
	@echo "  workspace-slim-clean-venv Clean safe files + local virtual env dirs"
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
	@echo "  branch-name-guard Fail if local/remote branch names include forbidden keywords"
	@echo "  check-service-entrypoint Fail if deployment docs/scripts use legacy execution modules"
	@echo "  docs-link-check  Validate local markdown links"
	@echo "  complexity-audit Run strict complexity governance checks"
	@echo "  daily-regression Run daily regression gate report"
	@echo "  weekly-operating-audit Generate weekly KPI and risk exception report"
	@echo "  weekly-close-gate Run weekly governance chain and enforce READY_FOR_CLOSE gate"
	@echo "  weekly-pnl-attribution Generate weekly PnL attribution report"
	@echo "  weekly-canary-checklist Generate weekly canary rollout checklist"
	@echo "  weekly-decision-log Generate weekly decision and rollback log"
	@echo "  weekly-manual-prefill Auto-prefill objective fields in weekly manual status"
	@echo "  weekly-signoff-pack Generate weekly manual sign-off package"
	@echo "  weekly-consistency-replay Generate online/offline consistency replay report"
	@echo "  weekly-adr-draft Generate ADR draft from weekly audit JSON"
	@echo "  clean            Clean build artifacts"
	@echo "  docs             Build documentation"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

install-dev-full:
	$(PIP) install -e ".[dev,full]"
	$(PYTHON) -m pre_commit install

workspace-slim-report:
	$(PYTHON) scripts/maintenance/workspace_slimmer.py

workspace-slim-clean:
	$(PYTHON) scripts/maintenance/workspace_slimmer.py --include-results --apply

workspace-slim-clean-venv:
	$(PYTHON) scripts/maintenance/workspace_slimmer.py --include-results --include-venv --apply

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

branch-name-guard:
	$(PYTHON) scripts/governance/branch_name_guard.py --forbidden codex

check-service-entrypoint:
	@if rg -n "python\\s+-m\\s+execution\\.(trading_engine|risk_monitor|market_data_collector)|\"execution\\.(trading_engine|risk_monitor|market_data_collector)\"" deployment docs/deployment.md docs/project-map-mermaid.md -S; then \
		echo "Legacy entrypoint detected. Use python -m execution.service_runner with SERVICE_NAME."; \
		exit 1; \
	else \
		echo "Service entrypoint check passed."; \
	fi

docs-link-check:
	$(PYTHON) scripts/docs/check_markdown_links.py \
		--paths README.md docs/GUIDE.md docs/plans docs/archive/README.md docs/archive/plans/README.md

complexity-audit:
	$(PYTHON) scripts/governance/complexity_guard.py \
		--config config/complexity_budget.json \
		--report-md artifacts/complexity-governance-report.md \
		--report-json artifacts/complexity-governance-report.json \
		--strict

daily-regression:
	$(PYTHON) scripts/governance/daily_regression_gate.py \
		--cmd "$(PYTHON) -m pytest -q --noconftest tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py" \
		--output-md artifacts/daily-regression-gate.md \
		--output-json artifacts/daily-regression-gate.json \
		--strict

weekly-operating-audit:
	$(PYTHON) scripts/governance/weekly_operating_audit.py \
		--inputs \
			tests/fixtures/weekly_operating/backtest_results_20260209_174752.json \
			tests/fixtures/weekly_operating/backtest_results_20260209_174850.json \
			tests/fixtures/weekly_operating/backtest_full_20260209_175236.json \
		--thresholds config/weekly_operating_thresholds.json \
		--consistency-thresholds config/consistency_thresholds.json \
		--output-md artifacts/weekly-operating-audit.md \
		--output-json artifacts/weekly-operating-audit.json \
		--regression-cmd "$(PYTHON) -m pytest -q --noconftest tests/test_pricing_inverse.py tests/test_volatility.py tests/test_hawkes_comparison.py tests/test_research_dashboard.py" \
		--strict
	$(MAKE) weekly-pnl-attribution
	$(MAKE) weekly-canary-checklist
	$(MAKE) weekly-adr-draft ADR_OWNER="$(ADR_OWNER)"
	$(MAKE) weekly-decision-log
	$(MAKE) weekly-manual-prefill
	$(MAKE) weekly-consistency-replay
	$(MAKE) weekly-signoff-pack

weekly-close-gate:
	$(MAKE) weekly-operating-audit
	$(PYTHON) scripts/governance/weekly_operating_audit.py \
		--close-gate-only \
		--strict-close \
		--signoff-json artifacts/weekly-signoff-pack.json

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

weekly-decision-log:
	$(PYTHON) scripts/governance/weekly_decision_log.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--canary-json artifacts/weekly-canary-checklist.json \
		--output-md artifacts/weekly-decision-log.md \
		--output-json artifacts/weekly-decision-log.json

weekly-manual-prefill:
	$(PYTHON) scripts/governance/weekly_manual_status_prefill.py \
		--decision-json artifacts/weekly-decision-log.json \
		--attribution-json artifacts/weekly-pnl-attribution.json \
		--manual-status-json artifacts/weekly-manual-status.json

weekly-signoff-pack:
	$(PYTHON) scripts/governance/weekly_signoff_pack.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--canary-json artifacts/weekly-canary-checklist.json \
		--decision-json artifacts/weekly-decision-log.json \
		--attribution-json artifacts/weekly-pnl-attribution.json \
		--manual-status-json artifacts/weekly-manual-status.json \
		--consistency-replay-json artifacts/online-offline-consistency-replay.json \
		--output-md artifacts/weekly-signoff-pack.md \
		--output-json artifacts/weekly-signoff-pack.json

weekly-consistency-replay:
	$(PYTHON) scripts/governance/online_offline_consistency_replay.py \
		--audit-json artifacts/weekly-operating-audit.json \
		--live-json artifacts/live-deviation-snapshot.json \
		--thresholds config/online_offline_consistency_thresholds.json \
		--output-md artifacts/online-offline-consistency-replay.md \
		--output-json artifacts/online-offline-consistency-replay.json

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
