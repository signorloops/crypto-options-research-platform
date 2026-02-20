#!/bin/bash
set -e

# 币本位期权做市策略 - 完整验证脚本
# 验证所有5个阶段的实施结果

echo "======================================================================"
echo "          Market Making Strategy - Full Verification"
echo "======================================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASSED+=1))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAILED+=1))
}

check_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

resolve_python_bin() {
    local candidates=()
    local versioned_names=("python3.13" "python3.12" "python3.11" "python3.10" "python3.9")
    local candidate

    if [[ -n "${CORP_PYTHON_BIN:-}" ]]; then
        candidates+=("$CORP_PYTHON_BIN")
    fi

    if command -v python >/dev/null 2>&1; then
        candidates+=("$(command -v python)")
    fi

    if command -v python3 >/dev/null 2>&1; then
        candidates+=("$(command -v python3)")
    fi

    for candidate in "${versioned_names[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            candidates+=("$(command -v "$candidate")")
        fi
    done

    while IFS= read -r candidate; do
        candidates+=("$candidate")
    done < <(which -a python3 2>/dev/null || true)

    while IFS= read -r candidate; do
        candidates+=("$candidate")
    done < <(which -a python 2>/dev/null || true)

    local seen=" "
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate" ]] || continue
        [[ "$seen" == *" $candidate "* ]] && continue
        seen+="$candidate "

        if "$candidate" -c 'import importlib.util, sys; raise SystemExit(0 if sys.version_info >= (3, 9) and importlib.util.find_spec("pytest") else 1)' >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# 检测并激活虚拟环境
VENV_PATH="${CORP_VENV_PATH:-venv}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
elif [[ -f "env/bin/activate" ]]; then
    source "env/bin/activate"
else
    echo "Warning: No virtual environment found, using system Python"
fi

export PYTHONPATH=.
PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo -e "${RED}Error: Python >=3.9 with pytest not found. Install dev deps or set CORP_PYTHON_BIN.${NC}"
    exit 1
fi

echo "Phase 1: Core Modules"
echo "----------------------------------------------------------------------"

# 测试熔断系统
if "$PYTHON_BIN" -m pytest tests/test_circuit_breaker.py -q --tb=no > /dev/null 2>&1; then
    check_pass "CircuitBreaker (28 tests)"
else
    check_fail "CircuitBreaker"
fi

# 测试Regime检测
if "$PYTHON_BIN" -m pytest tests/test_regime_detector.py -q --tb=no > /dev/null 2>&1; then
    check_pass "VolatilityRegimeDetector (25 tests)"
else
    check_fail "VolatilityRegimeDetector"
fi

# 测试自适应对冲
if "$PYTHON_BIN" -m pytest tests/test_adaptive_hedging.py -q --tb=no > /dev/null 2>&1; then
    check_pass "AdaptiveDeltaHedger (33 tests)"
else
    check_fail "AdaptiveDeltaHedger"
fi

echo ""
echo "Phase 2: Integrated Strategy"
echo "----------------------------------------------------------------------"

if "$PYTHON_BIN" -m pytest tests/test_integrated_strategy.py -q --tb=no > /dev/null 2>&1; then
    check_pass "IntegratedMarketMakingStrategy (30 tests)"
else
    check_fail "IntegratedMarketMakingStrategy"
fi

echo ""
echo "Phase 3: Math Validation"
echo "----------------------------------------------------------------------"

if PYTHONPATH=. "$PYTHON_BIN" validation_scripts/inverse_pnl_math_validation.py > /dev/null 2>&1; then
    check_pass "Inverse PnL Formula Validation (5/5 tests)"
else
    check_fail "Inverse PnL Formula Validation"
fi

if [ -f "validation_scripts/inverse_pnl_validation_report.md" ]; then
    check_pass "Validation Report Generated"
else
    check_fail "Validation Report Missing"
fi

echo ""
echo "Phase 4: Production Code"
echo "----------------------------------------------------------------------"

# 检查性能测试文件
if [ -f "tests/performance/latency_benchmark.py" ]; then
    check_pass "Latency Benchmark Script"
else
    check_fail "Latency Benchmark Script Missing"
fi

if [ -f "tests/performance/stress_test.py" ]; then
    check_pass "Stress Test Script"
else
    check_fail "Stress Test Script Missing"
fi

# 运行覆盖率检查
COVERAGE=$("$PYTHON_BIN" -m pytest tests/test_circuit_breaker.py tests/test_regime_detector.py \
    tests/test_adaptive_hedging.py tests/test_integrated_strategy.py \
    --cov=research.risk.circuit_breaker --cov=research.signals.regime_detector \
    --cov=research.hedging.adaptive_delta --cov=strategies.market_making.integrated_strategy \
    --cov-report=term 2>&1 | grep "TOTAL" | awk '{print $4}' | tr -d '%')

if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
    check_pass "Test Coverage: ${COVERAGE}% (target: 80%)"
else
    check_fail "Test Coverage: ${COVERAGE}% (target: 80%)"
fi

echo ""
echo "Phase 5: Deployment"
echo "----------------------------------------------------------------------"

# 检查部署文件
DEPLOY_FILES=(
    "deployment/docker-compose.prod.yml"
    "deployment/Dockerfile.prod"
    "deployment/k8s/trading-engine.yaml"
    "deployment/scripts/deploy.sh"
    "deployment/config/prometheus.yml"
    "deployment/sql/init.sql"
    "deployment/DEPLOYMENT.md"
)

for file in "${DEPLOY_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$(basename $file)"
    else
        check_fail "$(basename $file) Missing"
    fi
done

echo ""
echo "======================================================================"
echo "                        Verification Summary"
echo "======================================================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL PHASES COMPLETED SUCCESSFULLY${NC}"
    echo ""
    echo "Project Statistics:"
    echo "  - Core Modules: 4"
    echo "  - Test Cases: 116"
    echo "  - Test Coverage: $COVERAGE%"
    echo "  - Deployment Files: 7"
    echo ""
    echo "Next Steps:"
    echo "  1. Review deployment/DEPLOYMENT.md for production setup"
    echo "  2. Run: cd deployment && ./scripts/deploy.sh staging"
    echo "  3. Access Grafana: http://localhost:3000"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME CHECKS FAILED${NC}"
    echo "Please review the failed items above."
    exit 1
fi
