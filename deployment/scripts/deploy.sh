#!/bin/bash
set -euo pipefail

# Market Making Strategy - Production Deployment Script
# Usage: ./deploy.sh [environment] [version]
# Example: ./deploy.sh production v1.2.3

ENV=${1:-production}
VERSION=${2:-latest}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CORP_PROJECT_ROOT:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# Verify project root exists
if [[ ! -d "$PROJECT_ROOT" ]]; then
    log_error "Project root not found: $PROJECT_ROOT"
    log_error "Set CORP_PROJECT_ROOT environment variable to override"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
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

resolve_compose_file() {
    case "$ENV" in
        production) echo "$PROJECT_ROOT/deployment/docker-compose.prod.yml" ;;
        staging) echo "$PROJECT_ROOT/deployment/docker-compose.staging.yml" ;;
        *)
            log_error "Unknown environment: $ENV"
            exit 1
            ;;
    esac
}

resolve_env_file() {
    case "$ENV" in
        production)
            if [[ -f "$PROJECT_ROOT/deployment/.env.prod" ]]; then
                echo "$PROJECT_ROOT/deployment/.env.prod"
            else
                echo "$PROJECT_ROOT/.env.prod"
            fi
            ;;
        staging)
            if [[ -f "$PROJECT_ROOT/deployment/.env.staging" ]]; then
                echo "$PROJECT_ROOT/deployment/.env.staging"
            else
                echo "$PROJECT_ROOT/.env.staging"
            fi
            ;;
        *)
            log_error "Unknown environment: $ENV"
            exit 1
            ;;
    esac
}

COMPOSE_FILE="$(resolve_compose_file)"
ENV_FILE="$(resolve_env_file)"

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi

    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a

    required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "EXCHANGE_API_KEY" "EXCHANGE_API_SECRET")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required variable $var is missing in $ENV_FILE"
            exit 1
        fi
    done

    log_info "Running strategy regression tests..."
    cd "$PROJECT_ROOT"

    # 检测并激活虚拟环境
    VENV_PATH="${CORP_VENV_PATH:-venv}"
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        source "$VENV_PATH/bin/activate"
    elif [[ -f ".venv/bin/activate" ]]; then
        source ".venv/bin/activate"
    elif [[ -f "env/bin/activate" ]]; then
        source "env/bin/activate"
    else
        log_warn "No virtual environment found, using system Python"
    fi

    PYTHON_BIN="$(resolve_python_bin || true)"
    if [[ -z "$PYTHON_BIN" ]]; then
        log_error "Python >=3.9 with pytest not found. Install dev deps or set CORP_PYTHON_BIN."
        exit 1
    fi

    PYTHONPATH=. "$PYTHON_BIN" -m pytest \
        tests/test_circuit_breaker.py \
        tests/test_regime_detector.py \
        tests/test_adaptive_hedging.py \
        tests/test_integrated_strategy.py \
        -q --tb=short

    log_info "Pre-deployment checks passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"

    docker build -f deployment/Dockerfile.prod \
        -t "mm-trading-engine:$VERSION" \
        -t "mm-trading-engine:latest" \
        .

    log_info "Docker images built successfully"
}

# Deploy to environment
deploy() {
    log_info "Deploying to $ENV environment..."
    cd "$PROJECT_ROOT/deployment"

    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a

    docker-compose -f "$COMPOSE_FILE" up -d --build

    # Wait for health endpoint
    sleep 10
    HEALTH_CHECK_URL="${CORP_HEALTH_CHECK_URL:-http://localhost:8080/health}"
    if ! curl -fsS "$HEALTH_CHECK_URL" >/dev/null; then
        log_error "Health check failed after deployment: $HEALTH_CHECK_URL"
        return 1
    fi

    log_info "Deployment to $ENV completed"
}

# Post-deployment verification
post_deployment_verify() {
    log_info "Running post-deployment verification..."

    services=("trading-engine" "risk-monitor" "market-data" "redis" "postgres")
    for service in "${services[@]}"; do
        if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "$service.*Up"; then
            log_error "Service $service is not running"
            exit 1
        fi
    done

    HEALTH_CHECK_URL="${CORP_HEALTH_CHECK_URL:-http://localhost:8080/health}"
    if ! curl -fsS "$HEALTH_CHECK_URL" >/dev/null; then
        log_error "Trading engine health check failed: $HEALTH_CHECK_URL"
        exit 1
    fi

    PROMETHEUS_URL="${CORP_PROMETHEUS_URL:-http://localhost:9090/-/healthy}"
    if ! curl -fsS "$PROMETHEUS_URL" >/dev/null; then
        log_warn "Prometheus health check failed: $PROMETHEUS_URL"
    fi

    log_info "Post-deployment verification passed"
}

# Rollback on failure
rollback() {
    log_warn "Rolling back deployment..."

    if [[ -f "$COMPOSE_FILE" ]]; then
        docker-compose -f "$COMPOSE_FILE" down || true
    fi

    log_info "Rollback completed"
}

# Main deployment flow
main() {
    log_info "Starting deployment to $ENV environment (version: $VERSION)"

    trap rollback ERR

    pre_deployment_checks
    build_images
    deploy
    post_deployment_verify

    trap - ERR
    log_info "Deployment completed successfully"
}

main
