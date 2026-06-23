#!/usr/bin/env bash
# =============================================================================
# WarehouseGPT — Full Stack Launcher (local development)
# =============================================================================
#
# Usage:
#   chmod +x scripts/run_full_stack.sh
#   ./scripts/run_full_stack.sh [--no-docker] [--no-twin] [--no-agent]
#
# Options:
#   --no-docker   Skip docker-compose services (assume already running)
#   --no-twin     Skip Digital Twin API
#   --no-agent    Skip WarehouseGPT Agent API
#   --help        Show this help
#
# Prerequisites:
#   - Docker + Docker Compose plugin
#   - Python 3.10+ with warehousegpt installed (pip install -e .[dev])
#   - .env file in the project root (copy from .env.example and fill secrets)
#
# The script writes PID files to /tmp/warehousegpt/ for clean shutdown.
# Send SIGINT (Ctrl-C) or SIGTERM to this process to stop everything.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${GREEN}=== $* ===${NC}\n"; }

# ---------------------------------------------------------------------------
# Script location — find project root relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
info "Project root: ${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
START_DOCKER=true
START_TWIN=true
START_AGENT=true

for arg in "$@"; do
    case "$arg" in
        --no-docker) START_DOCKER=false ;;
        --no-twin)   START_TWIN=false ;;
        --no-agent)  START_AGENT=false ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,\}//' | head -30
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [[ ! -f ".env" ]]; then
    warn ".env file not found — copying from .env.example"
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        warn "Please edit .env and fill in required secrets before production use."
    else
        error ".env.example not found. Cannot continue."
        exit 1
    fi
fi

# shellcheck disable=SC1091
set -o allexport
source .env
set +o allexport

# ---------------------------------------------------------------------------
# PID / log directory
# ---------------------------------------------------------------------------
RUNDIR="/tmp/warehousegpt"
mkdir -p "${RUNDIR}"

TWIN_PID_FILE="${RUNDIR}/digital_twin.pid"
AGENT_PID_FILE="${RUNDIR}/agent_api.pid"
TWIN_LOG="${RUNDIR}/digital_twin.log"
AGENT_LOG="${RUNDIR}/agent_api.log"

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
cleanup() {
    header "Shutting down WarehouseGPT stack"

    if [[ -f "${AGENT_PID_FILE}" ]]; then
        AGENT_PID=$(cat "${AGENT_PID_FILE}")
        if kill -0 "${AGENT_PID}" 2>/dev/null; then
            info "Stopping Agent API (PID ${AGENT_PID})..."
            kill "${AGENT_PID}" 2>/dev/null || true
        fi
        rm -f "${AGENT_PID_FILE}"
    fi

    if [[ -f "${TWIN_PID_FILE}" ]]; then
        TWIN_PID=$(cat "${TWIN_PID_FILE}")
        if kill -0 "${TWIN_PID}" 2>/dev/null; then
            info "Stopping Digital Twin API (PID ${TWIN_PID})..."
            kill "${TWIN_PID}" 2>/dev/null || true
        fi
        rm -f "${TWIN_PID_FILE}"
    fi

    if [[ "${START_DOCKER}" == "true" ]]; then
        info "Stopping docker-compose services..."
        docker compose stop 2>/dev/null || true
    fi

    success "All services stopped."
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Helper: wait for HTTP endpoint to become healthy
# ---------------------------------------------------------------------------
wait_for_http() {
    local url="$1"
    local label="$2"
    local max_wait="${3:-60}"
    local elapsed=0

    info "Waiting for ${label} at ${url} ..."
    while ! curl -sf "${url}" > /dev/null 2>&1; do
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            error "${label} did not become healthy within ${max_wait}s"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""
    success "${label} is ready (${elapsed}s)"
}

# ---------------------------------------------------------------------------
# Step 1: Docker Compose services
# ---------------------------------------------------------------------------
if [[ "${START_DOCKER}" == "true" ]]; then
    header "Starting Docker Compose infrastructure services"

    # Only start the infrastructure services — not the app containers
    INFRA_SERVICES="postgres redis neo4j chromadb"

    info "Starting: ${INFRA_SERVICES}"
    docker compose up -d ${INFRA_SERVICES}

    # Wait for each service
    wait_for_http "http://localhost:5432"  "PostgreSQL" 60  || \
        warn "PostgreSQL health check via HTTP not available — checking via pg_isready"

    # Use pg_isready if available
    if command -v pg_isready &>/dev/null; then
        timeout 60 bash -c \
            'until pg_isready -h localhost -p 5432 -U "${POSTGRES_USER:-warehousegpt}"; do sleep 2; done' \
            && success "PostgreSQL ready" || warn "PostgreSQL pg_isready timed out"
    fi

    # Redis
    if command -v redis-cli &>/dev/null; then
        info "Waiting for Redis..."
        timeout 30 bash -c \
            'until redis-cli -h localhost -p 6379 -a "${REDIS_PASSWORD}" ping 2>/dev/null | grep -q PONG; do sleep 1; done' \
            && success "Redis ready" || warn "Redis check timed out"
    fi

    # Neo4j
    wait_for_http "http://localhost:7474/db/data/" "Neo4j" 90 || \
        warn "Neo4j not reachable at :7474"

    # ChromaDB
    wait_for_http "http://localhost:8001/api/v1/heartbeat" "ChromaDB" 60 || \
        warn "ChromaDB not reachable at :8001"
fi

# ---------------------------------------------------------------------------
# Step 2: Digital Twin API
# ---------------------------------------------------------------------------
if [[ "${START_TWIN}" == "true" ]]; then
    header "Starting Digital Twin API"

    # Kill any existing instance
    if [[ -f "${TWIN_PID_FILE}" ]]; then
        OLD_PID=$(cat "${TWIN_PID_FILE}")
        kill "${OLD_PID}" 2>/dev/null || true
        rm -f "${TWIN_PID_FILE}"
    fi

    DT_PORT="${DT_PORT:-8100}"

    info "Launching Digital Twin API on :${DT_PORT} — logs: ${TWIN_LOG}"
    uvicorn warehousegpt.digital_twin.api.main:app \
        --host 0.0.0.0 \
        --port "${DT_PORT}" \
        --log-level info \
        --access-log \
        > "${TWIN_LOG}" 2>&1 &

    TWIN_PID=$!
    echo "${TWIN_PID}" > "${TWIN_PID_FILE}"
    info "Digital Twin API PID: ${TWIN_PID}"

    wait_for_http "http://localhost:${DT_PORT}/health" "Digital Twin API" 30
fi

# ---------------------------------------------------------------------------
# Step 3: WarehouseGPT Agent API
# ---------------------------------------------------------------------------
if [[ "${START_AGENT}" == "true" ]]; then
    header "Starting WarehouseGPT Agent API"

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        warn "ANTHROPIC_API_KEY is not set — agent queries will fail."
    fi

    # Kill any existing instance
    if [[ -f "${AGENT_PID_FILE}" ]]; then
        OLD_PID=$(cat "${AGENT_PID_FILE}")
        kill "${OLD_PID}" 2>/dev/null || true
        rm -f "${AGENT_PID_FILE}"
    fi

    AGENT_PORT="${AGENT_PORT:-8200}"

    info "Launching WarehouseGPT Agent API on :${AGENT_PORT} — logs: ${AGENT_LOG}"
    uvicorn warehouse_agent.api.main:app \
        --host 0.0.0.0 \
        --port "${AGENT_PORT}" \
        --log-level info \
        --access-log \
        > "${AGENT_LOG}" 2>&1 &

    AGENT_PID=$!
    echo "${AGENT_PID}" > "${AGENT_PID_FILE}"
    info "Agent API PID: ${AGENT_PID}"

    wait_for_http "http://localhost:${AGENT_PORT}/health" "Agent API" 30
fi

# ---------------------------------------------------------------------------
# Print service URLs
# ---------------------------------------------------------------------------
header "WarehouseGPT Stack Ready"

echo -e "${BOLD}Infrastructure Services${NC}"
if [[ "${START_DOCKER}" == "true" ]]; then
    echo -e "  PostgreSQL     : ${CYAN}postgresql://localhost:5432/${POSTGRES_DB:-warehousegpt}${NC}"
    echo -e "  Redis          : ${CYAN}redis://localhost:6379${NC}"
    echo -e "  Neo4j Browser  : ${CYAN}http://localhost:7474${NC}"
    echo -e "  Neo4j Bolt     : ${CYAN}bolt://localhost:7687${NC}"
    echo -e "  ChromaDB       : ${CYAN}http://localhost:8001${NC}"
fi

echo ""
echo -e "${BOLD}Application Services${NC}"
if [[ "${START_TWIN}" == "true" ]]; then
    DT_PORT="${DT_PORT:-8100}"
    echo -e "  Digital Twin API       : ${CYAN}http://localhost:${DT_PORT}${NC}"
    echo -e "    GET  /health         : ${CYAN}http://localhost:${DT_PORT}/health${NC}"
    echo -e "    GET  /state          : ${CYAN}http://localhost:${DT_PORT}/state${NC}"
    echo -e "    GET  /state/bev      : ${CYAN}http://localhost:${DT_PORT}/state/bev${NC}"
    echo -e "    GET  /incidents      : ${CYAN}http://localhost:${DT_PORT}/incidents${NC}"
    echo -e "    WS   /stream         : ${CYAN}ws://localhost:${DT_PORT}/stream${NC}"
    echo -e "    GET  /analytics/throughput : ${CYAN}http://localhost:${DT_PORT}/analytics/throughput${NC}"
fi

if [[ "${START_AGENT}" == "true" ]]; then
    AGENT_PORT="${AGENT_PORT:-8200}"
    echo -e "  WarehouseGPT Agent API : ${CYAN}http://localhost:${AGENT_PORT}${NC}"
    echo -e "    GET  /health         : ${CYAN}http://localhost:${AGENT_PORT}/health${NC}"
    echo -e "    POST /query          : ${CYAN}http://localhost:${AGENT_PORT}/query${NC}"
    echo -e "    GET  /query/stream   : ${CYAN}http://localhost:${AGENT_PORT}/query/stream${NC}"
    echo -e "    WS   /chat           : ${CYAN}ws://localhost:${AGENT_PORT}/chat${NC}"
    echo -e "    GET  /sessions       : ${CYAN}http://localhost:${AGENT_PORT}/sessions${NC}"
    echo -e "  API Docs (Swagger)     : ${CYAN}http://localhost:${AGENT_PORT}/docs${NC}"
    echo -e "  API Docs (ReDoc)       : ${CYAN}http://localhost:${AGENT_PORT}/redoc${NC}"
fi

echo ""
info "Press Ctrl-C to stop all services."
echo ""

# ---------------------------------------------------------------------------
# Keep the script alive — wait for child processes
# ---------------------------------------------------------------------------
wait
