#!/usr/bin/env bash
# =============================================================
# doctor.sh — System health checker for cvat-sam2
# =============================================================
# Checks: Docker, Compose, GPU, ports, directories, model configs.
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; PASS=$((PASS+1)); }
warn_item() { echo -e "  ${YELLOW}[WARN]${NC} $*"; WARN=$((WARN+1)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; FAIL=$((FAIL+1)); }

section() {
    echo ""
    echo "  ── $* ──────────────────────────────────────"
}

# ──────────────────────────────────────────────────────────
section "Docker"

if command -v docker &>/dev/null; then
    pass "docker binary found: $(docker --version)"
else
    fail "docker not found in PATH. Install: https://docs.docker.com/get-docker/"
fi

if docker info &>/dev/null 2>&1; then
    pass "Docker daemon is running and accessible"
else
    fail "Docker daemon is not running or current user cannot access it"
    echo "     Fix: sudo systemctl start docker"
    echo "     Fix: sudo usermod -aG docker \$USER && newgrp docker"
fi

# ──────────────────────────────────────────────────────────
section "Docker Compose"

if docker compose version &>/dev/null 2>&1; then
    pass "docker compose plugin: $(docker compose version --short)"
elif command -v docker-compose &>/dev/null; then
    warn_item "docker-compose (standalone) found: $(docker-compose --version)"
    echo "     Prefer: Docker Compose V2 plugin (docker compose)"
else
    fail "Docker Compose not found. Install: https://docs.docker.com/compose/install/"
fi

# ──────────────────────────────────────────────────────────
section "NVIDIA GPU"

if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        pass "nvidia-smi found — GPU: ${GPU_NAME}"
    else
        fail "nvidia-smi present but failed (driver issue?)"
    fi
else
    warn_item "nvidia-smi not found — no NVIDIA GPU or driver not installed"
    echo "     SAM2 will run on CPU (slower)"
fi

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 \
        nvidia-smi &>/dev/null 2>&1; then
        pass "Docker GPU access verified (NVIDIA Container Toolkit OK)"
    else
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
            fail "NVIDIA GPU found but Docker cannot access it"
            echo "     Fix: Install NVIDIA Container Toolkit:"
            echo "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        else
            warn_item "GPU Docker test skipped (no GPU detected)"
        fi
    fi
fi

# ──────────────────────────────────────────────────────────
section "Ports"

# Load env if available
if [ -f "${REPO_ROOT}/.env" ]; then
    # shellcheck disable=SC1090
    source <(grep -E '^(CVAT_PORT|SAM2_PORT|ONNX_PORT|NUCLIO_PORT)=' "${REPO_ROOT}/.env" || true)
fi

CVAT_PORT="${CVAT_PORT:-8080}"
SAM2_PORT="${SAM2_PORT:-8000}"
ONNX_PORT="${ONNX_PORT:-8001}"
NUCLIO_PORT="${NUCLIO_PORT:-8070}"

check_port() {
    local port="$1"
    local service="$2"
    if ss -tlnp "sport = :${port}" 2>/dev/null | grep -q "${port}"; then
        warn_item "Port ${port} is already in use (${service})"
        echo "     This may cause conflicts. Change ${service}_PORT in .env"
    else
        pass "Port ${port} is free (${service})"
    fi
}

check_port "${CVAT_PORT}"   "CVAT"
check_port "${SAM2_PORT}"   "SAM2"
check_port "${ONNX_PORT}"   "ONNX runner"
check_port "${NUCLIO_PORT}" "Nuclio"

# ──────────────────────────────────────────────────────────
section "Repository files"

check_file() {
    local f="$1"
    if [ -f "${REPO_ROOT}/${f}" ]; then
        pass "${f} exists"
    else
        fail "${f} is missing"
    fi
}

check_dir() {
    local d="$1"
    if [ -d "${REPO_ROOT}/${d}" ]; then
        pass "${d}/ exists"
    else
        warn_item "${d}/ does not exist (will be created by 'up')"
    fi
}

check_file ".env"
check_file "docker-compose.yml"
check_file "docker-compose.gpu.yml"
check_file "docker-compose.cpu.yml"
check_file "services/sam2/Dockerfile"
check_file "services/onnx-runner/Dockerfile"
check_dir  "models"

# ──────────────────────────────────────────────────────────
section "Model configs"

if [ -d "${REPO_ROOT}/models" ]; then
    found=0
    for d in "${REPO_ROOT}/models"/*/; do
        [ -d "${d}" ] || continue
        found=$((found+1))
        name="$(basename "${d}")"
        if [ -f "${d}model.yaml" ]; then
            pass "models/${name}/model.yaml found"
            if ls "${d}"*.onnx &>/dev/null 2>&1; then
                pass "models/${name}/*.onnx weight found"
            else
                warn_item "models/${name}/ has no .onnx weight file (add model.onnx to enable inference)"
            fi
        else
            fail "models/${name}/ has no model.yaml"
        fi
    done
    if [ "${found}" -eq 0 ]; then
        warn_item "No model directories found in models/ (only SAM2 will be available)"
    fi
else
    warn_item "models/ directory not found (will be created on first 'up')"
fi

# ──────────────────────────────────────────────────────────
section ".env configuration"

if [ -f "${REPO_ROOT}/.env" ]; then
    pass ".env file exists"
    # Check for default passwords
    if grep -q 'ChangeMeNow123!' "${REPO_ROOT}/.env" 2>/dev/null; then
        warn_item "Default CVAT_SUPERUSER_PASS is in use — change before production use"
    fi
    if grep -q 'cvat_db_password_change_me' "${REPO_ROOT}/.env" 2>/dev/null; then
        warn_item "Default POSTGRES_PASSWORD is in use — change before production use"
    fi
else
    warn_item ".env not found — will be created from .env.example on 'up'"
fi

# ──────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Results: ${GREEN}${PASS} passed${NC}  ${YELLOW}${WARN} warnings${NC}  ${RED}${FAIL} failed${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${FAIL}" -gt 0 ]; then
    echo "  Some checks failed. Fix the issues above before running ./cvat-sam2 up"
    exit 1
fi
if [ "${WARN}" -gt 0 ]; then
    echo "  Some warnings found. The stack may still start but review the warnings."
fi
echo ""
