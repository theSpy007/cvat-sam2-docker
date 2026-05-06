#!/usr/bin/env bash
# =============================================================
# wait-for-health.sh — Wait until a compose service is healthy
# =============================================================
# Usage: ./wait-for-health.sh <service> [max_seconds]
# =============================================================

set -euo pipefail

SERVICE="${1:?Usage: wait-for-health.sh <service> [max_seconds]}"
MAX="${2:-120}"
INTERVAL=5
ELAPSED=0

echo "[wait-for-health] Waiting for ${SERVICE} to be healthy (max ${MAX}s) ..."

while [ "${ELAPSED}" -lt "${MAX}" ]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' \
        "$(docker compose ps -q "${SERVICE}" 2>/dev/null | head -1)" 2>/dev/null || true)
    if [ "${STATUS}" = "healthy" ]; then
        echo "[wait-for-health] ${SERVICE} is healthy."
        exit 0
    fi
    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "[wait-for-health] Still waiting ... (${ELAPSED}s / ${MAX}s, status=${STATUS:-unknown})"
done

echo "[wait-for-health] ERROR: ${SERVICE} did not become healthy within ${MAX}s."
exit 1
