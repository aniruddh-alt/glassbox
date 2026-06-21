#!/usr/bin/env bash
# Start local backend wired to a tunneled GPU pod.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export POD_URL="${POD_URL:-http://localhost:8001}"
export POD_TOKEN="${POD_TOKEN:-glassbox-dev-secret}"
export GLASSBOX_EAGER_LOAD="${GLASSBOX_EAGER_LOAD:-1}"

if ! curl -sf --max-time 3 "${POD_URL}/health" >/dev/null; then
  echo "Pod not reachable at ${POD_URL}/health"
  echo "Start the tunnel first: ./scripts/tunnel_pod.sh <100.x.x.x>"
  exit 1
fi

echo "Pod OK at ${POD_URL}"
exec uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
