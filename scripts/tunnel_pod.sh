#!/usr/bin/env bash
# Option A: SSH tunnel to gpu_service over Tailscale (bypasses eduroam → RunPod TCP block).
# Usage: ./scripts/tunnel_pod.sh 100.x.x.x
set -euo pipefail

POD_TS_IP="${1:?usage: $0 <pod-tailscale-ip>}"
LOCAL_PORT="${LOCAL_PORT:-8001}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
if [[ -x "$TS_CLI" ]] && "$TS_CLI" status 2>&1 | grep -q 'Logged out'; then
  echo "Tailscale is logged out. Open the Tailscale menu bar app and sign in first."
  exit 1
fi

echo "Tunneling localhost:${LOCAL_PORT} -> ${POD_TS_IP}:127.0.0.1:${REMOTE_PORT}"
echo "Press Ctrl+C to stop."

exec ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 \
  "root@${POD_TS_IP}"
