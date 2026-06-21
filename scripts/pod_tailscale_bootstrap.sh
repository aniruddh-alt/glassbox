#!/usr/bin/env bash
# Run on the GPU pod via RunPod web terminal (browser SSH — works on eduroam).
# Brings up Tailscale userspace + restarts gpu_service if present.
set -euo pipefail

cd /workspace/glassbox 2>/dev/null || cd /workspace || true

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | bash
fi

if pgrep -f 'tailscaled --tun=userspace-networking' >/dev/null; then
  echo "tailscaled already running — skipping start"
else
  nohup /usr/sbin/tailscaled \
    --tun=userspace-networking \
    --state=/workspace/tailscale.state \
    > /workspace/tailscaled.log 2>&1 &
  sleep 5
fi

echo ">>> Click the auth URL if prompted, then note the 100.x.x.x IP:"
tailscale up --ssh --hostname=glassbox-gpu

echo
echo "Tailscale IP: $(tailscale ip -4)"
echo "Hostname:     $(tailscale status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName',''))" 2>/dev/null || true)"

if [[ -x /workspace/glassbox/.venv/bin/uvicorn ]]; then
  export HF_HOME=/workspace/.cache/huggingface
  export GLASSBOX_EAGER_LOAD=1
  export POD_TOKEN="${POD_TOKEN:-glassbox-dev-secret}"
  export PYTHONPATH=/workspace/glassbox
  pkill -f 'uvicorn backend.gpu_service' 2>/dev/null || true
  nohup /workspace/glassbox/.venv/bin/uvicorn backend.gpu_service:app \
    --host 127.0.0.1 --port 8000 > /workspace/glassbox/gpu_service.log 2>&1 &
  sleep 3
  curl -s http://127.0.0.1:8000/health && echo
fi
