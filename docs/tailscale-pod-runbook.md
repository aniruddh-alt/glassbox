# Tailscale GPU Pod Runbook

How to run GlassBox on **eduroam** (or any network that blocks outbound SSH to RunPod’s public IP) by reaching the GPU pod over a **Tailscale tailnet** instead of `74.2.x.x:15331`.

Use this doc when chat stops working, health shows `mode: fallback`, or you’ve restarted your laptop / pod.

---

## Architecture (Option A)

```
┌──────────────── Mac (laptop) ────────────────┐
│  Frontend :5173                             │
│       │ proxies /api                         │
│       ▼                                      │
│  backend.app :8000  (CPU orchestration)     │
│       │ HTTP POD_URL=http://localhost:8001   │
│       ▼                                      │
│  ssh -L 8001:127.0.0.1:8000                  │
│       │ over Tailscale (UDP/WireGuard)       │
└───────┼──────────────────────────────────────┘
        │ tailnet only — NOT RunPod public TCP
        ▼
┌──────────────── RunPod GPU pod ──────────────┐
│  tailscaled (userspace)  100.98.245.123      │
│  gpu_service :8000     (torch + SAE + model) │
└──────────────────────────────────────────────┘
```

| Component | Where | Port | Must stay running? |
|-----------|-------|------|--------------------|
| `tailscaled` + `gpu_service` | Pod | 8000 (localhost on pod) | Yes (daemonized with `nohup`) |
| SSH tunnel | Mac terminal A | `localhost:8001` → pod:8000 | Yes |
| `backend.app` | Mac terminal B | `localhost:8000` | Yes |
| Vite frontend | Mac terminal C | `localhost:5173` | Yes (for UI) |
| RunPod **web terminal** | Browser | — | **No** (close after daemonizing) |

**Do not use** RunPod public SSH (`ssh -p 15331 root@74.2.96.15`) on eduroam — it times out. Use the pod’s **Tailscale IP** instead.

---

## Fixed reference (your setup)

| Item | Value |
|------|-------|
| Tailscale account | `aniruddhr04@` |
| Mac hostname | `aniruddhans-macbook-pro` |
| Mac Tailscale IP | `100.99.83.110` |
| Pod hostname | `glassbox-gpu` |
| Pod Tailscale IP | `100.98.245.123` *(may change after full pod recreate + re-auth)* |
| Pod code path | `/workspace/glassbox` |
| `POD_TOKEN` | `glassbox-dev-secret` |
| `POD_URL` (on Mac) | `http://localhost:8001` |

After a **full pod terminate + new pod**, the Tailscale IP may change. Always run `tailscale ip -4` on the pod and update `tunnel_pod.sh` argument.

---

## One-time setup

### 1. Mac — install & sign in to Tailscale

```bash
brew install --cask tailscale
# Open Tailscale from menu bar → Log in
/Applications/Tailscale.app/Contents/MacOS/Tailscale status
# Should show your Mac, NOT "Logged out"
```

### 2. Pod — Tailscale + gpu_service (RunPod **web terminal**)

Open **RunPod console → your pod → Web Terminal** (works on eduroam; public SSH does not).

Either paste the repo script (after `rsync`/deploy), or run manually:

```bash
cd /workspace/glassbox

# Install Tailscale (first time only)
curl -fsSL https://tailscale.com/install.sh | bash

# Start tailscaled (skip if already running — see troubleshooting)
nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state=/workspace/tailscale.state \
  > /workspace/tailscaled.log 2>&1 &
sleep 5

# First time: click the auth URL. Same Tailscale account as your Mac.
tailscale up --ssh --hostname=glassbox-gpu

tailscale ip -4    # note this IP — e.g. 100.98.245.123

# Start GPU service (model + SAE)
export HF_HOME=/workspace/.cache/huggingface
export GLASSBOX_EAGER_LOAD=1
export POD_TOKEN=glassbox-dev-secret
export PYTHONPATH=/workspace/glassbox
pkill -f 'uvicorn backend.gpu_service' 2>/dev/null || true
nohup .venv/bin/uvicorn backend.gpu_service:app \
  --host 127.0.0.1 --port 8000 \
  > /workspace/glassbox/gpu_service.log 2>&1 &
sleep 10
curl -s http://127.0.0.1:8000/health
```

Expected pod health:

```json
{"mode":"real","model_loaded":true,"sae_loaded":true,...}
```

Or use the helper script from repo root on the pod:

```bash
bash /workspace/glassbox/scripts/pod_tailscale_bootstrap.sh
```

**You can close the web terminal** after `nohup` — processes survive.

---

## Daily restart (most common)

Three terminals on your Mac, **in this order**:

### Terminal A — SSH tunnel (start first)

```bash
cd ~/.superset/worktrees/03bd06ac-9a5b-4bcf-8fd0-d4b5fdf19201/chatbot-sae-integration

./scripts/tunnel_pod.sh 100.98.245.123
# equivalent:
# ssh -N -L 8001:127.0.0.1:8000 root@100.98.245.123
```

Leave running. No output is normal.

### Terminal B — local backend (start after tunnel)

```bash
cd ~/.superset/worktrees/03bd06ac-9a5b-4bcf-8fd0-d4b5fdf19201/chatbot-sae-integration

./scripts/run_with_pod.sh
```

Refuses to start if the tunnel isn’t up (checks `localhost:8001/health`).

### Terminal C — frontend (if not already running)

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173**

---

## Health checks (run in 30 seconds)

```bash
# 1. Mac on tailnet?
/Applications/Tailscale.app/Contents/MacOS/Tailscale status
# Must list glassbox-gpu (100.98.245.123)

# 2. Pod reachable via tunnel?
curl -s http://localhost:8001/health
# → mode: real, model_loaded: true

# 3. Local backend wired correctly?
curl -s http://localhost:8000/api/health
# → mode: real, pod_reachable: true

# 4. Quick chat smoke (optional, ~60–90s)
curl -sN -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}' | tail -1
# Last line should be type: event with real features
```

### Good vs bad

| Symptom | Meaning | Fix |
|---------|---------|-----|
| `curl :8001/health` fails | Tunnel down or pod `gpu_service` dead | Terminal A, then pod web terminal |
| `:8001` OK but `:8000/api/health` shows `fallback` | Backend started **before** tunnel, or old process | Kill backend, restart `./scripts/run_with_pod.sh` **after** tunnel |
| `pod_reachable: false` | Same as above | Restart backend after tunnel |
| Chat returns gibberish / echoes prompt | Synthetic fallback mode | Fix tunnel + restart backend |
| `tailscale status` → `Logged out` | Mac Tailscale off | Open menu bar app → sign in |
| `ssh root@100.98.245.123` hangs | Pod `tailscaled` dead | Pod web terminal → restart tailscaled |
| `nohup tailscaled` → Exit 1 | **Already running** — not an error | Ignore; run `tailscale up` only |

---

## Restart only the pod side

Use **RunPod web terminal** when the Mac can’t reach `100.98.245.123` or `curl` on the pod’s localhost:8000 fails.

```bash
# Is tailscaled running?
pgrep -af tailscaled || echo "NOT RUNNING"

# Restart tailscaled
pkill -f 'tailscaled --tun=userspace-networking' 2>/dev/null || true
nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state=/workspace/tailscale.state \
  > /workspace/tailscaled.log 2>&1 &
sleep 5
tailscale up --ssh --hostname=glassbox-gpu
tailscale ip -4

# Is gpu_service running?
pgrep -af gpu_service || echo "NOT RUNNING"
curl -s http://127.0.0.1:8000/health

# Restart gpu_service
cd /workspace/glassbox
export HF_HOME=/workspace/.cache/huggingface GLASSBOX_EAGER_LOAD=1
export POD_TOKEN=glassbox-dev-secret PYTHONPATH=/workspace/glassbox
pkill -f 'uvicorn backend.gpu_service' 2>/dev/null || true
nohup .venv/bin/uvicorn backend.gpu_service:app \
  --host 127.0.0.1 --port 8000 \
  > gpu_service.log 2>&1 &
sleep 15
tail -20 gpu_service.log
curl -s http://127.0.0.1:8000/health
```

Then on Mac: restart tunnel + backend (daily restart section).

---

## Restart only the Mac side

When pod health works from web terminal but UI is broken:

```bash
# Kill stale processes
pkill -f 'ssh -N -L 8001' 2>/dev/null || true
pkill -f 'uvicorn backend.app' 2>/dev/null || true

# Restart A → B → C (see Daily restart)
./scripts/tunnel_pod.sh 100.98.245.123
# new terminal:
./scripts/run_with_pod.sh
```

---

## Full cold start (pod was stopped/terminated)

1. Start / resume RunPod pod.
2. **Pod web terminal** — full pod setup (one-time setup §2). You may need to click Tailscale auth URL again.
3. Note new `tailscale ip -4` if it changed.
4. **Mac** — daily restart with updated IP.
5. First model load can take several minutes; watch `gpu_service.log` on pod.

### What persists in `/workspace`

| Persists | Lost on new pod |
|----------|-----------------|
| `/workspace/tailscale.state` | Container OS packages (re-run install if needed) |
| `/workspace/.cache/huggingface` (model weights) | Running processes |
| `/workspace/glassbox` (code, venv) | In-memory state |

---

## Environment variables (Mac)

```bash
export POD_URL=http://localhost:8001
export POD_TOKEN=glassbox-dev-secret
export GLASSBOX_EAGER_LOAD=1          # poll pod health at backend startup
# optional:
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

`run_with_pod.sh` sets these automatically.

---

## Known gotchas

1. **Eduroam blocks RunPod public SSH** — always use Tailscale IP `100.x.x.x`, never `74.2.96.15:15331`.

2. **`tailscale ssh -N -L` does not work** — use plain `ssh -L` to the tailnet IP. `scripts/tunnel_pod.sh` does this.

3. **Backend startup order matters** — tunnel first, then `run_with_pod.sh`. With `GLASSBOX_EAGER_LOAD=1`, if the tunnel isn’t up at startup, backend stays in `fallback` until restarted.

4. **Second `tailscaled` exits with code 1** — means one is already running. Don’t panic.

5. **Tailscale auth in containers** — browser auth URL per cold start is normal; auth keys are unreliable for `--ssh` in unprivileged containers. State file in `/workspace` speeds rejoin.

6. **Userspace networking** — pod has no kernel TUN; `gpu_service` stays on `127.0.0.1:8000` on the pod. Mac reaches it via SSH `-L`, not direct `http://100.x.x.x:8000` (unless you later switch to Option B).

7. **Closing web terminal is fine** after `nohup`. Closing Mac tunnel/backend terminals is not.

8. **Laptop sleep** — SSH tunnel drops. Wake → restart tunnel + backend.

---

## Optional: Phoenix observability

```bash
uv run python -m phoenix.server.main serve --host 127.0.0.1 --port 6006
# UI: http://localhost:6006
```

Fanout errors about Phoenix don’t block chat; they only affect tracing.

---

## Quick copy-paste cheat sheet

```bash
# === MAC (3 terminals) ===
./scripts/tunnel_pod.sh 100.98.245.123
./scripts/run_with_pod.sh
cd frontend && npm run dev

# === VERIFY ===
curl -s localhost:8001/health && curl -s localhost:8000/api/health

# === POD (web terminal, if pod side dead) ===
nohup /usr/sbin/tailscaled --tun=userspace-networking --state=/workspace/tailscale.state > /workspace/tailscaled.log 2>&1 &
sleep 5 && tailscale up --ssh --hostname=glassbox-gpu && tailscale ip -4
cd /workspace/glassbox && nohup .venv/bin/uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000 > gpu_service.log 2>&1 &
```
