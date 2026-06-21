# Interpretability Agent — Platform Integration Design

**Date:** 2026-06-21
**Status:** Approved (ready for implementation)
**Branch:** `chatbot-sae-integration`

## 1. Goal & scope

Wire the Interpretability Agent (`agent/interp_agent.py` + `science/concept_synth.py`) into
the running GlassBox platform so a natural-language monitoring request becomes a **live**
persona-vector probe that scores subsequent chat turns. Today the agent is code-complete and
unit-tested but **not host-correct**: `/api/track` runs it on the CPU backend, where (a) it tries
to generate locally instead of on the GPU pod, and (b) `finalize` registers the probe into the
CPU process's `persona._trackers`, which is never used — live scoring happens in the **pod**
process (`gpu_service.py:/turn` → `score_all_trackers`).

A local end-to-end smoke test (the `C` step) proved the agent loop works and surfaced one real
bug, now fixed (see §2).

## 2. Prerequisite fix (done)

`concept_synth.fit_and_validate` called `.numpy()` directly on activation tensors that live on the
model device, raising `can't convert mps:0 / cuda:0 device type tensor to numpy`. This broke the
agent path on **any** accelerator (MPS locally and CUDA on the pod). Fixed by moving the stacked
activations to CPU once at construction:
`X = torch.stack([...]).cpu()`. The local smoke test (`smoke_test_agent.py`, `AGENT_MAX_QUESTIONS=8`)
then completed green (`status=ready`, AUROC 1.0 / baseline 1.0 on 12 rows — saturated as expected
on a small budget; the agent's verdict flagged this honestly).

## 3. Decisions

- **Topology = Approach A:** the agent runs on the **pod** (`gpu_service`), where the model and the
  live `_trackers` already are. The CPU backend's `/api/track` proxies to the pod.
- **Probe persistence = ephemeral.** Agent-trained probes register in the pod's in-memory
  `persona._trackers`; a `gpu_service` restart clears them (built-in JSON artifacts reload). No
  artifact writeback for now (YAGNI for the demo).
- **Frontend = include the poll fix.** Replace `ProbePanel`'s one-shot `pollTracker()` with an
  interval poll until `ready`/`rejected` so the UI shows a probe go from computing → live.
- **Pod restart = now.** After rsync, restart `gpu_service` immediately to run the real test
  (accepts ~1–3 min of `loading`/synthetic-fallback downtime).

## 4. Changes by file

| File | Change |
|---|---|
| `backend/science/concept_synth.py` | Device fix (done). |
| `backend/gpu_service.py` | Add `POST /api/track`, `GET /api/track/{tracker_id}`, and `_launch_agent` (mirror `app.py:112-139`). The agent's `generate_fn` defaults to local `engine.generate_and_capture` (correct on the pod). `finalize` registers into this process's `persona._trackers`. The job registry (`concept_synth._jobs`) lives here. Endpoints behind `_require_auth`. |
| `backend/pod_client.py` | Add `track(request)` → `POST /api/track` and `track_status(tracker_id)` → `GET /api/track/{id}`. |
| `backend/app.py` | `/api/track` + `/api/track/{tracker_id}` proxy to the pod via `pod_client`. On pod error or no `POD_URL`, return a clear non-fatal status (e.g. `{"status":"unavailable"}`) — never crash the request. The local `_launch_agent`/`concept_synth` import path is removed from the request path. |
| `frontend/src/components/ProbePanel.tsx` | `define()` polls `pollTracker()` on an interval (e.g. every ~3s, cap ~5 min) until `status` is `ready`/`rejected`/`unknown`, updating the row's `computing`/`auroc`/`name` as it goes. |
| `backend/tests/` | Add a `pod_client`-mocked test that `/api/track` proxies create + status and degrades gracefully when the pod is down. |
| Pod env | `ANTHROPIC_API_KEY` set on the pod (user's key → user's pod) so the agent can call Claude from there. |

## 5. Data flow

```
POST /api/track {request}     (CPU backend, app.py)
  → pod_client.track(request) → pod POST /api/track
      → concept_synth.create_job(request)         (pod process)
      → _launch_agent: asyncio.to_thread(run_interp_agent)   (pod bg thread)
            submit_spec → generate_contrastive (local gemma, CUDA)
            → judge_filter (Claude) → fit_and_validate → finalize
            → persona.register_tracker into POD _trackers   (if AUROC ≥ TRACK_AUROC_TAU)
  ← {tracker_id, status:"pending"}

GET /api/track/{id} (CPU) → pod_client.track_status → pod GET /api/track/{id}
  ← {status, progress, auroc, baseline_auroc, verdict, ...}

Next POST /api/chat → analyze_turn → pod_client.turn → gpu_service /turn
  → score_all_trackers now includes the new probe → cognition_event.trackers
  → frontend renders the live probe row.
```

## 6. Deployment

```bash
# 1. rsync code to the pod (over the tunnel or Tailscale IP)
rsync -avz --exclude .venv --exclude __pycache__ backend/ root@<pod>:/workspace/glassbox/backend/

# 2. restart gpu_service with the Anthropic key in env
ssh root@<pod> 'pkill -f "uvicorn backend.gpu_service" || true; cd /workspace/glassbox && \
  export HF_HOME=/workspace/.cache/huggingface POD_TOKEN=glassbox-dev-secret \
         PYTHONPATH=/workspace/glassbox ANTHROPIC_API_KEY=<key> && \
  nohup .venv/bin/uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000 \
        > gpu_service.log 2>&1 &'

# 3. confirm healthy, then real test
curl -s localhost:8001/health
curl -s -X POST localhost:8000/api/track -H 'Content-Type: application/json' \
  -d '{"request":"Watch for the model being sycophantic toward the clinician."}'
# poll /api/track/{id} until ready, then send a chat turn and confirm the tracker appears
```

## 7. Testing

- The 17 CPU/torch tests stay green.
- New mocked `pod_client` proxy test (create + status + pod-down fallback).
- Real end-to-end: `/api/track` against the platform → poll to `ready`/`rejected` → chat turn shows
  the new tracker in `cognition_event.trackers`.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Pod restart drops live demo for ~1–3 min | Accepted; restart off-demo if needed. |
| Agent's ~80 generations slow the pod / block a turn | Runs in a bg thread off the request path; CUDA is fast; `AGENT_MAX_QUESTIONS` clamps for demos. |
| `ANTHROPIC_API_KEY` on the pod | User's own key to user's own pod, set via env at restart (not committed). |
| Probe lost on restart | Accepted (ephemeral decision); built-in artifacts reload. |
| Claude sees synthetic medical Q + gemma responses (not PHI) | Within the trust boundary for the demo (public/synthetic data). |

## 9. Non-goals

- Persisting agent-trained probes to JSON artifacts.
- Changing the offline `harmfulness_pipeline` or built-in probes.
- The Claude honesty-judge / auto-interp labeling.
