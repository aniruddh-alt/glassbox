# GPU Microservice for Inference + SAE — Design

**Date:** 2026-06-20  
**Status:** Approved design, pending spec review → implementation plan  
**Scope (v1):** Lift the already-isolated torch path (`engine` + `science/sae` + `science/feature_provider`) onto a RunPod GPU pod behind a small HTTP service with **three primitive endpoints**. Have the existing orchestration backend call it instead of importing torch locally.

**Deferred (v2+):** Persona-vector probe training pipeline (dataset gen → layer sweep → AUROC → optional ablation). Runs as **offline jobs on the pod**, not as HTTP endpoints.

## 1. Context

`glassbox` is a cognition-observability tool for a medical chatbot (`gemma-3-4b-it`, layer 17, `gemma-scope-2-4b-it-res` SAEs at width 16k). Two signal families:

- **Family A** — SAE feature cloud (exploratory; labels via Neuronpedia on the laptop, CPU + cached).
- **Family B** — calibrated probe trackers (uncertainty/safety; reliable signal). **Not in v1 scope** beyond optionally scoring pre-trained artifacts if they exist.

The backend **already enforces a CPU/GPU split**: `analyze.py`, `runtime.py`, `app.py`, `labels.py`, `events.py`, `schema.py`, `fanout.py` must never import torch (Global Constraint); the torch path (`engine.py`, `science/sae.py`, `science/feature_provider.py`, `science/persona.py`) is imported lazily only when `runtime.STATE["mode"] == "real"`. This design turns that lazy import into a network boundary.

Target scale: hackathon, single-user demo. Throughput is **not** the binding constraint; inference + SAE encode dominate. Serving stays raw HF transformers + forward hooks (no vLLM) — required so the residual stream matches the distribution the SAEs were trained on.

## 2. Two workflows

### Workflow A — Top activating feature labels (v1, core)

The live demo path. GPU does model + SAE; laptop does Neuronpedia labeling + re-rank.

```
[laptop] orchestration  (NO torch)
  pod_client.sae_features(messages)  →  raw candidates [{index, act, source[, attr]}]
  _rank_features(candidates)         →  Neuronpedia labels + ceiling/attribution re-rank
  build_cognition_event(...)         →  UI feature cloud
        │  HTTP (SSH tunnel)
        ▼
[pod] GPU service
  POST /sae/features  →  generate_and_capture → LocalSAEProvider.features_for → candidates
```

This is the **primary reason the service exists**: run Gemma + SAE on GPU, return a small JSON candidate list, label the top features on CPU.

### Workflow B — Persona-vector pipeline (v2, offline on pod)

Full probe lifecycle. **Not exposed as HTTP endpoints** — batch scripts on the pod import `engine` / `science/persona` directly (same modules the service loads). Steps:

1. **Dataset generation** — `validation/build_medqa_dataset.py` (contrastive prompt pairs, labels).
2. **Inference + activation extraction** — `engine.generate_and_capture` at each layer in the sweep (or locked layer once validated).
3. **Probe training** — diff-of-means persona vectors + calibrated LogReg (`science/persona.py`); write `science/artifacts/*.json`.
4. **AUROC eval** — `validation/eval_auroc.py` vs plain LogReg baseline.
5. **(Optional) Causal ablation** — offline, not in the serving path.

Once artifacts exist, v1 can optionally load them at pod startup and `/turn` (or a later `/score_probes`) can call `score_all_trackers`. Training itself stays offline.

## 3. The seam

The network cut for the **live demo** goes **between raw SAE candidates and `_rank_features`** — exactly along the existing torch / no-torch line.

```
[laptop] orchestration  (backend/app.py — NO torch)
   /api/chat, /api/analyze
        │  analyze_turn → _real_turn
        │    r        = pod_client.turn(...) or pod_client.sae_features(...)
        │    feats    = _rank_features(r["candidates"])   ← Neuronpedia, unchanged
        │    trackers = r.get("trackers", {})
        │    event    = build_cognition_event(...)
        ▼
[pod] GPU service  (backend/gpu_service.py — owns torch)
   startup: engine.load_engine() + science.sae.load_sae()
   primitives: /inference | /activations | /sae/features
   convenience: POST /turn  (= /sae/features + optional tracker scoring)
   GET  /health
```

**Wire budget:** full per-token activation tensors (`seq × 2560`) **never cross HTTP**. SAE encode runs on the pod. The `/activations` endpoint returns only **pooled vectors** (`act_last`, `act_resp`, each 2560 floats) — enough for probe *scoring*, not for retraining at scale. Batch training reads activations in-process on the pod.

## 4. Contract (pod service)

Auth on all POST routes: `Authorization: Bearer <POD_TOKEN>`.

### `POST /inference`

Generate assistant text only. No hooks beyond what generation requires.

Request:
```json
{ "messages": [{"role": "user", "content": "..."}], "max_new": 48 }
```
Response:
```json
{ "answer": "string" }
```
Maps to: `engine.generate_and_capture` → return `answer` only (or a thin generate wrapper).

### `POST /activations`

Generate (or forward over an existing completion) and capture layer-`LAYER` `resid_post`. Returns **pooled** activations for probe scoring — not the full sequence tensor.

Request:
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "max_new": 48,
  "attribution": false
}
```
Response:
```json
{
  "answer": "string",
  "resp_start": 42,
  "act_last": [2560 floats],
  "act_resp": [2560 floats]
}
```
Maps to: `engine.generate_and_capture` → mean-pool response positions → JSON lists. Used later for Family B scoring; v1 may omit from the orchestration path if no probe artifacts exist.

### `POST /sae/features`

The **core endpoint** for Family A. Capture + SAE encode + top-k candidate pool on the pod.

Request:
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "max_new": 48,
  "attribution": true,
  "cap": 50
}
```
Response:
```json
{
  "answer": "string",
  "candidates": [
    {"index": 12345, "act": 0.873, "source": "17-gemmascope-2-res-16k", "attr": 0.041}
  ],
  "reliable": true
}
```
Maps to: `generate_and_capture` → mask special token ids → `LocalSAEProvider.features_for(..., cap=TOPK_CANDIDATES)`. Candidates are **raw, unranked, unlabelled**; `labels.py` + `_rank_features` stay on the laptop.

### `POST /turn` (convenience, v1)

Single round-trip for the chat demo: `/sae/features` + optional `score_all_trackers` when probe artifacts are loaded.

Request / response: same as before (§3 of prior draft) — `{ answer, candidates, trackers, reliable }`. Implemented as a thin composition of the primitives internally, not duplicated logic.

### `GET /health`

Returns `{mode, model_loaded, sae_loaded, layer, model, trackers:[names]}`. Orchestration polls this to set `runtime.STATE["mode"]`.

## 5. Components

### New (v1)

- **`backend/gpu_service.py`** — FastAPI app on the pod. Reuses `engine.py` + `science/sae.py` + `science/feature_provider.py` **as-is**. Routes call into existing functions; no forked model logic.
- **`backend/pod_client.py`** — orchestration-side `httpx` client: `inference`, `activations`, `sae_features`, `turn` (wraps `/turn`), `health`. Typed errors for fallback.

### Edited (orchestration)

- **`backend/analyze.py`** — `_real_turn` calls `pod_client.turn(...)` (or `sae_features` + empty trackers). `_rank_features` unchanged.
- **`backend/runtime.py`** — health poll via `pod_client.health()` instead of local torch load.
- **`backend/app.py`** — `/api/health` exposes pod reachability + last pod health.

### Unchanged

`schema.py`, `events.py`, `fanout.py`, `labels.py`, `_rank_features`, `/api/feature` proxy, cosmetic streaming replay, **`engine.py` + `science/*`** (run on pod).

### Later (v2, not v1)

- **`backend/validation/*`** — persona pipeline scripts run **on the pod** via `python -m backend.validation...`, importing torch modules directly.
- Probe artifact hot-reload or `/score_probes` endpoint — only after training exists.

## 6. Error handling / degradation

| Condition | Behaviour |
|---|---|
| Pod reachable, `mode==real` | orchestration `mode=real`; turns routed to pod |
| Pod unreachable | `mode=fallback`; `fallback.synth_turn` |
| Pod up but request errors / timeout | that turn → synthetic; re-check health on next poll |
| Probe artifacts absent | `trackers == {}` |

## 7. Transport & persistence

- **Transport:** SSH local-forward: `ssh -L 8000:localhost:8000 <pod>`. Orchestration: `POD_URL=http://localhost:8000`.
- **Pod process:** `uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000`. Eager load at startup (`GLASSBOX_EAGER_LOAD=1`).
- **Persistence (RunPod volume):** `HF_HOME`, SAE weights, optional `science/artifacts/*.json`.

## 8. Out of scope (explicit)

| Item | When |
|---|---|
| Persona dataset gen, layer sweep, probe training, AUROC | v2 offline jobs on pod |
| Causal ablation tests | v2 optional |
| True per-token streaming | later |
| vLLM / nnsight-vLLM | never for this SAE path (distribution mismatch) |
| Shipping full `seq×d_in` activation tensors over HTTP | never (wire budget) |
| Neuronpedia as remote SAE fallback | pod-down → synthetic fallback |

## 9. Precondition / risk (pre-existing)

Gemma-scope-2 SAE at layer 17 on gemma-3-4b is **unconfirmed**. Run `science.sae.reconstruction_error()` on a captured activation (cosine ≈ 0.9+) before trusting feature output. Orthogonal to microservice plumbing.

## 10. Testing (v1)

- **`pod_client`** — monkeypatched `httpx` for each route + error paths.
- **`gpu_service`** — contract tests with monkeypatched `engine` / `feature_provider`: each primitive returns documented JSON; `/turn` composes correctly.
- **`analyze._real_turn`** — stubbed pod client → `_rank_features` applied (extends `tests/test_rank.py`).
- **`runtime`** — health poll flips `mode` real/fallback.
- Orchestration tests run **without torch installed**.

Real model load: existing `smoke_test_gs2.py` on the pod.

## 11. Implementation priority

1. **`/health` + `/sae/features` + `pod_client` + orchestration wiring** — unblocks live feature cloud + top-k label workflow.
2. **`POST /turn`** — convenience for chat demo (thin wrapper).
3. **`POST /inference` + `/activations`** — explicit primitives for debugging and future probe scoring.
4. **Persona pipeline (workflow B)** — separate milestone; uses same pod modules, not new HTTP surface.
