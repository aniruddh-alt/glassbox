# GlassBox

**Cognition-observability for medical LLMs.** A clinician chats with an open model; GlassBox surfaces the model's *internal state* — an SAE "feature cloud" of what concepts are firing, plus calibrated probes that flag when the model is **internally uncertain but verbally confident** (the confident-wrong zone). Every message emits one `cognition_event` that fans out to Sentry, Arize Phoenix, the UI, and (when flagged) a Claude honesty-judge.

> UC Berkeley AI Hackathon · 24h · 3 lanes (A-Backend / B-Science / C-Frontend)
> **Surface uncertainty, never suppress it.**

---

## The one-paragraph architecture

A clinician chats with `unsloth/gemma-3-4b-it` over **POST + NDJSON** (never SSE — it breaks through Cloudflare). On the GPU, the model generates while **one forward hook on `model.model.layers[17]`** captures the residual stream. That single activation feeds **two method families**:

- **Family A — SAE feature cloud** (Gemma Scope `layer_17/width_16k`): top-k firing features → the exploratory "what's lighting up" view. *Labels are auto-interp and unreliable — always shown with caveats.*
- **Family B — persona-vector probes** (diff-of-means + calibrated logistic regression at layer 17): the **reliable** uncertainty / harmful / hallucination scores, plus **user-defined concepts** computed on demand. *(In progress — scores emit `null` until probes are trained.)*

Downstream on **CPU**, the FastAPI handler assembles **exactly one `cognition_event`** per message and fans it out to four consumers that never touch the GPU: **Sentry** (Issue), **Arize Phoenix** (span + eval), the **chat UI**, and **Claude** (auto-interp labels + async adjudication of flagged events).

```
UI ──POST /api/chat──▶ [GPU] gemma-3-4b-it generate + layer-17 hook
                            │  (one residual activation)
                ┌───────────┴───────────┐
        [GPU] SAE top-k          [GPU] persona-vector probes
         (Family A: cloud)        (Family B: reliable scores)
                └───────────┬───────────┘
                     [CPU] build ONE cognition_event
                            │ fanout()
        ┌──────────┬────────┴────────┬──────────────┐
      Sentry    Phoenix          chat UI       Claude judge (async, if flagged)
```

**Model decision (LOCKED):** `unsloth/gemma-3-4b-it` (layer 17) + Gemma Scope. `unsloth/gemma-3-4b-it` is ungated and loads without a HuggingFace token. Llama-3.1-8b was rejected — Neuronpedia label coverage is a verified tie, so 8b's ~4× VRAM / ~3–4× slower tok/s buys nothing, and the safety signal (Family B) is model-agnostic.

---

## The 4 interface contracts (freeze in hour 1, then build in parallel)

1. **Shape contract** — `backend/schema.py` (pydantic `CognitionEvent`) is the source of truth. `frontend/src/types.ts` and `fixtures/cognition_event.sample.json` mirror it. No lane changes the shape without 3-way agreement.
2. **A↔C wire protocol** — `POST /api/chat` returns `application/x-ndjson`: zero-or-more `{"type":"token",...}` lines, then exactly one `{"type":"event", ...CognitionEvent}`. **Frontend builds fully against `fixtures/` before Backend streams real data.**
3. **A↔B function contract** — Science exposes three torch-only functions Backend imports: `sae_topk(act, k=15)`, `score_all_trackers(act_last, act_resp)`, `synth_concept(name, desc)`. The **layer-12 activation tensor is the only object crossing the GPU→science boundary.** B never imports FastAPI; A never touches torch internals.
4. **A↔sponsors seam** — `backend/fanout.py:fanout(event: dict)` is the single place every sponsor SDK lives. Adding/removing a sponsor = editing only `fanout.py`. Nothing on this path imports torch.

---

## Quickstart

```bash
# 1. Install dependencies
uv sync                   # light install (no ML stack — runs in synthetic fallback mode)
uv sync --extra ml        # full install (torch + sae_lens — required for real SAE activations)

# 2. Configure environment
cp .env.example .env      # fill in ANTHROPIC_API_KEY; SENTRY_DSN is optional

# 3. Start the backend  (port 8000)
uv run uvicorn backend.app:app --port 8000
#  → Without the ml extra (or when model weights are absent), the backend
#    automatically falls back to synthetic mode: health reports "mode":"fallback",
#    features are generated synthetically, and uncertainty scores are null.
#  → With the ml extra and weights present, health reports "mode":"real".

# 4. (Optional) Observability stack
bash scripts/run_phoenix.sh         # Arize Phoenix UI on :6006

# 5. Frontend
cd frontend && npm install && npm run dev   # Vite on :5173, proxies /api -> :8000
```

**Verify startup:** `curl -s localhost:8000/api/health` should return:
```json
{"mode":"fallback","model":"unsloth/gemma-3-4b-it","layer":17,"trackers":[]}
```
(or `"mode":"real"` with the full ml stack).

`GLASSBOX_MODE=posthoc` (default) analyzes each completed turn; `GLASSBOX_MODE=live` streams per-token features. **Demo runs local** — do not stream through a RunPod/Cloudflare proxy.

---

## Verified gotchas (don't relearn these at hour 5)

- **`hidden_states[0]` is the embedding** — sanity-check that `hidden_states[12]` matches SAELens `hook_resid_post` layer-12 numbering once before trusting that both families share one hook.
- **Gemma is gated** — pre-accept the license + use a *plain* read token, or fall back to `unsloth/gemma-2-2b-it` (ungated, identical weights, same feature indices).
- **Neuronpedia bulk export is removed** (400s) — use the keyless per-feature GET (cached server-side) or a pre-pulled S3 v1 dump.
- **Claude adjudication runs async** (`asyncio.create_task`) *after* the event line is emitted, so it never blocks the stream.
- **`ObservabilityView` is not a custom dashboard** — it's linkout/iframe cards to the live Sentry project + Phoenix (`localhost:6006`).

See `LANES.md` for who-owns-what and the hour-by-hour build order.
