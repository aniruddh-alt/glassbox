# GlassBox — Chatbot ↔ SAE Integration Design

**Date:** 2026-06-20
**Status:** Approved (ready for implementation planning)
**Branch:** `chatbot-sae-integration`

## 1. Goal & scope

Turn the mock GlassBox demo into a working product. A clinician holds a **multi-turn
chat** with the real model. Each assistant message is run **post-hoc through the SAE**
to extract the top firing features over the response tokens; those features are
**labelled** (Neuronpedia), assembled into **one `cognition_event`**, streamed to the
frontend, and the firing features **illuminate in the feature field** (the "feature map").

Family B (calibrated uncertainty/safety probes) is explicitly **WIP** for this task:
the signal is **empty (null)** when no probes are registered, and the probe panel shows
a placeholder. The chat, SAE feature extraction + labelling, cognition-event assembly,
streaming, and feature-map highlighting are the deliverables.

### Non-goals (explicitly deferred)
- Persona-vector training / calibrated probes (Family B science).
- The Claude honesty-judge (`fanout._claude_judge` stays a no-op; it only fires on
  `flag == true`, which cannot happen while uncertainty is null).
- Claude auto-interp feature labeling (Neuronpedia is the label source for now).
- Live per-token feature firing (we do post-hoc, animate-on-completion).
- Real user-defined concept synthesis (`synth_concept` stays unimplemented; the
  "define a probe" box remains a marked client-side preview).

## 2. Current state (what we build on)

Validated/working per `PLAN.md` and the smoke tests (in a separate ML env):
- `engine.py` — model load + single forward hook on `config.LAYER` (17) +
  `generate_and_capture()`.
- `science/sae.py` — `load_sae()`, `sae_topk()`, `reconstruction_error()`.
- `science/feature_provider.py` — `LocalSAEProvider` (primary) + `NeuronpediaProvider`
  (fallback) + `get_provider()`.
- `labels.py` — Neuronpedia keyless per-feature label cache (`get_label`).
- `events.py` — pure-CPU `build_cognition_event()`.
- `fanout.py` — the single sponsor seam (Sentry + Phoenix wired; Claude judge is a TODO).
- Frontend (`frontend/src/*`) — polished `App` / `ChatPanel` / `FeatureField` (the
  canvas feature map) / `ProbePanel` / `AdjudicationBanner`, currently running in
  `mock: true` against `DEMO_EVENT`.

The integration gap: `app.py:/api/chat` replays the fixture; the frontend is in mock
mode. Nothing real flows through.

**Two constraints that shape the design:**
1. The heavy ML stack is **not installed in this worktree** (`pyproject.toml` declares
   only `fastapi` + `sentry-sdk`; `import torch` fails). It runs on a separate
   GPU/MPS box.
2. **No persona vectors are trained** — `persona._trackers` is empty, so
   `score_all_trackers()` returns `{}`.

## 3. Backend readiness model (per-request)

Because gemma-3-4b + the SAE are an ~8GB download and slow to load, and torch may be
absent, backend path selection is **per-request**:

- At startup, a background thread attempts: `import torch` → `engine.load_engine()` →
  `science.sae.load_sae()`. It records `_state.mode ∈ {"loading", "real", "fallback"}`
  (and `model_loaded`, `sae_loaded`). A failed import or load → `"fallback"`, never a
  crash; the app always starts.
- `/api/chat` and `/api/analyze` choose the path by **current** `_state.mode`: the
  **real** pipeline when `"real"`, the **synthetic fallback** when `"loading"` or
  `"fallback"`.
- `/api/health` reports `{mode, model_loaded, sae_loaded, model, layer, trackers}` so
  the frontend can show an unambiguous badge for what produced an answer.

This is the chosen "graceful fallback": identical pipeline *shape* both ways, demoable
on this Mac today, real on a GPU box by simply having the weights present.

## 4. Backend data flow (`/api/chat`)

```
POST /api/chat {messages:[{role,content}...]}

REAL path (_state.mode == "real"):
  res   = engine.generate_and_capture(messages)        # {answer, acts[seq,d_in], out_ids, resp_start}
  feats = provider.features_for(                         # LocalSAEProvider
            text=res.answer,
            activations=res.acts[res.resp_start:],
            token_ids=res.out_ids[res.resp_start:],
            k=config.TOPK, cap=config.TOPK_EVENT)         # masks special tokens, dedups, ranks
  for f in feats: f["label"] = labels.get_label(f["index"])   # Neuronpedia, cached; "feature N" on miss
  act_last = res.acts[res.resp_start - 1]                      # last prompt token
  act_resp = res.acts[res.resp_start:].mean(0)                 # mean-pooled response tokens
  trackers = persona.score_all_trackers(act_last, act_resp)   # {} until probes exist
  event = events.build_cognition_event(..., trackers=trackers, features=feats,
                                       model=config.MODEL_ID, layer=config.LAYER)

FALLBACK path (_state.mode in {"loading","fallback"}):
  answer = deterministic templated response (seeded by prompt hash)
  feats  = deterministic synthetic features (indices/acts seeded by prompt hash),
           labelled via labels.get_label (offline-safe → "feature N" if no network)
  trackers = {}
  event  = events.build_cognition_event(...)             # uncertainty=None, flag=False

BOTH paths then:
  stream zero+ {"type":"token","text":...} lines      # post-hoc: replay answer in word chunks
  stream exactly one {"type":"event", ...CognitionEvent} line
  AFTER the event line: fanout(event.model_dump())     # Sentry/Phoenix; judge no-ops (no flag)
```

A shared core (e.g. `analyze_turn(messages) -> CognitionEvent` + `answer`) backs both
`/api/chat` (streaming) and `/api/analyze` (single JSON response).

The event's `model` / `layer` / feature `source` are aligned to `config`
(`unsloth/gemma-3-4b-it` · layer 17 · `17-gemmascope-2-res-16k`), replacing the stale
gemma-2 / layer-12 defaults baked into the schema, fixture, and frontend header.

### Other endpoints
- `/api/health` — real load state (see §3).
- `/api/feature/{index}` — real label via `labels.get_label`.
- `/api/track` / `/api/track/{id}` — left as-is (WIP; `synth_concept` unimplemented).

## 5. Contract change (schema.py + types.ts + fixture, in lockstep)

Make the Family-B fields nullable so "no probes registered → empty":

| Field | Before | After |
|---|---|---|
| `uncertainty` | `float` (required) | `float \| None = None` |
| `uncertainty_proj` | `float` (required) | `float \| None = None` |
| `flag` | `bool` (required) | `bool = False` |
| `severity` | `Severity` | defaults `"info"` |
| `trackers` | `{}` default | `{}` when none registered (unchanged shape) |

`events.build_cognition_event`: when the `uncertainty` tracker is absent, leave
`uncertainty`/`uncertainty_proj`/`uncertainty_proj_pre` null, `flag=False`,
`severity="info"`. `frontend/src/types.ts` mirrors the nullability
(`uncertainty: number | null`, etc.). `fixtures/cognition_event.sample.json` is updated
to the gemma-3 / layer-17 source values and serves as a backend test/contract reference
(it stays a flagged example to exercise the non-null branch). `frontend/src/mock.ts`
(`DEMO_EVENT`) is no longer the runtime path once `mock:false`, but is retained for the
`isSuspect` helper and offline replay; its source values are aligned to gemma-3 / L17.

## 6. Frontend

- **`mock: false`** — drive the chat off real `/api/chat` via the existing
  `useCognitionStream` NDJSON reader.
- **Multi-turn thread** — lift a `messages[]` array into `App`. `ChatPanel` renders the
  full thread (it currently shows a single Q+A). Each send appends the user turn and
  streams the assistant turn; the cognition stage (feature field / probes / verdict)
  reflects the **latest** event.
- **Header** — model · layer driven by `/api/health`, not the hardcoded
  `gemma-2-2b-it · L12`. Add a backend-mode badge: `real` / `synthetic · offline` /
  `warming up`.
- **Feature field** — rendering unchanged; now fed real `event.features`, so real
  features illuminate in the map.
- **ProbePanel (WIP)** — when `trackers` is empty, show a placeholder:
  *"Family B — calibrated uncertainty / safety probes (in progress)"* instead of an
  empty list. Keep the "define a probe in natural language" box **visible but clearly
  marked as a preview** (stays on the client-side mock; `synth_concept` is
  unimplemented).
- **Remove auto-run-on-load** — it would fire a 3s generation on every page load. Start
  with an empty thread + composer ready (optionally a couple of suggested prompts).

## 7. Testing

- **Pure / CPU units (no torch):**
  - `build_cognition_event` with empty trackers → `uncertainty is None`, `flag is False`,
    `severity == "info"`, schema-valid.
  - The fallback pipeline produces a schema-valid `CognitionEvent` (deterministic for a
    given prompt).
  - `/api/chat` via FastAPI `TestClient` (fallback mode): yields zero+ `token` lines then
    **exactly one** `event` line, valid NDJSON, last line parses as a `CognitionEvent`.
  - `/api/health` returns the documented shape.
- **Real-model verification:** the existing `smoke_test_gs2.py` remains the GPU/MPS check
  (reconstruction cosine, real labels). Optionally a thin assert that
  `analyze_turn` returns labelled features when run against a loaded engine.
- **Frontend:** manual — run the dev server, send a couple of turns, confirm the thread
  accumulates, the feature field illuminates real features, the probe panel shows the
  WIP placeholder, and the health badge reflects backend mode.

## 8. Dependencies

Add the ML stack to `pyproject.toml` so the real path can run where weights are present:
`torch`, `transformers`, `sae-lens`, `scikit-learn`, `httpx`, `accelerate` (and
`anthropic` is already implied for later Claude work but not required by this task).
All real-path imports remain **lazy** (inside functions / the startup thread) so the app
starts and the fallback path works even when these are not installed.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Model/SAE load is slow or OOMs on MPS | Per-request readiness → fallback keeps UX working; load happens off the request path in a background thread. |
| Neuronpedia labels miss for gemma-3 source | `get_label` already degrades to `"feature N"`; caveat copy already communicates unreliability. |
| Contract change breaks a consumer | schema.py + types.ts + fixture updated in one commit; frontend handles null uncertainty explicitly. |
| Mixed real/fallback within a session is confusing | Health badge surfaces the current mode; mode only transitions loading → real once. |
```
