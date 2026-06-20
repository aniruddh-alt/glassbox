# Lanes & build order

Three people, parallel from hour 1. The unlock: **freeze `schema.py` + `types.ts` + `fixtures/cognition_event.sample.json` together first**, then everyone builds against the fixture.

## Ownership

| Lane | Owner | Files | Owns |
|------|-------|-------|------|
| **A — Backend** | | `app.py`, `engine.py`, `events.py`, `fanout.py`, `labels.py`, `config.py`, `schema.py` | Model load + the layer-12 hook + KV-cache decode, the `cognition_event` builder, FastAPI NDJSON streaming, and the single `fanout()` sponsor seam. |
| **B — Science** | | `science/sae.py`, `science/persona.py`, `science/concept_synth.py`, `validation/*` | Gemma Scope SAE top-k (Family A), persona-vector + calibrated probes (Family B), on-demand user concepts, and the offline AUROC validation vs a logistic-regression baseline. |
| **C — Frontend** | | `frontend/src/*` | The clinician chat view, the live feature cloud (`react-force-graph-2d`), the uncertainty meter (green→red zone), tracked-concept chips, and the Observability tab (Sentry + Phoenix linkouts). |

## Build order (dependency-ordered)

| When | Lane | Deliverable |
|------|------|-------------|
| **H0–1** | shared | **Freeze the contract.** Write `schema.py` + `types.ts` + `fixtures/cognition_event.sample.json` together. Agree the NDJSON line protocol. |
| **H0–2** | all | Scaffold in parallel. A: FastAPI streams the fixture from `/api/chat`. B: load model + SAE on GPU, confirm the layer-12 hook fires + reconstruction error sane. C: render the full UI from the fixture. |
| **H2–5** | B→A | Real signals: persona vectors for the 3 built-ins (diff-of-means + calibrated LogReg), wire `score_all_trackers` + `sae_topk` into `/api/chat`. |
| **H5–8** | A | `fanout()` → Sentry `capture_event` (fingerprinted Issue) + Phoenix OpenInference span. |
| **H8–11** | A+B | Claude layer: offline auto-interp labels for the demo feature set; async `claude_honesty_judge` on flagged events → `adjudication`. |
| **H11–15** | B+C | User-defined concepts: `concept_synth` behind `POST /api/track`; `TrackedConcepts` chips with spinner→live meter. |
| **H15–19** | B | Honest validation (offline): build MedMCQA/PubMedQA confident-wrong labels with a fixed confidence gate; AUROC probe vs baseline + reliability diagram. |
| **H19–22** | C+A | Polish: feature-cloud unverified badges + Neuronpedia embed modal + k-slider; Observability tab; severity tuning. |
| **H22–24** | all | Demo hardening: scripted prompt that reliably trips a flag (meter red → adjudication → click into Sentry/Arize); default to posthoc if live is jittery; pre-record a fallback. |

## Sponsor wiring (all in `fanout.py`)

- **Sentry** — `capture_event(fingerprint=['glassbox', concept, severity], contexts={'cognition': event})` → grouped Issue.
- **Arize Phoenix** — OpenInference LLM span with `cognition.*` attributes + one eval over traces.
- **Anthropic/Claude** — (a) offline auto-interp feature labels; (b) async adjudication of flagged events → `adjudication` field. Build *in* Claude Code.
- **Fetch (stretch)** — a uAgent that POSTs the flagged event to escalate; only if ahead of schedule.
