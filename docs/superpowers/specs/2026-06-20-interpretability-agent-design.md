# Interpretability Agent — Design Spec

**Date:** 2026-06-20
**Project:** GlassBox (UC Berkeley AI Hackathon)
**Status:** Approved — ready for implementation plan

---

## 1. One-line thesis

An autonomous **Interpretability Agent** that takes a natural-language behavior-monitoring
request, replicates the Persona Vectors method end-to-end in the background — synthesize a
contrastive dataset → capture activations → train a calibrated linear probe → confirm the
behavior is linearly represented (held-out AUROC) → register it as a **live guardrail** — and
reports a plain-language verdict. It turns GlassBox's existing "user-defined trackers" feature
into a fully autonomous loop.

Method follows **Persona Vectors** (arXiv 2507.21509). Monitoring (detection) only — steering
is out of scope.

---

## 2. Decisions locked during brainstorming

| Decision | Choice | Why |
|---|---|---|
| Relationship to GlassBox | **Extend** | Reuse `persona.py`, the inference seam, and `fanout` — the guardrail half is already wired; only the synthesis half + the agent are new. |
| Agent platform | **Claude tool-use loop** (`claude-opus-4-8`) | Native fit for runtime orchestration; keeps the Anthropic prize load-bearing; built in Claude Code. Cognition/Devin rejected (a coding agent, not a runtime agent). Fetch deferred to a stretch wrapper. |
| Dataset synthesis | **Claude-direct, persona-vectors templates** | Fewest dependencies; the generation *is* the agent's reasoning. Oumi synth config and the persona_vectors repo verbatim were both considered and rejected for time. |
| Validation / deploy gate | **Autonomous AUROC gate** | Split train/held-out, deploy as a live monitor only if held-out AUROC ≥ τ (default 0.75); otherwise report "not reliably linearly represented" and do not deploy. This is literally "confirm, then monitor." |
| Agent altitude | **Hybrid orchestrator** | Claude does the judgment steps; deterministic Python does the GPU-heavy steps. Reasoning where it matters, deterministic code where it doesn't. |

---

## 3. Architecture & boundaries

A new `backend/agent/` package holds the Interpretability Agent. It sits *in front of* the
existing science layer and reuses everything below it.

```
NL request ("watch for the model being sycophantic toward the clinician")
        │  POST /api/track
        ▼
┌───────────────────────────────────────────────┐
│  Interpretability Agent  (Claude loop)          │   backend/agent/interp_agent.py
│  reasons + calls tools, runs as a bg task       │
└───┬─────────────────────────────────────────────┘
    │ tools (the only things the agent can do):
    │  1. design_spec(request)        → Claude's own structured output
    │  2. generate_contrastive(spec)  → gemma pos/neg responses + acts   [GPU]
    │  3. judge_filter(spec, resp)    → keep clean pos/neg                [Claude]
    │  4. fit_and_validate(acts,y)    → persona_vector + train_probe + held-out AUROC
    │  5. deploy_or_reject(result)    → register tracker if AUROC≥τ, else reject
    ▼
backend/science/{concept_synth.py, persona.py}   ← the work happens here
    ▼
persona._trackers[tid]  ──► score_all_trackers()  ← ALREADY wired into /api/chat
    ▼
every future message scores this concept → cognition_event → fanout (Sentry/Phoenix/UI)
```

**Boundary rules** (consistent with GlassBox's existing 4 interface contracts):

- The `agent/` package **never imports torch directly** — it calls `science.*` functions, exactly
  as Backend does. No FastAPI inside `science/`; no torch inside `agent/` or `fanout/`.
- `concept_synth.py` stops being a stub and becomes the home for the deterministic
  synthesis/fit steps (tools 2 & 4 implementations).
- The agent produces a **Family-B tracker only** (persona-vector + calibrated probe). It never
  touches the SAE dictionary — Family A is a fixed dictionary; new SAE features cannot be
  synthesized.
- The guardrail seam already exists: once `register_tracker` drops a
  `{dir, calibrator, threshold}` record into `persona._trackers`, every subsequent message
  scores it for free via `score_all_trackers`.

---

## 4. The agent loop & its five tools

Standard Claude tool-use cycle, seeded with a system prompt casting it as an interpretability
researcher replicating Persona Vectors. Runs to completion autonomously, emitting a status
update after each tool returns.

**Tool 1 — `design_spec(request, model_name)`** → the agent's own structured output (how the
loop turns NL into a plan), following the persona-vectors `generate_trait` template:
- `trait_name` + crisp definition (what counts, what doesn't)
- a **contrastive system-prompt pair**: `pos` (elicit the trait) / `neg` (suppress / neutral)
- `~40 questions` spanning the medical-chat domain where the trait could surface
- a `judge_rubric` (1–5) for scoring how strongly a response expresses the trait

**Tool 2 — `generate_contrastive(spec)`** [GPU, deterministic] → for each question, run
`gemma-3-4b-it` twice (under `pos` and `neg` system prompts) via `engine.generate_and_capture`,
capturing layer-17 `act_resp` (mean-pooled response tokens) and `act_last` (last prompt token).
Returns ~80 `(response_text, act_resp, act_last, intended_label)` rows.

**Tool 3 — `judge_filter(spec, rows)`** [Claude judge] → score each response with the rubric;
**drop rows where the behavior didn't land** (a `pos`-prompted response that didn't express the
trait; a `neg` that leaked it). Labels come from *observed behavior*, not from which prompt was
used. Returns the clean labeled set.

**Tool 4 — `fit_and_validate(rows)`** [deterministic] → stratified train/held-out split;
`persona_vector(pos, neg)` for the steering direction; `train_probe(X, y)` for the calibrated
detector; compute **held-out AUROC**, a plain-LogReg-on-raw-acts baseline, and reliability
stats. Returns `{auroc, baseline_auroc, threshold, n_kept, calibrated}`. Returns
`insufficient_data` if the judge filtered out too much or a class collapsed.

**Tool 5 — `deploy_or_reject(spec, fit)`** → if `held_out_auroc ≥ τ` (default 0.75): register
`{dir, calibrator, threshold, meta:{user_defined:true, auroc, request, reliability:"synthetic-validated"}}`
into `persona._trackers[tid]`, status → `ready`. Else status → `rejected` with the reason.
Either way the agent writes a short natural-language **verdict**, e.g.:
- *"Sycophancy is linearly represented at layer 17, held-out AUROC 0.89 — now monitoring."*
- *"Couldn't confirm a reliable linear signal, AUROC 0.62 — not deploying a monitor I don't trust."*

---

## 5. Async execution & state model

`POST /api/track {request: "..."}` returns a `tracker_id` **immediately** (matches the existing
`synth_concept` stub contract) and launches the loop in a background task (`asyncio.create_task`,
same pattern as the async Claude honesty-judge).

The tracker record carries a `status` that walks through:

```
pending → designing → generating → judging → fitting → (ready | rejected | error)
```

plus a `progress` blob (current step, %, the verdict and headline AUROC when done).

`GET /api/track/{tracker_id}` returns that record for the UI to poll. The `TrackedConcepts` chip
shows: spinner (in progress) → live uncertainty meter (`ready`) → greyed "not linearly
represented" state (`rejected`) → retry affordance (`error`).

State is in-memory in `persona._trackers` (single-box demo — no DB). Persistence across restarts
is **out of scope**.

A typical concept = ~80 gemma generations + 2 Claude calls ≈ a couple of minutes on the local GPU.

---

## 6. Validation honesty

Protects the science pitch against the inevitable "AUROC on *what*?" question.

- Held-out AUROC is measured on a **held-out split of the synthetic contrastive set**. A high
  value confirms the trait is **linearly separable in layer-17 activations under contrastive
  elicitation** — exactly the Persona Vectors claim, and a real, defensible result.
- It is **necessary but not sufficient** for real-world reliability: the synthetic set may not
  cover the natural distribution of clinician chats. Deployed user-concepts are therefore tagged
  `reliability: "synthetic-validated"`, distinct from the three built-ins (`uncertainty`,
  `harmful`, `hallucination`) which get the heavier MedMCQA/PubMedQA real-data validation. The
  UI surfaces this distinction.
- The plain-LogReg baseline (same split) is reported alongside, so "the probe beats the trivial
  baseline" is shown, not asserted.

---

## 7. Error handling & failure modes

| Failure | Handling |
|---|---|
| Generation/judge error mid-loop | Background task catches per-tool → `status: error` with failing step + message; chip shows retry. Never wedges silently. |
| Degenerate synthetic set (class collapse / over-filtered) | `fit_and_validate` returns `insufficient_data` → agent rejects with that reason rather than training a garbage probe. |
| Claude API hiccup | Bounded retries (reuse the existing fanout-judge retry pattern); on exhaustion → `status: error`. |
| Gate tuning | `τ` lives in `config.py`, not hard-coded — tunable during demo dry-run. |
| Concurrent submissions | Two background tasks run; the GPU serializes generation. Acceptable on one box. |

---

## 8. API & test surface

**New:**
- `POST /api/track` → `{tracker_id, status}`
- `GET /api/track/{id}` → tracker record (status, progress, verdict, auroc)
- Package `backend/agent/`: `interp_agent.py` (the loop), `tools.py` (tool impls/dispatch),
  `prompts.py` (system + generate_trait + judge templates)
- Fill in `backend/science/concept_synth.py` (deterministic synthesis + fit steps)

**Unchanged contracts:** `CognitionEvent` shape, `/api/chat` NDJSON protocol,
`score_all_trackers`, `fanout.py`. The agent rides existing seams.

**Tests:**
- `backend/smoke_test_agent.py` — full loop end-to-end on one cheap concept against real gemma
  (mirrors the existing smoke-test style).
- A unit test for the AUROC gate (deploy above τ / reject below τ) using synthetic activations,
  so it runs without the GPU.

---

## 9. Out of scope (YAGNI)

- Steering / activation editing (detection only).
- Synthesizing new SAE features (Family A is a fixed dictionary).
- Persistence of trackers across server restarts.
- A Fetch.ai uAgent wrapper (stretch only, post-core, if ahead of schedule).
- Oumi synthesis config and persona_vectors-repo-verbatim generation (considered, deferred).
- Multi-layer probing (single layer-17 hook, per the locked architecture).
