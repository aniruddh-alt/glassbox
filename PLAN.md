# GlassBox — Build Plan

**Live "fMRI for a medical LLM."** Watch SAE features fire token-by-token as an open model answers medical questions, surface *confident-wrongness* from the model's own internal uncertainty, and keep the whole patient-data path on-prem. UC Berkeley AI Hackathon · 24h · 3 people.

---

## 0. One-line thesis
Teams deploying medical LLMs are flying blind: a confident answer and a hallucination look identical from the outside. GlassBox instruments the model's **internal** state — surfacing uncertainty/hallucination features in real time — so the deploying team gets an alert *before* a confident-wrong answer reaches a patient. **Surface, never suppress.**

---

## 1. Current state (validated ✅)
- Repo scaffolded; `.venv` (uv, py3.12, torch 2.12, transformers 5.12, sae-lens 6, scikit-learn).
- **`config.py`** — locked to Gemma Scope 2 + Gemma-3-4b (see §2).
- **`engine.py`** — model load (multimodal-aware loader + nested-layer auto-detect + tensor/tuple hook), `generate_and_capture()`. ✅
- **`science/sae.py`** — `load_sae` / `sae_topk` / `reconstruction_error`. ✅
- **`science/persona.py`** — `persona_vector` (diff-of-means) / `train_probe` (calibrated LogReg) / `project` / `score_all_trackers`. ✅
- **`science/feature_provider.py`** — Local (primary) + Neuronpedia (fallback) + `get_provider`. ✅
- **`labels.py`** — Neuronpedia keyless label cache. ✅
- **`smoke_test.py`** — Family-A end-to-end: **reconstruction cosine 0.999**, correct medical answer, labels resolve. ✅
- **`smoke_test_probe.py`** — Family-B layer sweep: diff-of-means **cleanly separates** hedging vs confident (mechanism proven; toy set saturated AUROC=1.0 → real validation still owed).
- **`bench_latency.py`** — interp overhead **<100 ms**; generation ~3 s (MPS) dominates; cold label fetch 503 ms → pre-warm.

**Translation:** the hard, risky parts (SAE on Gemma-3-4b, layer wiring, live capture, probe mechanism, latency) are *proven*. What remains is product polish, validation, integration, and sponsor wiring.

---

## 2. Locked architecture

**Stack:** `unsloth/gemma-3-4b-it` (ungated; gated `google/` mirror as alt) · Gemma Scope 2 `gemma-scope-2-4b-it-res` `layer_17_width_16k_l0_medium` (d_in 2560) · Neuronpedia labels (~100% on residual SAEs) · single GPU (local for demo) · FastAPI POST + `fetch()/ReadableStream` NDJSON (never EventSource).

**Two families, one layer-17 hook:**
- **A — SAE feature cloud** (exploratory): top-k features/token → labels (caveated). The "wow" viz.
- **B — persona-vector probes** (reliable): diff-of-means + calibrated LogReg for `uncertainty` / `harmful` / `hallucination` + user-defined trackers. The safety signal.

**One `cognition_event` per message** → `fanout()` (CPU, no torch) → Sentry · Arize/Phoenix · UI · async Claude judge.

### 🔒 Trust boundary (the privacy architecture — say this out loud to judges)
> **PHI never leaves the box.** The local model does *all* patient-data work — generation, SAE features, probes, detection. **Claude only sees the de-identified interpretability layer** — feature labels (from a generic corpus, offline), scores, and aggregate cognition patterns — **never a raw patient query.**

| Crosses to Claude (de-identified) | Stays local (PHI) |
|---|---|
| SAE feature labeling (generic corpus, offline) | Patient question + model response |
| Concept-synth (synthetic contrastive pairs) | Raw residual activations |
| Cognition agent input = **telemetry only** (feature IDs, scores, counts) | Probe internals |
| "built in Claude Code" | |

Demo uses **public** MedMCQA/PubMedQA (no PHI) so the honesty-judge on raw text is fine *for the demo*; production keeps the judge local or de-identifies. State this as a deliberate design decision.

---

## 3. Prize strategy (one build → stacked prizes)
- **Ddoski's Lab ($5k)** — submit here (technical depth + real-world application).
- **Sentry (4× Switch 2 + interview)** — real SDK: `capture_event` with fingerprint → grouped Issue on confident-wrong; alt-tab to live dashboard = money shot. Half the rubric is **team dynamics** → 3-speaker pitch + a "what broke / how we course-corrected" beat.
- **Arize ($1k)** — self-hosted Phoenix sidecar (CPU); OpenInference span per message + 1 eval.
- **Anthropic** — Claude is load-bearing at 3 points (labeler, honesty-judge, cognition agent **on telemetry**), built in Claude Code. Use `claude-opus-4-8`.
- **Devin** — free: point at scaffolding (FastAPI, dashboard, tests, deploy), capture PR provenance. Keep AWAY from the SAE/probe core.
- **Fetch** — stretch only. **Annapurna — skip** (Trainium fuses the graph → forward hooks never fire → breaks the core).

---

## 4. Lanes
- **A — Backend** (`engine`, `events`, `fanout`, FastAPI streaming, Sentry + Arize wiring). Owns the cognition_event + the seam.
- **B — Science** (probe validation, MedMCQA/PubMedQA confident-wrong set, hero features, honesty assets, Claude labeler/judge/cognition-agent). Leads the scientific pitch.
- **C — Frontend** (chat, feature cloud, uncertainty meter, "unverified" badges, Observability tab, user-defined-tracker UI). Owns the visual wow + legibility (<30s to grasp).

**Hour-1 contract freeze:** all three write `schema.py` (pydantic `CognitionEvent`) + `types.ts` + `fixtures/cognition_event.sample.json` together, then build against the fixture.

---

## 5. 24-hour timeline
> Core/scaffold/validation already proven (§1). If rebuilding live during the event, the prototype is the reference. Blocks are owner-tagged.

| Hours | Owner | Deliverable |
|---|---|---|
| **0–2** | shared | Box up + `whoami`; freeze `schema.py`/`types.ts`/fixture; confirm smoke_test green on the event box; **pick layer (17 vs 22) via the MedMCQA validation, not the toy sweep** |
| **0–2** | C | FastAPI POST-NDJSON skeleton streaming the fixture; UI shell (chat / cloud / meter areas) |
| **2–6** | A | `POST /analyze` (post-hoc) + `POST /chat` (NDJSON) → real `cognition_event`; mask special tokens |
| **2–8** | B | **Real validation**: build MedMCQA/PubMedQA confident-wrong set (fixed confidence gate) → train probe at 9/17/22 → **AUROC vs LogReg baseline**; lock the layer |
| **2–8** | C | Live feature cloud (react-force-graph) + token-firing default view + uncertainty meter (green→red) |
| **6–10** | B | **Cloud ranking filter** ⭐ (down-weight high-norm grammatical features so medical ones surface) + try `l0_small`; pre-warm label cache for demo questions |
| **8–12** | A | `fanout()`: Sentry `capture_event`(fingerprint+context) + Phoenix span + eval |
| **8–12** | B/S | Claude layer (on telemetry only): offline feature **labeler** (Claude vs Gemini diff) + async **honesty-judge** on flagged events |
| **11–15** | B/C | User-defined trackers: `concept_synth` (Oumi/Claude → contrastive pairs → on-demand persona vector) behind `/track`; chip UI |
| **12–16** | B/S | **Cognition agent** (Claude tool-use loop over the de-identified telemetry → incident digest) — the agentic finale |
| **12–16** | all | Freeze ~15 demo questions (confident-right / confident-wrong / PubMedQA "maybe"); pre-record one perfect run; lock honesty panel |
| **16–20** | all | Polish cloud + meter; cut latency; full dry-run of the demo script; fix top-3 jank |
| **20–23** | all | Rehearse 3-speaker pitch (Lab + Sentry + Anthropic framings + trust-boundary slide); push repo; submit minimal Devpost by ~9:30 AM |
| **23–24** | all | Two clean dry runs (live + recorded fallback); finalize submission |

---

## 6. Demo script (5 min, live, at the table)
1. **Name the elephant** — "SAEs had a rough 18 months; we built the tool for the one thing the field agrees they're good at: surfacing, not deciding."
2. Type a medical question → watch the **feature cloud** animate token-by-token (labels + "unverified" badges).
3. **Money shot** — on a confident-*wrong* question the **uncertainty meter spikes red** while the model states a wrong answer confidently. Its insides knew; its words didn't.
4. Flip to a question it gets right → meter green.
5. **Honesty beat** — click a hero feature; token-firing shows a *wrong* label (coffee-on-coffin) + two disagreeing auto-labels. "We don't trust labels — here's how we check." Show the LogReg baseline side-by-side.
6. **Sentry money shot** — alt-tab to the live Sentry dashboard; the hallucination-risk Issue fired in real time with the cognition context attached.
7. **Cognition agent finale** — "Analyze session" → Claude reads the **de-identified telemetry** and writes an incident digest.
8. **Trust-boundary close** — "PHI never left the box; Claude only ever saw de-identified telemetry. Surface, never suppress."

---

## 7. Risk register / de-risk order
| Risk | Sev | Mitigation |
|---|---|---|
| Cloud looks grammatical, not medical | **high** | the **ranking filter** (§5, 6–10) — the #1 gap to a demo-quality cloud |
| Probe doesn't separate confident-wrong cleanly | **high** | real MedMCQA validation early (2–8); fall back to feature-rivalry score or hand-pick clearest examples; always show baseline |
| Privacy contradiction (PHI → cloud) | **high** | trust boundary (§2): Claude sees telemetry only; demo uses public data |
| Demo network (SSE through proxy) | med | run **local**; POST + fetch/ReadableStream |
| Model gating at hour 1 | med | unsloth ungated mirror; `whoami` first |
| Cold label fetch on stage | med | pre-warm cache offline |
| Scope creep (Fetch/Annapurna/multi-layer) | med | one layer, drop Fetch unless ahead, skip Annapurna |

---

## 8. Open decisions
- **Layer 17 vs 22** — default 17 (proven); validate on MedMCQA, switch to 22 if it wins for the probe (literature leans late-middle).
- **Honesty-judge in demo** — public-data path (simplest) vs local-only (purest). Demo = public; note local for prod.
- **Fetch uAgent** — only if ahead at hour 18.
- **`l0_small` vs `l0_medium`** — A/B for cloud cleanliness during the ranking-filter work.
