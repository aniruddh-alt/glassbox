# Open-Source Readiness — Design

**Date:** 2026-06-22
**Status:** Draft for review
**Goal:** Turn GlassBox from a hackathon prototype into a general-purpose, open-source LLM
interpretability-observability tool that others can run on their own model + SAE, with their
own keys, on their own GPU.

---

## 1. Summary

GlassBox instruments an open model's internal state per chat turn: a SAE "feature cloud" of
what concepts are firing, plus calibrated probes that flag when the model is internally
uncertain but verbally confident. Today it is wired specifically to `unsloth/gemma-3-4b-it` +
Gemma Scope, framed around a medical use case, and depends on a maintainer-owned GPU pod and
Anthropic key.

This milestone makes it a clean OSS project along five workstreams. The strategy, per the
maintainer's calls, is **ship a focused, well-architected v1 and file open-source issues for
the ambitious generalizations** rather than building everything now.

### Locked decisions

| Decision | Call |
|---|---|
| Positioning | **Generalize away from medical.** Reposition as a general interpretability/observability tool; keep medical as ONE clearly-labeled example. |
| Project name | **Keep "GlassBox"** (glass box = anti–black box; reads general). Confirm at review. |
| Model/SAE scope (v1) | **Gemma + Gemma Scope only**, but built behind a clean config seam. Arbitrary HF model+SAE → OSS issue. |
| Agent provider | **Claude-only** (leans on Claude's agentic tool-use). Multi-provider → OSS issue. BYO Anthropic *key*. |
| GPU service | **Self-host Docker first.** Optional managed endpoint → OSS issue. Never "we SSH into your box." |
| Architecture | **One parent `AppConfig`**, single source of truth, loaded once, passed down — replaces flat `config.py` globals. |
| Observability | **Drop Arize/Phoenix entirely; keep Sentry.** Rename `cognition_event` → `introspection_event`. |

### Out of scope for v1 (tracked as issues, see §9)
Arbitrary HF SAE repos / non-Gemma models; multi-provider agent; managed/hosted GPU endpoint;
serializing the probe calibrator across restart.

---

## 2. The backbone: unified `AppConfig`

The central change. Today `backend/config.py` is ~40 flat module-level globals
(`config.MODEL_ID`, `config.LAYER`, `config.SAE_RELEASE`, …) imported across both processes.
This scatters the model/SAE coupling everywhere and makes a swap a code edit.

Replace it with one validated config object loaded once and threaded through the backend.

```
AppConfig                       # single source of truth (pydantic-settings)
├── model: ModelConfig          # model_id, layer(s), hook point, dtype, system_prompt,
│                               #   special/mask tokens, preamble_skip  ← Gemma specifics live here
├── sae: SAEConfig              # release, sae_id, d_in, d_sae, label_source  ← the swap seam
├── probes: ProbeConfig         # enabled[], artifacts_dir, auroc_threshold,
│                               #   builder { agent_model, judge_model }   ← probe builder writes here
├── observability: ObsConfig    # sentry { send_io, org_slug, project_slug, api_base, ... }
├── pod: PodConfig              # url, token, timeout   ← GPU service connection
└── runtime: RuntimeConfig      # mode (real|fallback|live), product_name, ...
```

**Loading rules:**
- **Structure** (model id, layer, sae release, enabled probes, branding) loads from **`config.yaml`**.
  A committed **`config.example.yaml`** is the template; the user's working **`config.yaml`** is
  git-ignored. With no `config.yaml`, the app falls back to the Gemma defaults so it still boots.
- **Secrets** (`ANTHROPIC_API_KEY`, `SENTRY_DSN`, `POD_TOKEN`, `HF_TOKEN`) load **only from env /
  `.env`** — never the YAML. Preserves the no-leaks property.
- Env overrides YAML for any field (12-factor friendly).

**Threading:** built once at startup, stored in FastAPI `app.state` on the orchestration side
(`backend/app.py`) and on the pod (`backend/gpu_service.py`), and passed into the science/engine
entry points (`engine`, `sae`, `persona`, `concept_synth`, `events`, `fanout`) rather than
imported as globals. Pure-CPU modules (`events`, `fanout`) take the relevant sub-config; torch
modules take `model`/`sae`/`probes`.

**Why this is the right backbone:** it absorbs nearly every coupling problem found in the audit
(hardcoded `SAE_RELEASE`, scattered Gemma constants, dual config sources), **and** it *is* the
generalization seam — the future "any model/SAE" issue becomes "populate `model`/`sae` with
non-Gemma values," not a rewrite.

**Effort/risk:** mechanical but wide (touches most backend files). Test-guarded — the existing
30 test files exercise the import surface, so breakage surfaces fast. Do this first; everything
else sits on it.

---

## 3. Workstreams

### WS0 — Config backbone + hygiene + rebrand foundation  *(do first; unblocks the rest)*

**Config:** implement §2. Migrate `config.py` consumers to `AppConfig`. Ship `config.example.yaml`.

**Hygiene (from audit):**
- `git rm` tracked cruft: `err.txt` (empty), `main.py` (hello-world stub), `batch_medqa_results.json` (demo output).
- Gitignore generated artifacts (one `git add -A` from shipping): `backend/science/probe_jobs/`,
  `backend/science/artifacts/watch-*.json`, `.pytest_cache/`. Keep the 6 built-in probe artifacts
  (`harmful`, `harmful_prompt`, `over_confidence`, `hallucination`, `uncertainty`, `risk_awareness`).
- Delete the on-disk `.superpowers/` scaffolding before publishing.
- Reconcile `backend/requirements.txt` vs `pyproject.toml` → single source of truth (`pyproject.toml`);
  fix the stale "torch 2.12" comment.
- Pick one frontend package manager: remove either `frontend/bun.lock` or `frontend/package-lock.json`;
  align README.
- `LICENSE`: set a real copyright holder.

**Rebrand foundation:** lift the product name and the demo "domain" into config
(`runtime.product_name`, neutral default `system_prompt`). Keep "GlassBox" as the name; strip the
"for medical LLMs" positioning from package metadata, README, and UI copy. Medical becomes an
opt-in example (a `config.example.yaml` "medical" profile + the existing fixtures, clearly labeled).

### WS1 — Observability: drop Arize, keep + rename Sentry  *(small, self-contained)*

**Remove Arize/Phoenix (complete list from audit):**
- Delete whole files: `backend/coherence_eval.py`, `backend/phoenix_eval_features.py`,
  `scripts/run_phoenix.sh`, `backend/tests/test_coherence_eval.py`,
  `backend/tests/test_phoenix_eval_features.py`.
- `backend/fanout.py`: remove `PhoenixSink`, `_phoenix_is_local()`, `_stage_spans`/`_STAGE_KIND`,
  the OpenInference env block, the OTel import, and the Phoenix branch of `init_sponsors()`.
- `backend/app.py`: remove `/api/observability/eval` and `snap["phoenix_ui_url"]`.
- `backend/observability.py`: remove the `feature_incoherence`/`phoenix-eval` taxonomy entry.
- Frontend: remove `PhoenixPanel` + coherence-eval button/state in `ObservabilityPage.tsx`,
  `runEval()` in `api.ts`, `phoenix_ui_url` in `types.ts` and `mock.ts`.
- Deps: drop `arize-phoenix`, `openinference-instrumentation` from `requirements.txt`.
- Env/docs: remove `PHOENIX_*`; purge Phoenix mentions from README/PLAN/LANES.
- Update `backend/tests/test_fanout.py` and `test_observability_endpoint.py` (drop Phoenix assertions).

**Keep Sentry**, genericize it:
- Rename event `cognition_event` → `introspection_event` (see §4).
- Genericize medical strings: `fanout.py` fingerprint `"medical-cognition"` → neutral
  (`"glassbox","introspection",reason`), message `"Confident-wrong medical answer"` → neutral,
  `app.py` hint text.
- Fix the `.env.example` `SENTRY_ORG`/`SENTRY_PROJECT` vs code `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG`
  mismatch. Sentry stays fully env-gated (no key → no-op), unchanged.

### WS2 — BYO model + SAE (Gemma scope, config-shaped)  *(the headline feature, v1 slice)*

- Move the hardcoded `SAE_RELEASE` (`config.py:20`) and the `SAE_ID` naming pattern into `sae` config.
- Derive `d_in`/`d_sae` from the loaded model/SAE at runtime, with config values as fallback (today
  `D_IN=2560`/`D_SAE=16384` are never re-derived).
- Keep the auto decoder-layer detection (`engine._find_decoder_layers`) — already model-agnostic.
- v1 ships Gemma defaults and is *documented* as Gemma-only-validated. The Gemma-specific knobs
  (`MASK_TOKENS`, `PREAMBLE_SKIP`, system-role-merge) live in `model` config with Gemma defaults.
- The real generalization (arbitrary HF SAE loader, Neuronpedia-label fallback, tokenizer-derived
  special tokens, preamble auto-detect) is **OSS issue #1** — the config seam makes it additive.

### WS3 — Probe pipeline OSS (BYO key + GPU self-host)

**Agent / BYO Anthropic key:** the key already lives on the pod via env; for self-host that's the
clean model (user sets `ANTHROPIC_API_KEY` on their own pod). The client-injection points already
exist (`interp_agent.run_interp_agent(client=...)`, `concept_synth.judge_filter(client=...)`) —
keep them so a future per-request/provider path is easy. Surface `anthropic_configured` in
`/health` (already present). Genericize the agent's medical prompts (`prompts.py:9-23,39`
"medical-chat LLM") to a configurable domain string from `model.system_prompt`/a domain field.

**GPU self-host:**
- Ship a `Dockerfile` + `compose` for `backend/gpu_service:app` (torch + transformers + model/SAE
  download) so users run it on their own GPU instead of the manual `nohup`/Tailscale dance.
- **Harden auth to fail closed:** `gpu_service._require_auth` currently returns (allows) when the
  token is empty (`gpu_service.py:30-36`). Make `POD_TOKEN` mandatory for any non-loopback bind;
  reject when unset.
- Remove the committed `glassbox-dev-secret` default from `scripts/run_with_pod.sh:8`,
  `pod_tailscale_bootstrap.sh:32`, and `batch_medqa_observability.py:12`; document generating a
  strong token.
- Demote Tailscale+SSH to an optional "behind a restrictive firewall" appendix (it was only an
  eduroam workaround).
- Managed/you-host-it endpoint → **OSS issue #3**.

### WS4 — OSS infrastructure  *(docs, CI, community files)*

- Rewrite `README.md` for the general framing + self-host quickstart; keep the strong architecture
  section, drop prize/hackathon framing.
- Add `ARCHITECTURE.md` (the two-process split, the config, the event contract, the science
  boundary), `CONTRIBUTING.md`, `SECURITY.md`, a real **self-host / deployment guide** (distinct
  from the Tailscale runbook), `CHANGELOG.md`.
- Remove/relocate hackathon-internal docs: `PLAN.md`, `LANES.md`, `docs/superpowers/*`.
- Add `.github/workflows/` CI: lint + the non-GPU pytest subset (the GPU-dependent `engine`/`sae`
  tests stay opt-in).
- File the OSS issues (§9).

---

## 4. The `cognition_event` → `introspection_event` rename

The payload carries probe scores + SAE features + uncertainty — model *introspection*, nothing
medical. Rename scope (mechanical, do as part of WS1):
- Backend: `CognitionEvent` class, `build_cognition_event`, `capture_cognition_alarm`, docstrings,
  Sentry fingerprint/attribute keys.
- Frontend: `CognitionEvent` interface, `useCognitionStream` hook, `mock.ts`, comments.
- Fixture: rename `fixtures/cognition_event.sample.json` → `introspection_event.sample.json`;
  update references (`schema.py`, `smoke_fanout.py`, README).
- Keep `schema_version` and bump if the shape changes.

---

## 5. Architecture (unchanged seams, generalized contents)

The two-process split stays: a CPU **orchestration** backend (`app.py`, never imports torch) and
a GPU **pod service** (`gpu_service.py`, owns torch/model/SAE/probes), talking over HTTP. The four
interface contracts stay intact — `AppConfig` becomes a fifth, explicit contract. The NDJSON wire
protocol, the torch-only science boundary, and the `fanout` sink registry are unchanged in shape;
only their *contents* generalize (Arize sink removed, medical strings neutralized, config injected).

---

## 6. Data flow (unchanged)

`UI → POST /api/chat → [pod] generate + layer hook → SAE top-k + probe scores →
[CPU] build ONE introspection_event → fanout() → {Sentry (if flagged), in-process store → /api/observability, chat UI}`.
Claude honesty-judge remains an async post-emit task; the Phoenix sink is gone.

---

## 7. Error handling & safety (unchanged posture, hardened edges)

- Fallback mode (synthetic features when pod is down/no GPU) stays — it's what makes the UI run
  with zero setup. Genericize its canned medical labels to neutral defaults.
- PII scrubbing in the Sentry path stays (`send_default_pii=False`, `_scrub_pii`).
- New: pod auth fails closed (WS3); secrets never enter `config.yaml` (§2).

---

## 8. Testing

- Keep the 30-file backend suite green through every workstream; it's the safety net for the
  config refactor and the Phoenix removal.
- Update/remove Phoenix tests (WS1). Add tests for `AppConfig` load/override/secret-exclusion.
- Document that `engine.py`/`science/sae.py` are GPU-gated (untested without weights) — CI runs the
  non-GPU subset.

---

## 9. Open-source issues to file (WS4)

1. **Generalize to arbitrary HF model + SAE** — custom SAE loader (read weights/cfg from any HF
   repo, not just SAELens releases), label fallback when no Neuronpedia coverage, tokenizer-derived
   special tokens, preamble auto-detect, layer/d_in cross-checks.
2. **Multi-provider agent** — abstraction layer (e.g. LiteLLM) over the Claude-specific
   `thinking`/structured-output/tool-use shapes so OpenAI/Gemini/local models drive the probe builder.
3. **Optional managed / you-host-it GPU endpoint** — a hosted `gpu_service` users point `POD_URL` at
   (never SSH-into-their-box), with real multi-tenant auth/TLS/rate-limiting.
4. **Serialize the probe calibrator** — today only the scalar-projection form survives a pod restart
   (`docs/probe-training.md:155`); persist the full calibrator.

---

## 10. Sequencing & dependencies

```
WS0 (config + hygiene + rebrand foundation)  ──┬──▶ WS1 (observability)
                                               ├──▶ WS2 (BYO model/SAE)
                                               └──▶ WS3 (probe pipeline / GPU)
WS1, WS2, WS3 ──▶ WS4 (docs, CI, issues)   [WS4 docs depend on the others landing]
```
WS0 is the gate. WS1/WS2/WS3 are largely parallelizable once the config exists. WS4 closes it out.

---

## 11. Confirm at review

- **Keep the name "GlassBox"** (repositioned as general), or rename the project entirely?
- **Event name** `introspection_event` — good, or prefer another (`cognition_trace`, `model_state_event`)?
- **Neutral default `system_prompt`** content — generic assistant, or a specific non-medical example?
- Anything in §9 you want pulled *into* v1 rather than deferred to an issue?
