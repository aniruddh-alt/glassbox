# WS4 — OSS Infrastructure (docs, CI, issues) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

Goal: Ship the open-source-readiness infrastructure for GlassBox — a general-framing README + new top-level docs (ARCHITECTURE/CONTRIBUTING/SECURITY/CHANGELOG), a real self-host deployment guide, GitHub Actions CI that runs lint + the non-GPU pytest subset, removal of hackathon-internal docs, and four ready-to-post OSS issues.

Architecture: WS4 is the closing workstream — it touches only documentation, CI config, and `pyproject.toml` metadata; it never changes runtime Python behavior. The two-process split (CPU orchestration `backend/app.py` + GPU pod `backend/gpu_service.py`) and the `AppConfig` contract are described in the docs but not modified here. CI exercises the import surface that the WS0 config refactor and WS1 Phoenix removal rely on, so it runs the non-GPU test subset (the torch-importing `engine`/`sae` tests stay opt-in).

Tech Stack: GitHub Actions (`actions/checkout@v4`, `astral-sh/setup-uv@v3`), `uv` (sync + run), `ruff` 0.13 (lint), `pytest` 8 (non-GPU subset selected by `--ignore`), Markdown docs, MIT license.

## Global Constraints

- Python >= 3.12 (matches .python-version).
- Base deps (pyproject.toml): fastapi>=0.138.0, sentry-sdk[fastapi]>=2.63.0, httpx>=0.27.0.
- ML extra (uv sync --extra ml): torch>=2.4, transformers>=4.50, sae-lens>=6.0, scikit-learn>=1.5, accelerate>=0.34, anthropic>=0.40.
- pyproject.toml is the canonical dependency source; backend/requirements.txt is being retired.
- License: MIT.
- Naming/copy: KEEP the product name "GlassBox"; REMOVE medical-specific positioning (reposition as a general-purpose LLM interpretability/observability tool); medical survives only as ONE clearly-labeled optional example. Rename the per-turn event from cognition_event to introspection_event (class CognitionEvent -> IntrospectionEvent).
- Secrets (ANTHROPIC_API_KEY, SENTRY_DSN, POD_TOKEN, HF_TOKEN, plus the optional fifth SENTRY_AUTH_TOKEN for the Observe REST integration) load ONLY from env / .env, NEVER from config.yaml. This is the canonical five-secret list per the AppConfig contract; every doc WS4 ships (README Configuration, ARCHITECTURE, CONTRIBUTING, SECURITY) must list the same five (SENTRY_AUTH_TOKEN may be footnoted as the optional fifth, but not omitted).
- COMMIT RULE (CRITICAL): commit messages MUST NOT add Claude as a co-author. No "Co-Authored-By: Claude" trailer anywhere. Use Conventional Commits style.
- Tests: keep the existing backend pytest suite green; CI runs the non-GPU subset; backend/engine.py and backend/science/sae.py are GPU-gated (untested without weights).
- Two processes: orchestration backend (backend/app.py, NEVER imports torch) and GPU pod service (backend/gpu_service.py, owns torch). AppConfig is built once per process and threaded via FastAPI app.state.

---

## File Structure

Files this workstream creates or modifies, each with its single responsibility:

- `README.md` — **Modify (full rewrite).** General-purpose framing, self-host quickstart, architecture summary; no hackathon/prize/medical/Phoenix copy.
- `ARCHITECTURE.md` — **Create.** The two-process split, the `AppConfig` contract, the `introspection_event` contract, the science boundary.
- `CONTRIBUTING.md` — **Create.** Dev setup, lint/test commands, the GPU-gated-tests note, commit/PR conventions.
- `SECURITY.md` — **Create.** Supported versions, private vulnerability reporting, the secrets-never-in-YAML + PHI-gate posture.
- `CHANGELOG.md` — **Create.** Keep-a-Changelog format; `0.1.0` documenting the OSS-readiness release.
- `docs/deployment.md` — **Create.** Real self-host & deployment guide (env, config, Docker GPU pod, reverse proxy) — distinct from the Tailscale runbook.
- `docs/tailscale-pod-runbook.md` — **Modify (header only).** Demote to an optional "behind a restrictive firewall" appendix pointing at `docs/deployment.md`.
- `docs/issues/01-arbitrary-hf-model-sae.md` — **Create.** OSS issue #1 text (§9.1).
- `docs/issues/02-multi-provider-agent.md` — **Create.** OSS issue #2 text (§9.2).
- `docs/issues/03-managed-gpu-endpoint.md` — **Create.** OSS issue #3 text (§9.3).
- `docs/issues/04-serialize-probe-calibrator.md` — **Create.** OSS issue #4 text (§9.4).
- `.github/workflows/ci.yml` — **Create.** CI: ruff lint + non-GPU pytest subset on push/PR.
- `.github/PULL_REQUEST_TEMPLATE.md` — **Create.** PR checklist (lint/tests/docs/no-secrets).
- `.github/ISSUE_TEMPLATE/bug_report.md` — **Create.** Bug report issue template.
- `.github/ISSUE_TEMPLATE/feature_request.md` — **Create.** Feature request issue template.
- `pyproject.toml` — **Modify.** Add `ruff` to the dev group, add `[tool.ruff]` config, register the `gpu` pytest marker, and update the package `description` to the general (non-medical) framing. **SHARED FILE** (WS0 adds `pydantic-settings`; WS1 may drop arize deps). Coordinate: WS4 only touches the `description`, `[dependency-groups].dev`, `[tool.ruff]`, and `[tool.pytest.ini_options].markers` keys.
- `.env.example` — **Read-only here; SHARED FILE owned by WS0/WS1/WS3 — DO NOT rewrite in WS4.** README (Task 2), CONTRIBUTING (Task 4), and deployment.md (Task 8) all instruct `cp .env.example .env` and then describe a clean 4-/5-secret world. The real `.env.example` is currently stale: it still has `PHOENIX_COLLECTOR_ENDPOINT`, the wrong-name `SENTRY_ORG`/`SENTRY_PROJECT` (code/contract use `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG`), the flat `GLASSBOX_MODE` (vs the `GLASSBOX__RUNTIME__MODE` convention the docs tout), `MODEL_ID`, and the committed default `POD_TOKEN=glassbox-dev-secret`. Sequencing: **WS1** fixes `SENTRY_ORG` → `SENTRY_ORG_SLUG` and removes `PHOENIX_*`; **WS3** removes the `glassbox-dev-secret` default; WS0/WS1 align it to the `AppConfig` contract. WS4's docs ASSUME the cleaned `.env.example`. The doc tasks below VERIFY it is clean (no `PHOENIX_`, no `glassbox-dev-secret`) before committing — see the shared `.env.example` precondition added to Tasks 2/4/8.
- `PLAN.md` — **Delete (`git rm`).** Hackathon-internal build plan.
- `LANES.md` — **Delete (`git rm`).** Hackathon-internal lane ownership.
- `docs/superpowers/` — **Delete (`git rm -r`).** Internal specs/plans scaffolding (the published repo ships no `superpowers/`).

> Sequencing note: WS4 is the final workstream (design doc §10 — "WS4 docs depend on the others landing"). README/ARCHITECTURE/CHANGELOG describe the **post-WS1** state: the event is `introspection_event` / `IntrospectionEvent`, Arize Phoenix is **gone**, and config is `AppConfig` (`config.yaml` + `.env`). If WS0/WS1 have not landed when WS4 runs, the doc *content* below still describes the target state — verify the referenced symbols exist before committing each doc task. The CI task (Task 11) is the only WS4 task that runs the test suite; it depends on WS1 having deleted `backend/tests/test_coherence_eval.py` and `backend/tests/test_phoenix_eval_features.py`. Task 11 lists the exact files to `--ignore` so it is robust whether or not those two files still exist.

---

### Task 1: Remove hackathon-internal docs (PLAN.md, LANES.md, docs/superpowers/)

Files:
- Delete: `PLAN.md`
- Delete: `LANES.md`
- Delete: `docs/superpowers/` (recursive — specs + plans)
- Test: shell assertions (no unit test for a deletion)

Interfaces:
- Consumes: nothing.
- Produces: a repo with zero tracked hackathon-internal docs. Later doc tasks (Task 2 README, Task 3 ARCHITECTURE, Task 8 deployment) must NOT link to `PLAN.md`, `LANES.md`, or `docs/superpowers/*`.

> NOTE: `docs/superpowers/plans/2026-06-22-ws4-oss-infra.md` is THIS plan. Deleting `docs/superpowers/` removes the plan file from the working tree, which is fine for the published repo — but do this task LAST if you are executing the plan from that file. The plan author has saved a copy outside the repo. Steps below assume you run this after all other tasks, OR that your executor reads the plan from memory/another path. If executing in order, skip to Task 2 and return here at the end.

- [ ] Step 1: Write a failing assertion that the hackathon docs are gone. Create the check script.
```bash
cat > /tmp/ws4_check_cruft.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
for p in PLAN.md LANES.md docs/superpowers; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    echo "STILL TRACKED: $p"; fail=1
  fi
done
if [ -e PLAN.md ] || [ -e LANES.md ] || [ -d docs/superpowers ]; then
  echo "STILL ON DISK"; fail=1
fi
[ "$fail" -eq 0 ] && echo "OK: hackathon docs removed"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_cruft.sh
```
- [ ] Step 2: Run the check; expect failure (files still present).
```bash
/tmp/ws4_check_cruft.sh
```
Expected output (non-zero exit):
```
STILL TRACKED: PLAN.md
STILL TRACKED: LANES.md
STILL TRACKED: docs/superpowers
STILL ON DISK
```
- [ ] Step 3: Remove the files from git and disk. Note that `docs/superpowers/` contains BOTH git-tracked files (the 2026-06-20/21 specs + plans) and untracked-on-disk files (the 2026-06-22 WS4 spec/plan/AppConfig-contract, including this plan). `git rm` only touches tracked files, so we follow it with `rm -rf` to clear the untracked remainder — otherwise `[ -d docs/superpowers ]` in the Step 1 check stays true and the check fails.
```bash
git rm -q PLAN.md LANES.md
git rm -rq --ignore-unmatch docs/superpowers
rm -rf docs/superpowers
```
- [ ] Step 4: Re-run the check; expect success.
```bash
/tmp/ws4_check_cruft.sh
```
Expected output (zero exit):
```
OK: hackathon docs removed
```
- [ ] Step 5: Commit.
```bash
git add -A
git commit -m "docs: remove hackathon-internal PLAN/LANES/superpowers scaffolding"
```

---

### Task 2: Rewrite README.md for the general (non-medical) framing + self-host quickstart

Files:
- Modify: `README.md` (full rewrite — the entire file is replaced)
- Test: shell grep assertions (README is prose; the test asserts absence of forbidden strings and presence of required sections)

Interfaces:
- Consumes: the `AppConfig` contract (config via `config.yaml` + `.env`; secrets env-only), the `introspection_event` name (post-WS1), the two-process architecture (§5 of the design doc).
- Produces: a README that links to `ARCHITECTURE.md` (Task 3), `CONTRIBUTING.md` (Task 4), `docs/deployment.md` (Task 8), `docs/probe-training.md` (existing). Those targets are created in later tasks; the links are forward references resolved by the end of the plan.

> SHARED CONTEXT: the README describes the post-WS1 world — no Arize Phoenix, event is `introspection_event`, mode flag is `GLASSBOX__RUNTIME__MODE` / `config.yaml`. Do NOT mention `scripts/run_phoenix.sh` (deleted in WS1), `cognition_event`, prizes, hackathon, or medical (except as one labeled optional example).

- [ ] Step 1: Write the failing README content test.
```bash
cat > /tmp/ws4_check_readme.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
forbidden=("hackathon" "Hackathon" "UC Berkeley" "prize" "Prize" "cognition_event" "Arize" "Phoenix" "run_phoenix" "clinician" "patient" "medical LLMs")
for s in "${forbidden[@]}"; do
  if grep -qi -- "$s" README.md; then echo "FORBIDDEN PRESENT: $s"; fail=1; fi
done
required=("GlassBox" "interpretability" "config.example.yaml" "ARCHITECTURE.md" "CONTRIBUTING.md" "docs/deployment.md" "introspection_event" "uv sync" "MIT")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" README.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
# Shared-file precondition: the README tells users to `cp .env.example .env`, so the
# committed .env.example must already be cleaned by WS1/WS3 before this doc is accurate.
if [ -f .env.example ]; then
  if grep -q "PHOENIX_" .env.example; then echo "STALE .env.example: PHOENIX_* present (WS1 must remove)"; fail=1; fi
  if grep -q "glassbox-dev-secret" .env.example; then echo "STALE .env.example: glassbox-dev-secret default present (WS3 must remove)"; fail=1; fi
fi
[ "$fail" -eq 0 ] && echo "OK: README framing correct"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_readme.sh
```
- [ ] Step 2: Run the test; expect failure (current README is medical/hackathon framed).
```bash
/tmp/ws4_check_readme.sh
```
Expected output (non-zero exit), e.g.:
```
FORBIDDEN PRESENT: hackathon
FORBIDDEN PRESENT: cognition_event
FORBIDDEN PRESENT: Arize
MISSING REQUIRED: config.example.yaml
MISSING REQUIRED: ARCHITECTURE.md
...
```
- [ ] Step 3: Replace README.md with the general-framing version.
```markdown
# GlassBox

**Open-source interpretability & observability for open-weight LLMs.** GlassBox instruments a model's *internal state* on every chat turn: a sparse-autoencoder (SAE) "feature cloud" of which concepts are firing, plus calibrated probes that flag when the model is **internally uncertain but verbally confident**. Each turn emits exactly one `introspection_event` that fans out to Sentry, the in-app Observe dashboard, and (when flagged) an async Claude honesty-judge.

GlassBox is a glass box for black-box models: see what the model is "thinking", not just what it says.

Licensed under [MIT](LICENSE).

---

## What it does

| Surface | What you see |
|---------|--------------|
| **Chat** | Talk to an open model with a live SAE feature cloud and probe meters alongside each response. |
| **Build** | Describe a behavior in natural language; Claude designs a contrastive probe, the model generates pairs, and a calibrated probe deploys live. |
| **Observe** | Probe-score trends, Sentry alarms for confident-but-uncertain turns, and per-turn KPIs. |

GlassBox is **general-purpose**: it ships validated on `unsloth/gemma-3-4b-it` + Gemma Scope, but the model and SAE are a configuration seam (see [`config.example.yaml`](config.example.yaml)), not hardcoded. A clearly-labeled clinical-decision-support profile ships as one optional example in `config.example.yaml` — GlassBox itself is domain-neutral.

---

## Quickstart

GlassBox runs in two tiers. **Tier A** needs no GPU and no model weights — the backend serves synthetic features so you can explore the UI. **Tier B** adds a GPU pod for real activations, live probes, and the Build pipeline.

### Tier A — UI walkthrough (no GPU)

```bash
# 1. Install the light dependency set (no ML stack)
uv sync

# 2. Configure structure + secrets
cp config.example.yaml config.yaml   # edit model/probes/branding; config.yaml is gitignored
cp .env.example .env                  # set ANTHROPIC_API_KEY (Build + labels); SENTRY_DSN optional

# 3. Start the orchestration backend (port 8000) — runs in synthetic fallback mode
uv run uvicorn backend.app:app --port 8000

# 4. Start the frontend (Vite on :5173, proxies /api -> :8000)
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

Verify startup:

```bash
curl -s localhost:8000/api/health
# → {"mode":"fallback","model":"unsloth/gemma-3-4b-it","layer":17,"trackers":[]}
```

In fallback mode `health` reports `"mode":"fallback"`, features are synthetic, and uncertainty scores are null. That is expected with no GPU pod attached.

### Tier B — full stack (GPU pod)

Run the GPU pod service (`backend.gpu_service`) on your own GPU box, point the backend at it via `pod.url`, and set secrets in `.env`. The end-to-end self-host walkthrough — Docker for the GPU pod, config, reverse proxy, and health checks — is in **[`docs/deployment.md`](docs/deployment.md)**.

```bash
curl -s localhost:8000/api/health
# → {"mode":"real","pod_reachable":true,...} once the pod is attached
```

---

## Configuration

GlassBox loads **structure** from `config.yaml` and **secrets** from the environment / `.env` — secrets never appear in `config.yaml`.

- `config.example.yaml` is the committed template; copy it to `config.yaml` (gitignored) and edit.
- With no `config.yaml`, the app boots on the built-in Gemma + Gemma Scope defaults.
- Secrets — `ANTHROPIC_API_KEY`, `SENTRY_DSN`, `POD_TOKEN`, `HF_TOKEN`, plus the optional `SENTRY_AUTH_TOKEN` (only for the Observe REST integration) — load **only** from `.env` / env. See `.env.example`.
- Any structural field is env-overridable (12-factor) via `GLASSBOX__SECTION__FIELD`, e.g. `GLASSBOX__MODEL__LAYER=20`.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full `AppConfig` contract.

---

## Architecture in one paragraph

GlassBox is two processes. A **CPU orchestration backend** (`backend/app.py`, never imports torch) serves the API and the UI; a **GPU pod service** (`backend/gpu_service.py`, owns torch + model + SAE + probes) does all inference. They talk over **POST + NDJSON** (never SSE). On the GPU, the model generates while one forward hook on the configured residual layer (default 17) captures the residual stream, which feeds two method families: **Family A** — an SAE feature cloud (top-k firing features, labels caveated) — and **Family B** — calibrated persona-vector probes (e.g. `harmful`, `over_confidence`, plus user-defined probes from the Build tab). Downstream on CPU, the handler assembles exactly one `introspection_event` and fans it out via `backend/fanout.py` to Sentry, the Observe store, the chat UI, and an async Claude judge. Full detail: [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
UI ──POST /api/chat──▶ [GPU pod] generate + residual-layer hook
                            │  (one residual activation)
                ┌───────────┴───────────┐
        [GPU] SAE top-k          [GPU] persona-vector probes
         (Family A: cloud)        (Family B: calibrated scores)
                └───────────┬───────────┘
                     [CPU] build ONE introspection_event
                            │ fanout()
              ┌─────────────┼──────────────┐
            Sentry       chat UI      Claude judge (async, if flagged)
```

The model/SAE are config-driven: v1 is validated on Gemma + Gemma Scope; generalizing to arbitrary HF model+SAE repos is tracked as an [open issue](docs/issues/01-arbitrary-hf-model-sae.md).

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — process split, `AppConfig`, the `introspection_event` contract, the science boundary.
- **[docs/deployment.md](docs/deployment.md)** — self-host & deployment guide (GPU pod via Docker, config, reverse proxy).
- **[docs/probe-training.md](docs/probe-training.md)** — training Family-B persona-vector probes.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — dev setup, lint/test, conventions.
- **[SECURITY.md](SECURITY.md)** — reporting vulnerabilities and the secrets/PHI posture.
- **[docs/tailscale-pod-runbook.md](docs/tailscale-pod-runbook.md)** — optional appendix for reaching a pod behind a restrictive firewall.

---

## License

[MIT](LICENSE).
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_readme.sh
```
Expected output (zero exit):
```
OK: README framing correct
```
- [ ] Step 5: Commit.
```bash
git add README.md
git commit -m "docs: rewrite README for general (non-medical) framing and self-host quickstart"
```

---

### Task 3: Add ARCHITECTURE.md

Files:
- Create: `ARCHITECTURE.md`
- Test: shell grep assertions

Interfaces:
- Consumes: the `AppConfig` contract (seven sub-models: `model`, `sae`, `feature_cloud`, `probes`, `observability`, `pod`, `runtime`), the `introspection_event` name, the two-process split, the science boundary functions (`sae.sae_topk`, `persona.score_all_trackers`, and the Build pipeline `concept_synth.create_job` / `generate_contrastive` / `judge_filter` — verify these against `backend/science/concept_synth.py`; the live module has NO `synth_concept`).
- Produces: `ARCHITECTURE.md` referenced by README (Task 2) and CONTRIBUTING (Task 4).

> SHARED CONTEXT: describe the post-WS0/WS1 state. The seven sub-models and field names come from the AppConfig contract verbatim. No Phoenix, no `cognition_event`.
>
> VERIFY THESE SYMBOLS/PATHS EXIST BEFORE COMMITTING (they describe target state and several land in sibling workstreams):
> - `class IntrospectionEvent` in `backend/schema.py` (WS1 renames `CognitionEvent` → `IntrospectionEvent`). Until WS1 lands, the repo still has `CognitionEvent`.
> - `fixtures/introspection_event.sample.json` (WS1 renames `fixtures/cognition_event.sample.json`). The doc asserts the new path; it only becomes true post-WS1.
> - The Build-pipeline functions `create_job`, `generate_contrastive`, `judge_filter` in `backend/science/concept_synth.py` (these exist today; there is NO `synth_concept`).
> - `sae.sae_topk` and `persona.score_all_trackers` as the Family A / Family B boundary surfaces.

- [ ] Step 1: Write the failing ARCHITECTURE content test.
```bash
cat > /tmp/ws4_check_arch.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f ARCHITECTURE.md ] || { echo "MISSING FILE: ARCHITECTURE.md"; exit 1; }
required=("AppConfig" "ModelConfig" "SAEConfig" "FeatureCloudConfig" "ProbeConfig" "ObsConfig" "PodConfig" "RuntimeConfig" "introspection_event" "app.state" "gpu_service" "sae_topk" "score_all_trackers" "config.yaml" ".env")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" ARCHITECTURE.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
forbidden=("Phoenix" "Arize" "cognition_event" "hackathon")
for s in "${forbidden[@]}"; do
  if grep -qi -- "$s" ARCHITECTURE.md; then echo "FORBIDDEN PRESENT: $s"; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "OK: ARCHITECTURE complete"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_arch.sh
```
- [ ] Step 2: Run the test; expect failure (file absent).
```bash
/tmp/ws4_check_arch.sh
```
Expected output:
```
MISSING FILE: ARCHITECTURE.md
```
- [ ] Step 3: Create `ARCHITECTURE.md`.
```markdown
# Architecture

GlassBox is two cooperating processes plus a thin science boundary. This document is the canonical reference for the four-and-a-half contracts that hold them together: the **process split**, the **`AppConfig`** contract, the **`introspection_event`** wire shape, and the **science boundary**.

## Two processes

| Process | Entry point | Owns | torch? |
|---------|-------------|------|--------|
| Orchestration backend | `backend/app.py` | HTTP API, NDJSON streaming, fanout to sinks, the Observe store, serving the UI | **No** — must never import torch |
| GPU pod service | `backend/gpu_service.py` | model load, the residual-layer hook, SAE top-k, probe scoring, the probe-builder agent | **Yes** — owns the entire torch/transformers/sae-lens stack |

They communicate over **POST + NDJSON** (`application/x-ndjson`): zero-or-more `{"type":"token",...}` lines followed by exactly one `{"type":"event", ...IntrospectionEvent}` line. NDJSON is used (not SSE) because SSE breaks through some reverse proxies/CDNs.

The orchestration backend reaches the pod over HTTP at `pod.url`. The pod authenticates requests with `POD_TOKEN` (fails closed on non-loopback binds). If the pod is unreachable, the backend serves **synthetic fallback** features so the UI still runs.

## The `AppConfig` contract

Configuration is a single validated object, `AppConfig` (in `backend/config.py`), built **once per process** at startup and stored on `app.state.config`, then threaded into the modules that need it. It replaces the former ~50 flat `config.X` globals.

- **Structure** loads from `config.yaml` (template: `config.example.yaml`; the working `config.yaml` is gitignored). With no `config.yaml`, built-in Gemma + Gemma Scope defaults apply and the app still boots.
- **Secrets** (`ANTHROPIC_API_KEY`, `SENTRY_DSN`, `SENTRY_AUTH_TOKEN`, `POD_TOKEN`, `HF_TOKEN`) load **only** from env / `.env` — never from `config.yaml`. A secret key placed in `config.yaml` is ignored.
- Any structural field is env-overridable with the `GLASSBOX__SECTION__FIELD` convention (e.g. `GLASSBOX__MODEL__LAYER=20`). Env beats YAML.

`AppConfig` has seven sub-models:

| Sub-model | Holds | Read by (illustrative) |
|-----------|-------|------------------------|
| `model: ModelConfig` | `model_id`, `layer`, `sae_layers`, `device`, `system_prompt`, `mask_tokens`, `preamble_skip`, `max_new_tokens` | engine, gpu_service, analyze |
| `sae: SAEConfig` | `release`, `sae_id_pattern`, `d_in`/`d_sae` (fallbacks), Neuronpedia label source, `recon_min_cosine`, `recon_probe` | science/sae, feature_provider, labels |
| `feature_cloud: FeatureCloudConfig` | top-k knobs, density/penalty filters, `rank_method`, contrast baseline, auto-interp | analyze, feature_provider, science/sae |
| `probes: ProbeConfig` | `enabled`/`disabled` probe sets, `artifacts_dir`, `default_threshold`, and `builder` (agent/judge models, AUROC gate) | science/persona, agent/tools |
| `observability: ObsConfig` | `sentry` sub-config (environment, `send_io` PHI gate, org/project slugs, API base) | app, fanout, sentry_api |
| `pod: PodConfig` | `url`, `timeout`, `poll_interval` | runtime, pod_client |
| `runtime: RuntimeConfig` | `mode` (`posthoc`/`live`), `product_name` | mode switch, branding |

Secrets are injected onto the parent `AppConfig` by the loader (`cfg.anthropic_api_key`, `cfg.sentry_dsn`, `cfg.sentry_auth_token`, `cfg.pod_token`, `cfg.hf_token`). Derived helpers: `cfg.sae_id(layer)`, `cfg.np_source(layer)`, `cfg.resolve_device(pref)`.

**Why this is the generalization seam:** swapping to a different model/SAE is "populate `model`/`sae` with new values", not a code rewrite. The arbitrary-HF-model/SAE work is tracked in [docs/issues/01](docs/issues/01-arbitrary-hf-model-sae.md).

## The `introspection_event` contract

Each chat turn produces exactly one `IntrospectionEvent` (defined in `backend/schema.py`; mirrored in `frontend/src/types.ts` and `fixtures/introspection_event.sample.json`). It carries the SAE feature cloud, probe scores, the uncertainty/flag summary, and (when adjudicated) the Claude judge verdict. The shape is versioned by `schema_version`. No part of this payload is domain-specific — it is model introspection telemetry.

The orchestration backend builds the event on CPU and passes it to `backend/fanout.py:fanout(...)`, the single seam where every sink lives:

```
UI → POST /api/chat → [pod] generate + layer hook → SAE top-k + probe scores
   → [CPU] build ONE introspection_event → fanout() → { Sentry (if flagged),
     in-process Observe store → /api/observability, chat UI, Claude judge (async) }
```

Adding or removing a sink is an edit to `fanout.py` only; nothing on the fanout path imports torch.

## The science boundary

The GPU pod exposes a small torch-only surface that the orchestration code calls over HTTP (never by direct import). The boundary functions are:

- `sae.sae_topk(act, k=...)` — Family A: top-k firing SAE features for the captured activation.
- `persona.score_all_trackers(act_last, act_resp)` — Family B: calibrated probe scores for every enabled tracker.
- The Build pipeline lives in `science/concept_synth.py`: `create_job(request) -> tracker_id` opens a build job, `generate_contrastive(...)` produces the contrastive prompt pairs, and `judge_filter(spec, rows, *, client=...)` runs the Claude judge to filter the generated rows before the probe is fit and deployed.

The **only object that crosses the GPU→science boundary is the configured-layer activation tensor**. Science modules never import FastAPI; the orchestration backend never touches torch internals.

> Note: `backend/science/__init__.py` still carries a stale docstring referencing a removed `concept_synth.synth_concept(name, desc)` entry point (fixing that docstring is out of WS4 scope). This document must NOT propagate that stale name — the live Build surface is `create_job` / `generate_contrastive` / `judge_filter`, verified against `backend/science/concept_synth.py`.

## Observability

GlassBox emits to **Sentry** only (Arize Phoenix was removed for v1). Sentry is fully env-gated: with no `SENTRY_DSN` the path is a no-op. The PHI gate (`observability.sentry.send_io`, default `false`) keeps raw user input/response out of Sentry events. PII scrubbing (`send_default_pii=False`) stays on.
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_arch.sh
```
Expected output:
```
OK: ARCHITECTURE complete
```
- [ ] Step 5: Commit.
```bash
git add ARCHITECTURE.md
git commit -m "docs: add ARCHITECTURE.md describing the two-process split, AppConfig, and event contract"
```

---

### Task 4: Add CONTRIBUTING.md

Files:
- Create: `CONTRIBUTING.md`
- Delete: `frontend/bun.lock` (hygiene backstop for the npm-only rule; canonical owner is WS0)
- Test: shell grep assertions

Interfaces:
- Consumes: the dev tooling (`uv sync`, `uv run ruff check`, `uv run pytest`), the GPU-gated-tests fact (engine/sae untested without weights), the commit rule (Conventional Commits, no Claude co-author).
- Produces: `CONTRIBUTING.md` referenced by README (Task 2). The exact lint/test commands here must match the CI workflow created in Task 11.

- [ ] Step 1: Write the failing CONTRIBUTING content test.
```bash
cat > /tmp/ws4_check_contributing.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f CONTRIBUTING.md ] || { echo "MISSING FILE: CONTRIBUTING.md"; exit 1; }
required=("uv sync" "ruff check" "uv run pytest" "non-GPU" "Conventional Commits" "config.example.yaml" ".env")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" CONTRIBUTING.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
# Shared-file precondition: CONTRIBUTING tells users to `cp .env.example .env`, so the
# committed .env.example must already be cleaned by WS1/WS3 before this doc is accurate.
if [ -f .env.example ]; then
  if grep -q "PHOENIX_" .env.example; then echo "STALE .env.example: PHOENIX_* present (WS1 must remove)"; fail=1; fi
  if grep -q "glassbox-dev-secret" .env.example; then echo "STALE .env.example: glassbox-dev-secret default present (WS3 must remove)"; fail=1; fi
fi
# CONTRIBUTING claims "Do not commit a bun.lock" — make that claim true at merge time:
# frontend/bun.lock must NOT be tracked once this task lands (npm is the standard).
if git ls-files --error-unmatch frontend/bun.lock >/dev/null 2>&1; then
  echo "STALE LOCKFILE: frontend/bun.lock still tracked (CONTRIBUTING says npm-only — remove it)"; fail=1
fi
[ "$fail" -eq 0 ] && echo "OK: CONTRIBUTING complete"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_contributing.sh
```
- [ ] Step 2: Run the test; expect failure (file absent).
```bash
/tmp/ws4_check_contributing.sh
```
Expected output:
```
MISSING FILE: CONTRIBUTING.md
```
- [ ] Step 3: Create `CONTRIBUTING.md`.
```markdown
# Contributing to GlassBox

Thanks for your interest in GlassBox. This guide covers local setup, the checks CI runs, and our conventions.

## Development setup

GlassBox uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management. Python >= 3.12 is required (see `.python-version`).

```bash
# Light install — enough for the orchestration backend, the full test suite's
# non-GPU subset, lint, and the synthetic-fallback UI.
uv sync

# Full install — adds the ML stack (torch, transformers, sae-lens, scikit-learn,
# accelerate, anthropic). Needed only where GPU weights are available.
uv sync --extra ml

# Configure
cp config.example.yaml config.yaml   # structure (model/probes/branding); gitignored
cp .env.example .env                  # secrets: ANTHROPIC_API_KEY, SENTRY_DSN, POD_TOKEN, HF_TOKEN
```

`pyproject.toml` is the single source of truth for dependencies. `backend/requirements.txt` is retired — do not add deps there.

## Lint

We use [`ruff`](https://docs.astral.sh/ruff/) for linting (config in `pyproject.toml`). Run it before pushing:

```bash
uv run ruff check backend
```

## Tests

The backend test suite lives in `backend/tests/`. CI runs the **non-GPU subset** — the tests that require the ML stack (torch directly, or transitively via numpy / `backend.science.persona` — `engine`/`science.sae` and the probe/agent integration tests) are GPU-gated and stay opt-in, because they need the ML extra and model weights CI does not have.

```bash
# Non-GPU subset (what CI runs) — no ML extra required:
uv run pytest backend/tests \
  --ignore=backend/tests/test_agent_persist.py \
  --ignore=backend/tests/test_attribution.py \
  --ignore=backend/tests/test_fit.py \
  --ignore=backend/tests/test_generate.py \
  --ignore=backend/tests/test_gpu_service.py \
  --ignore=backend/tests/test_harmfulness_pipeline.py \
  --ignore=backend/tests/test_judge.py \
  --ignore=backend/tests/test_loop.py \
  --ignore=backend/tests/test_persona.py \
  --ignore=backend/tests/test_prompt_tracker.py

# Full suite (requires `uv sync --extra ml` and, for some tests, a GPU/MPS device):
uv run pytest backend/tests
```

`backend/engine.py` and `backend/science/sae.py` are GPU-gated — they are exercised only with model weights present and are not run in CI.

## Frontend

The frontend is a Vite + React app in `frontend/`. We standardize on **npm** (`npm install` / `npm run dev`). Do not commit a `bun.lock`.

```bash
cd frontend && npm install && npm run dev   # Vite on :5173, proxies /api -> :8000
```

## Conventions

- **Commit messages:** use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `ci:`). Keep messages imperative and scoped.
- **Secrets never go in `config.yaml`** or any committed file — only in `.env` / the environment. The five secrets are `ANTHROPIC_API_KEY`, `SENTRY_DSN`, `SENTRY_AUTH_TOKEN`, `POD_TOKEN`, and `HF_TOKEN` (`SENTRY_AUTH_TOKEN` is the optional fifth, only for the Observe REST integration).
- **The orchestration backend (`backend/app.py`) must never import torch.** Torch lives only in the GPU pod service and the `backend/science/*` modules it calls.
- Open a PR against `main`; fill in the PR template; make sure `ruff check` and the non-GPU pytest subset pass.

## Reporting security issues

Please do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
```
- [ ] Step 3b: Hygiene — make the "npm-only, do not commit a bun.lock" rule true. The repo currently tracks BOTH `frontend/bun.lock` and `frontend/package-lock.json`; the design doc assigns lockfile cleanup to WS0, but WS4 enforces it here so the doc's claim is not contradicted at merge time. Remove the bun lockfile (idempotent if WS0 already did it):
```bash
git rm -q --ignore-unmatch frontend/bun.lock
rm -f frontend/bun.lock
```
> NOTE (cross-plan): the canonical lockfile cleanup lives in WS0. This step is the WS4 backstop so CONTRIBUTING.md's npm-only claim is accurate; if WS0 has already deleted `frontend/bun.lock`, the commands above are no-ops.
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_contributing.sh
```
Expected output:
```
OK: CONTRIBUTING complete
```
- [ ] Step 5: Commit.
```bash
git add CONTRIBUTING.md
git rm -q --cached --ignore-unmatch frontend/bun.lock 2>/dev/null || true
git commit -m "docs: add CONTRIBUTING.md with dev setup, lint/test commands, and conventions; drop frontend/bun.lock"
```

---

### Task 5: Add SECURITY.md

Files:
- Create: `SECURITY.md`
- Test: shell grep assertions

Interfaces:
- Consumes: the secrets-never-in-YAML posture, the PHI gate (`observability.sentry.send_io` default false), the fail-closed pod auth (WS3), the `0.1.0` version.
- Produces: `SECURITY.md` referenced by README (Task 2) and CONTRIBUTING (Task 4).

- [ ] Step 1: Write the failing SECURITY content test.
```bash
cat > /tmp/ws4_check_security.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f SECURITY.md ] || { echo "MISSING FILE: SECURITY.md"; exit 1; }
required=("Supported Versions" "0.1" "Reporting" "secrets" ".env" "POD_TOKEN" "send_io")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" SECURITY.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "OK: SECURITY complete"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_security.sh
```
- [ ] Step 2: Run the test; expect failure (file absent).
```bash
/tmp/ws4_check_security.sh
```
Expected output:
```
MISSING FILE: SECURITY.md
```
- [ ] Step 3: Create `SECURITY.md`.
```markdown
# Security Policy

## Supported Versions

GlassBox is pre-1.0. Security fixes are applied to the latest release on `main`.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately via GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository (Security tab → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce,
- affected version / commit.

We aim to acknowledge reports within 5 business days.

## Security posture

GlassBox is designed so that running it does not leak credentials or sensitive input:

- **Secrets are env-only.** `ANTHROPIC_API_KEY`, `SENTRY_DSN`, `SENTRY_AUTH_TOKEN`, `POD_TOKEN`, and `HF_TOKEN` load **only** from the environment / `.env`. They are never read from `config.yaml` — a secret key placed in `config.yaml` is explicitly ignored by the loader. `.env` is gitignored.
- **GPU pod auth fails closed.** The pod service requires `POD_TOKEN` for any non-loopback bind and rejects requests when the token is unset. Generate a strong token (e.g. `openssl rand -hex 32`) and never commit it. See [docs/deployment.md](docs/deployment.md).
- **PHI / raw-I/O gate.** `observability.sentry.send_io` defaults to `false`, so raw user messages and model responses are **not** attached to Sentry events. Sentry's own PII scrubbing (`send_default_pii=False`) is also enabled.
- **No telemetry by default.** With no `SENTRY_DSN` set, the Sentry path is a complete no-op.

If you deploy GlassBox publicly, terminate TLS at a reverse proxy and keep the GPU pod bound to a private network. See [docs/deployment.md](docs/deployment.md).
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_security.sh
```
Expected output:
```
OK: SECURITY complete
```
- [ ] Step 5: Commit.
```bash
git add SECURITY.md
git commit -m "docs: add SECURITY.md with private reporting and the secrets/PHI posture"
```

---

### Task 6: Add CHANGELOG.md

Files:
- Create: `CHANGELOG.md`
- Test: shell grep assertions

Interfaces:
- Consumes: the `0.1.0` version from `pyproject.toml`, the milestone summary (general framing, AppConfig, Phoenix removal, introspection_event rename, self-host).
- Produces: `CHANGELOG.md` referenced by no other doc but expected by OSS conventions.

- [ ] Step 1: Write the failing CHANGELOG content test.
```bash
cat > /tmp/ws4_check_changelog.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f CHANGELOG.md ] || { echo "MISSING FILE: CHANGELOG.md"; exit 1; }
required=("Keep a Changelog" "Semantic Versioning" "0.1.0" "AppConfig" "introspection_event" "Removed" "Phoenix")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" CHANGELOG.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "OK: CHANGELOG complete"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_changelog.sh
```
- [ ] Step 2: Run the test; expect failure (file absent).
```bash
/tmp/ws4_check_changelog.sh
```
Expected output:
```
MISSING FILE: CHANGELOG.md
```
- [ ] Step 3: Create `CHANGELOG.md`.
```markdown
# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-22

First open-source release. GlassBox is repositioned from a hackathon prototype into a
general-purpose, self-hostable LLM interpretability & observability tool.

### Added
- Unified `AppConfig` (pydantic-settings) as the single source of truth: structure from
  `config.yaml`, secrets from `.env`/env only. `config.example.yaml` ships as the template;
  the app boots on Gemma + Gemma Scope defaults with no `config.yaml`.
- Self-host & deployment guide (`docs/deployment.md`): GPU pod via Docker, config, reverse proxy.
- OSS infrastructure: `ARCHITECTURE.md`, `CONTRIBUTING.md`, `SECURITY.md`, this changelog,
  GitHub Actions CI (lint + the non-GPU pytest subset), and PR/issue templates.
- Four open-source issues capturing deferred generalizations (`docs/issues/`): arbitrary HF
  model+SAE, multi-provider agent, managed GPU endpoint, and serialized probe calibrator.

### Changed
- Repositioned from "cognition-observability for medical LLMs" to a domain-neutral,
  general-purpose tool. The medical clinical-decision-support prompt survives as one
  clearly-labeled optional profile in `config.example.yaml`.
- Renamed the per-turn event `cognition_event` → `introspection_event` (class
  `CognitionEvent` → `IntrospectionEvent`); fixture renamed accordingly.
- GPU pod auth now fails closed: `POD_TOKEN` is required for non-loopback binds.

### Removed
- Arize Phoenix integration removed entirely (sink, eval endpoint, panels, deps, env vars).
  Sentry remains the sole external observability sink.
- Hackathon-internal docs (`PLAN.md`, `LANES.md`, `docs/superpowers/`) and tracked cruft.

[Unreleased]: https://github.com/aniruddh-alt/glassbox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aniruddh-alt/glassbox/releases/tag/v0.1.0
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_changelog.sh
```
Expected output:
```
OK: CHANGELOG complete
```
- [ ] Step 5: Commit.
```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md documenting the 0.1.0 open-source release"
```

---

### Task 7: Verify the LICENSE copyright holder (VERIFY-ONLY — WS0 owns the write)

Files:
- Verify: `LICENSE` (line 3) — **no write here**
- Test: shell grep assertion

Interfaces:
- Consumes: the LICENSE copyright line written by **WS0 Task 20** (the single owner of the LICENSE copyright line).
- Produces: nothing — this task only asserts the LICENSE holder string is present.

> **Single-owner note:** WS0 Task 20 is the ONLY plan that writes the LICENSE copyright line, setting it to `Copyright (c) 2026 Aniruddhan Ramesh and the GlassBox contributors`. WS4 does NOT rewrite the file — it verifies that exact same line is present (the assertion below is byte-for-byte identical to the WS0 string). If the assertion fails, WS0 Task 20 has not landed yet (or used a different string); fix it in WS0, not here. Keep MIT and keep the year 2026.

- [ ] Step 1: Write the LICENSE verification test (asserts the WS0-written holder line is present).
```bash
cat > /tmp/ws4_check_license.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
grep -q "^MIT License" LICENSE || { echo "NOT MIT"; fail=1; }
grep -q "Copyright (c) 2026 Aniruddhan Ramesh and the GlassBox contributors" LICENSE \
  || { echo "MISSING REAL HOLDER (WS0 Task 20 must write it)"; fail=1; }
[ "$fail" -eq 0 ] && echo "OK: LICENSE holder verified"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_license.sh
```
- [ ] Step 2: Run the test. It PASSES once WS0 Task 20 has written the holder line; if it fails with `MISSING REAL HOLDER`, land WS0 Task 20 first (do NOT edit LICENSE here).
```bash
/tmp/ws4_check_license.sh
```
Expected output (once WS0 has landed):
```
OK: LICENSE holder verified
```
- [ ] Step 3: No write step — WS0 owns the LICENSE copyright line. This task is verify-only; there is nothing to commit for LICENSE in WS4.

---

### Task 8: Add docs/deployment.md (self-host guide) and demote the Tailscale runbook

Files:
- Create: `docs/deployment.md`
- Modify: `docs/tailscale-pod-runbook.md` (lines 1-6, header)
- Test: shell grep assertions

Interfaces:
- Consumes: the two-process model, `config.yaml` + `.env`, `pod.url`/`POD_TOKEN`, fail-closed pod auth (WS3), the `backend.gpu_service:app` ASGI entry point, a `Dockerfile`/`compose` for the pod (WS3 ships these; this doc documents how to use them and degrades gracefully if they are absent).
- Produces: `docs/deployment.md` referenced by README (Task 2), SECURITY (Task 5), and CHANGELOG (Task 6). The Tailscale runbook now points back here as an optional appendix.

> SHARED CONTEXT: WS3 ships the `Dockerfile` + compose for `backend/gpu_service`. This guide references `docker compose up` for the pod; if WS3's Docker files are not present yet, the manual `uvicorn` path documented here still works. Do NOT reintroduce Phoenix.

- [ ] Step 1: Write the failing deployment-guide test.
```bash
cat > /tmp/ws4_check_deploy.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f docs/deployment.md ] || { echo "MISSING FILE: docs/deployment.md"; exit 1; }
required=("Self-host" "config.yaml" ".env" "POD_TOKEN" "openssl rand" "backend.gpu_service:app" "pod:" "reverse proxy" "uvicorn backend.app:app" "/api/health")
for s in "${required[@]}"; do
  if ! grep -q -- "$s" docs/deployment.md; then echo "MISSING REQUIRED: $s"; fail=1; fi
done
forbidden=("Phoenix" "Arize")
for s in "${forbidden[@]}"; do
  if grep -qi -- "$s" docs/deployment.md; then echo "FORBIDDEN PRESENT: $s"; fail=1; fi
done
# Shared-file precondition: deployment.md tells users to `cp .env.example .env`, so the
# committed .env.example must already be cleaned by WS1/WS3 before this doc is accurate.
if [ -f .env.example ]; then
  if grep -q "PHOENIX_" .env.example; then echo "STALE .env.example: PHOENIX_* present (WS1 must remove)"; fail=1; fi
  if grep -q "glassbox-dev-secret" .env.example; then echo "STALE .env.example: glassbox-dev-secret default present (WS3 must remove)"; fail=1; fi
fi
# Tailscale runbook must be demoted (point at deployment.md, call itself optional/appendix)
grep -qi "appendix\|optional" docs/tailscale-pod-runbook.md || { echo "RUNBOOK NOT DEMOTED"; fail=1; }
grep -q "docs/deployment.md" docs/tailscale-pod-runbook.md || { echo "RUNBOOK MISSING LINK"; fail=1; }
[ "$fail" -eq 0 ] && echo "OK: deployment guide + demoted runbook"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_deploy.sh
```
- [ ] Step 2: Run the test; expect failure (file absent, runbook not demoted).
```bash
/tmp/ws4_check_deploy.sh
```
Expected output:
```
MISSING FILE: docs/deployment.md
```
- [ ] Step 3: Create `docs/deployment.md`.
```markdown
# Self-host & Deployment Guide

This is the canonical guide for running GlassBox on your own infrastructure. It covers the
two processes, configuration, the GPU pod, and exposing the app behind a reverse proxy.

For reaching a GPU pod that sits behind a restrictive firewall (e.g. campus WiFi that blocks
outbound SSH), see the optional appendix [docs/tailscale-pod-runbook.md](tailscale-pod-runbook.md).

## Topology

GlassBox is two processes:

```
[ users ] → reverse proxy (TLS) → orchestration backend (backend.app, CPU)
                                        │  HTTP, pod.url + POD_TOKEN
                                        ▼
                                   GPU pod service (backend.gpu_service, torch)
```

- **Orchestration backend** (`backend.app:app`) — CPU only, serves the API + UI. Safe to run
  on a small instance. Never imports torch.
- **GPU pod service** (`backend.gpu_service:app`) — needs a CUDA GPU, downloads the model + SAE,
  does all inference. Run it on your GPU box.

You can run both on one GPU machine for a single-node deployment, or split them.

## 1. Configuration

GlassBox loads **structure** from `config.yaml` and **secrets** from `.env` / the environment.

```bash
cp config.example.yaml config.yaml   # edit model/sae/probes/branding; config.yaml is gitignored
cp .env.example .env
```

Set secrets in `.env` (never in `config.yaml`):

```bash
ANTHROPIC_API_KEY=sk-ant-...     # Build pipeline + auto-interp labels + honesty judge
SENTRY_DSN=                      # optional; empty = Sentry disabled (no-op)
POD_TOKEN=...                    # see "Generate a pod token" below — REQUIRED for the pod
HF_TOKEN=                        # only if your model/SAE repo is gated
SENTRY_AUTH_TOKEN=               # optional fifth secret; only for the Observe REST integration
```

Point the backend at the pod in `config.yaml`:

```yaml
pod:
  url: http://<pod-host>:8001
  timeout: 120
```

Any structural field can be overridden by env using the `GLASSBOX__SECTION__FIELD` convention,
e.g. `GLASSBOX__POD__URL=http://10.0.0.5:8001`.

### Generate a pod token

`POD_TOKEN` authenticates the backend → pod connection. The pod **fails closed** — it rejects
requests when the token is unset on a non-loopback bind. Generate a strong token and set it on
**both** processes:

```bash
openssl rand -hex 32
# put the same value in .env on the backend host and in the pod's environment
```

## 2. Run the GPU pod service

### Option A — Docker (recommended)

> **Requires the `Dockerfile` + compose file shipped with the GPU-pod release.** If your
> checkout does not yet contain them (`ls Dockerfile docker-compose.yml`), use **Option B**
> (manual `uvicorn`) below until they land.

GlassBox ships a `Dockerfile` and a compose file for the GPU pod (torch + transformers +
model/SAE download). From the repo root on your GPU box:

```bash
# Set POD_TOKEN, ANTHROPIC_API_KEY, and (if gated) HF_TOKEN in .env first.
docker compose up gpu_service        # builds + runs backend.gpu_service:app on the GPU
```

The pod binds to port 8001 by default. First boot downloads weights and can take several
minutes; watch the container logs.

### Option B — manual (no Docker)

```bash
uv sync --extra ml
export POD_TOKEN=...                  # the token you generated
export HF_HOME=/path/to/hf-cache      # persist model weights across restarts
uv run uvicorn backend.gpu_service:app --host 0.0.0.0 --port 8001
```

Bind to `0.0.0.0` only on a private network or behind a firewall; the token is mandatory there.

Verify the pod:

```bash
curl -s http://localhost:8001/health
# → {"mode":"real","model_loaded":true,"sae_loaded":true,...}
```

## 3. Run the orchestration backend

On the backend host (CPU is fine):

```bash
uv sync
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Verify the full stack is wired:

```bash
curl -s http://localhost:8000/api/health
# → {"mode":"real","pod_reachable":true,...}
```

If `pod_reachable` is false or `mode` is `fallback`, the backend cannot reach `pod.url` —
check the pod is up, the URL is correct, and the `POD_TOKEN` matches on both sides. With the
pod down, the backend serves synthetic features so the UI still loads.

## 4. Frontend

Build the static frontend and serve it from your reverse proxy, or run the dev server:

```bash
cd frontend && npm install && npm run build   # output in frontend/dist
# or for development:
npm run dev                                    # Vite on :5173, proxies /api -> :8000
```

## 5. Reverse proxy & TLS

Put a reverse proxy (nginx, Caddy, Traefik, …) in front of the orchestration backend to
terminate TLS and serve the built frontend. Keep the **GPU pod on a private network** — it
should not be internet-exposed. Example nginx location for the API:

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;   # NDJSON streaming: do not buffer /api/chat responses
}
```

`proxy_buffering off` matters: `/api/chat` streams NDJSON, and a buffering proxy will delay the
token stream. SSE is deliberately not used.

## 6. Health & troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/api/health` shows `mode: fallback` | Backend can't reach the pod | Verify `pod.url`, pod is up, tokens match |
| Pod returns 401/403 | `POD_TOKEN` mismatch or unset | Set the same token on both processes |
| Chat stream stalls behind proxy | Proxy buffering | `proxy_buffering off` on `/api/` |
| Model download hangs | Gated repo without token | Set `HF_TOKEN` in `.env` |

For firewalled networks where the backend cannot reach the pod directly, see the optional
appendix [docs/tailscale-pod-runbook.md](tailscale-pod-runbook.md).
```
- [ ] Step 4: Edit `docs/tailscale-pod-runbook.md` header (lines 1-6) to demote it. Replace:
```
# Tailscale GPU Pod Runbook

How to run GlassBox on **eduroam** (or any network that blocks outbound SSH to RunPod’s public IP) by reaching the GPU pod over a **Tailscale tailnet** instead of `74.2.x.x:15331`.

Use this doc when chat stops working, health shows `mode: fallback`, or you’ve restarted your laptop / pod.
```
with:
```
# Appendix: Tailscale GPU Pod Runbook (optional)

> **This is an optional appendix.** The canonical self-host & deployment guide is
> [docs/deployment.md](deployment.md). Use this runbook **only** when the GPU pod is behind a
> restrictive firewall that blocks the normal backend → pod HTTP connection (for example campus
> WiFi that blocks outbound SSH to a public IP). It documents reaching the pod over a Tailscale
> tailnet + SSH tunnel instead of a direct `pod.url`.

Use this doc when, on such a network, chat stops working, health shows `mode: fallback`, or you’ve restarted your laptop / pod.
```
- [ ] Step 5: Run the test; expect success.
```bash
/tmp/ws4_check_deploy.sh
```
Expected output:
```
OK: deployment guide + demoted runbook
```
- [ ] Step 6: Commit.
```bash
git add docs/deployment.md docs/tailscale-pod-runbook.md
git commit -m "docs: add self-host deployment guide and demote Tailscale runbook to an appendix"
```

---

### Task 9: Author the four OSS issues under docs/issues/

Files:
- Create: `docs/issues/01-arbitrary-hf-model-sae.md`
- Create: `docs/issues/02-multi-provider-agent.md`
- Create: `docs/issues/03-managed-gpu-endpoint.md`
- Create: `docs/issues/04-serialize-probe-calibrator.md`
- Test: shell grep assertions

Interfaces:
- Consumes: design doc §9 (the four issues, verbatim intent), plus concrete repo anchors: `backend/config.py` (`SAE_RELEASE`/`SAE_ID`), `backend/engine.py:_find_decoder_layers`, `backend/agent/interp_agent.py:run_interp_agent(client=...)`, `backend/science/concept_synth.py:judge_filter(client=...)`, `backend/gpu_service.py:_require_auth`, `docs/probe-training.md` (calibrator-not-persisted caveat).
- Produces: four ready-to-post GitHub issue files referenced by README (issue #1) and CHANGELOG (Task 6).

- [ ] Step 1: Write the failing issues test.
```bash
cat > /tmp/ws4_check_issues.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
declare -A need=(
 ["docs/issues/01-arbitrary-hf-model-sae.md"]="SAEConfig|Neuronpedia|tokenizer|preamble|## Motivation|## Acceptance"
 ["docs/issues/02-multi-provider-agent.md"]="LiteLLM|interp_agent|judge_filter|tool-use|## Motivation|## Acceptance"
 ["docs/issues/03-managed-gpu-endpoint.md"]="POD_URL|multi-tenant|TLS|rate|## Motivation|## Acceptance"
 ["docs/issues/04-serialize-probe-calibrator.md"]="calibrator|LogReg|artifact|restart|## Motivation|## Acceptance"
)
for f in "${!need[@]}"; do
  [ -f "$f" ] || { echo "MISSING FILE: $f"; fail=1; continue; }
  IFS='|' read -ra toks <<< "${need[$f]}"
  for t in "${toks[@]}"; do
    grep -q -- "$t" "$f" || { echo "MISSING in $f: $t"; fail=1; }
  done
  grep -qi "^# " "$f" || { echo "NO TITLE in $f"; fail=1; }
done
[ "$fail" -eq 0 ] && echo "OK: four OSS issues complete"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_issues.sh
```
- [ ] Step 2: Run the test; expect failure (no files).
```bash
/tmp/ws4_check_issues.sh
```
Expected output:
```
MISSING FILE: docs/issues/01-arbitrary-hf-model-sae.md
...
```
- [ ] Step 3a: Create `docs/issues/01-arbitrary-hf-model-sae.md`.
```markdown
# Generalize to an arbitrary HF model + SAE

**Labels:** enhancement, model-seam
**Tracked from:** open-source-readiness milestone (§9.1)

## Motivation

v1 is validated on `unsloth/gemma-3-4b-it` + Gemma Scope only. The `AppConfig` `model`/`sae`
seam (`ModelConfig`, `SAEConfig` in `backend/config.py`) was designed so swapping is config, not
a rewrite — but several Gemma/SAELens/Neuronpedia assumptions still need to become generic before
arbitrary repos work. This issue makes "any HF model + any SAE" additive on top of the existing seam.

## Scope

- **Custom SAE loader.** Read weights + config from any HF repo (raw `safetensors` + a small
  config), not only SAELens releases. Keep `sae.release` / `sae.sae_id_pattern` as the SAELens
  path; add a `sae.source: "hf"` branch that loads from a repo id.
- **Label fallback when there is no Neuronpedia coverage.** Today `sae.np_*` assumes Neuronpedia
  has the SAE. When `np_source` returns nothing, fall back to auto-interp only and clearly mark
  features as unlabeled.
- **Tokenizer-derived special tokens.** Replace the hardcoded Gemma `model.mask_tokens`
  (`<bos>`, `<start_of_turn>`, `<end_of_turn>`) with values derived from the loaded tokenizer's
  special-token map.
- **Preamble auto-detect.** Replace the fixed `model.preamble_skip` (Gemma's formulaic preamble)
  with a detector, falling back to the configured value.
- **Layer / d_in / d_sae cross-checks.** Derive `d_in`/`d_sae` from the loaded SAE (config values
  as fallback — partially done in WS2) and assert the SAE's `d_in` matches the model's residual
  width at `model.layer`; the auto decoder-layer detection (`engine._find_decoder_layers`) is
  already model-agnostic and should be reused.

## Acceptance

- A non-Gemma model + a non-SAELens SAE can be configured purely via `config.yaml` (`model`/`sae`)
  with no code changes to load and score a turn.
- Missing Neuronpedia coverage degrades to auto-interp labels with an "unlabeled/unverified" marker
  rather than erroring.
- Mismatched `d_in`/layer width fails loudly at load with an actionable message.
- `engine.py`/`science/sae.py` remain GPU-gated; a CPU unit test covers the config-resolution and
  cross-check logic with mocked SAE metadata.
```
- [ ] Step 3b: Create `docs/issues/02-multi-provider-agent.md`.
```markdown
# Multi-provider agent (decouple the probe builder from Claude)

**Labels:** enhancement, agent
**Tracked from:** open-source-readiness milestone (§9.2)

## Motivation

The Build pipeline (probe builder) is Claude-only by design in v1 — it leans on Claude's
`thinking`/structured-output/tool-use shapes. The client-injection points already exist
(`backend/agent/interp_agent.py:run_interp_agent(..., client=...)` and
`backend/science/concept_synth.py:judge_filter(..., client=...)`), so a provider abstraction is
additive. This issue adds a provider layer so OpenAI / Gemini / local models can drive the builder.

## Scope

- Introduce a provider abstraction (e.g. **LiteLLM** or a thin internal interface) over the three
  Claude-specific shapes the builder uses: extended `thinking`, structured/JSON output, and tool-use.
- Thread provider choice through `ProbeBuilderConfig` (`probes.builder` — alongside `agent_model`
  / `judge_model`) so it is config-selected, not hardcoded.
- Preserve the existing `client=` injection points so a future per-request / per-provider client is
  trivial.
- Map each provider's tool-use / structured-output dialect to the builder's internal contract; fail
  with a clear message when a provider cannot satisfy a required capability.

## Acceptance

- A user can select a non-Claude provider via `config.yaml` (`probes.builder`) + the relevant
  provider key in `.env`, and build + deploy a probe end-to-end.
- The Claude path is unchanged (same behavior, same tests green).
- A unit test exercises the provider abstraction against a mocked client for at least Claude and
  one other provider, covering the tool-use and structured-output paths.
```
- [ ] Step 3c: Create `docs/issues/03-managed-gpu-endpoint.md`.
```markdown
# Optional managed / you-host-it GPU endpoint

**Labels:** enhancement, deployment
**Tracked from:** open-source-readiness milestone (§9.3)

## Motivation

v1 is **self-host first**: users run `backend.gpu_service` on their own GPU and the backend points
`POD_URL` (`pod.url`) at it. We deliberately never "SSH into the user's box". Some users would
prefer a managed endpoint they can point `POD_URL` at without operating a GPU. This issue specs a
hosted, multi-tenant `gpu_service` — strictly opt-in and separate from the self-host default.

## Scope

- A hosted deployment of `backend.gpu_service` that users point `POD_URL` at, with **real
  multi-tenant auth** (per-tenant credentials, not the single shared `POD_TOKEN`).
- **TLS** termination and transport security for the public endpoint.
- **Rate limiting** and per-tenant quotas.
- Tenant isolation for loaded probes/artifacts so one tenant cannot read another's probes.
- Clear docs distinguishing the managed endpoint from the self-host path; the self-host path stays
  the default and the recommended posture for sensitive data.

## Acceptance

- A backend can point `pod.url` at the managed endpoint with per-tenant credentials and run a turn.
- Auth is multi-tenant and fails closed; the single-token self-host path
  (`gpu_service._require_auth`) is unaffected.
- TLS and rate-limiting are enforced and documented.
- Self-host remains fully functional and the documented default.
```
- [ ] Step 3d: Create `docs/issues/04-serialize-probe-calibrator.md`.
```markdown
# Serialize the probe calibrator across pod restarts

**Labels:** enhancement, probes
**Tracked from:** open-source-readiness milestone (§9.4)

## Motivation

Probe artifacts persist `direction` + `threshold`, but the **full calibrator** (LogReg weights /
normalization stats) is not serialized — see `docs/probe-training.md` (the "Calibrator not
persisted" caveat). After a pod restart, live scoring falls back to direction-projection + sigmoid
unless an in-memory calibrator was registered, so the calibrated form is lost. This issue persists
the full calibrator so scoring is identical before and after a restart.

## Scope

- Extend the artifact JSON schema (`backend/science/artifacts/*.json`) to store the full calibrator:
  LogReg coefficients/intercept and any normalization stats (`norm_mean`/`norm_std` already exist
  for the normed direction method).
- Update the training pipeline to write the calibrator into the artifact.
- Update `science/persona.load_artifacts` / scoring to load and apply the persisted calibrator,
  falling back to direction-projection only when absent (backward compatible with old artifacts).
- Bump the artifact schema version and document the format in `docs/probe-training.md`.

## Acceptance

- A probe trained, deployed, and scored produces the **same** score before and after a pod restart
  (within float tolerance), with no in-memory re-registration.
- Old artifacts without a serialized calibrator still load and score via the direction-projection
  fallback.
- A CPU unit test round-trips a calibrator through the artifact JSON and asserts score equality.
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_issues.sh
```
Expected output:
```
OK: four OSS issues complete
```
- [ ] Step 5: Commit.
```bash
git add docs/issues
git commit -m "docs: add the four open-source issues as ready-to-post issue text"
```

---

### Task 10: Update pyproject.toml — description, ruff config, ruff dev dep, gpu marker

Files:
- Modify: `pyproject.toml` (line 4 `description`; lines 25-28 `[dependency-groups].dev`; lines 30-32 `[tool.pytest.ini_options]`; append `[tool.ruff]`)
- Test: `uv run ruff check backend` (proves ruff is installed + configured); `uv run python -c` (proves marker registered, no `addopts` warning)

Interfaces:
- Consumes: ruff 0.13 (already on PATH; we pin it as a dev dep so CI installs it), pytest 8.
- Produces: a `ruff` dev dependency, a `[tool.ruff]` config, and a registered `gpu` pytest marker — all consumed by the CI workflow (Task 11) and CONTRIBUTING (Task 4). The general-framing package `description` is owned/written by WS0 Task 18; WS4 only verifies it.

> SHARED FILE: `pyproject.toml` is also edited by WS0 (adds `pydantic-settings>=2.0`; **WS0 also owns and writes the `project.description`**) and possibly WS1 (drops arize deps from any extras). WS4 WRITES ONLY: `[dependency-groups].dev`, `[tool.pytest.ini_options].markers`, and a new `[tool.ruff]` block; it VERIFIES (does NOT write) `project.description`. Do not touch `dependencies` or the `ml` extra. If WS0 has already reformatted the file, re-apply these edits to the current content.

- [ ] Step 1: Write the failing pyproject test.
```bash
cat > /tmp/ws4_check_pyproject.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
grep -q 'description = "Open-source interpretability & observability for open-weight LLMs"' pyproject.toml \
  || { echo "DESCRIPTION NOT UPDATED"; fail=1; }
grep -q '"ruff>=0.13"' pyproject.toml || { echo "RUFF DEV DEP MISSING"; fail=1; }
grep -q "\[tool.ruff\]" pyproject.toml || { echo "TOOL.RUFF MISSING"; fail=1; }
grep -q 'markers = \[' pyproject.toml || { echo "MARKERS MISSING"; fail=1; }
grep -qi "gpu:" pyproject.toml || { echo "GPU MARKER MISSING"; fail=1; }
if grep -qi "medical LLMs" pyproject.toml; then echo "MEDICAL DESCRIPTION REMAINS"; fail=1; fi
[ "$fail" -eq 0 ] && echo "OK: pyproject updated"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_pyproject.sh
```
- [ ] Step 2: Run the test; expect failure.
```bash
/tmp/ws4_check_pyproject.sh
```
Expected output:
```
DESCRIPTION NOT UPDATED
RUFF DEV DEP MISSING
TOOL.RUFF MISSING
MARKERS MISSING
GPU MARKER MISSING
MEDICAL DESCRIPTION REMAINS
```
- [ ] Step 3a: VERIFY-ONLY — do NOT write the `description`. **WS0 Task 18 owns and writes the `project.description`**, setting it to EXACTLY `Open-source interpretability & observability for open-weight LLMs`. WS4 only asserts that exact string is present (the grep in Step 1 covers it). Confirm:
```bash
grep -q 'description = "Open-source interpretability & observability for open-weight LLMs"' pyproject.toml \
  && echo "DESCRIPTION OK (WS0-owned)" \
  || echo "DESCRIPTION MISSING — land WS0 Task 18 (do NOT edit it here)"
```
If the description is wrong or missing, fix it in WS0 Task 18, not here.
- [ ] Step 3b: Edit the `[dependency-groups]` block. Replace:
```
[dependency-groups]
dev = [
    "pytest>=8.0",
]
```
with:
```
[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.13",
]
```
- [ ] Step 3c: Edit the `[tool.pytest.ini_options]` block to register the `gpu` marker. Replace:
```
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
addopts = "-q"
```
with:
```
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
addopts = "-q"
markers = [
    "gpu: tests that require model weights / a GPU (excluded from CI; run with --extra ml)",
]
```
- [ ] Step 3d: Append a `[tool.ruff]` block to the end of `pyproject.toml`.
```
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["backend"]

[tool.ruff.lint]
# Pyflakes (F) + pycodestyle errors (E) + isort (I). Conservative for an OSS baseline.
select = ["E", "F", "I"]
ignore = ["E501"]  # line length is enforced by the formatter, not the linter
```
- [ ] Step 4: Run the verifications; expect pass. First confirm the grep test, then confirm ruff installs + runs and pytest config parses.
```bash
/tmp/ws4_check_pyproject.sh
uv sync
uv run ruff check backend >/dev/null && echo "RUFF OK"
uv run pytest backend/tests --collect-only -q >/dev/null 2>&1 && echo "PYTEST CONFIG OK" || echo "PYTEST CONFIG OK (collection may need ml extra; no marker warning is the signal)"
```
Expected output:
```
OK: pyproject updated
RUFF OK
PYTEST CONFIG OK
```
(If `ruff check` reports lint findings in existing code, fix only trivial import-order/unused issues it flags, or extend `ignore` minimally; the goal is a green baseline. Re-run until `RUFF OK`.)
- [ ] Step 5: Commit.
```bash
git add pyproject.toml uv.lock
git commit -m "build: add ruff config + dev dep, register gpu marker (description owned by WS0)"
```

---

### Task 11: Add GitHub Actions CI (ruff lint + non-GPU pytest subset)

Files:
- Create: `.github/workflows/ci.yml`
- Test: local dry-run of the exact CI commands (ruff + the ignored-path pytest subset) + YAML parse

Interfaces:
- Consumes: the ruff config + dev dep + gpu marker from Task 10, the non-GPU test list (the 10 torch-importing files to `--ignore`), `astral-sh/setup-uv`.
- Produces: `.github/workflows/ci.yml`. CONTRIBUTING (Task 4) documents the same commands.

> HARD DEPENDENCY ON WS1 (sequence Task 11 AFTER WS1 lands): the non-GPU subset CI asserts green includes `test_api.py`, `test_events.py`, `test_analyze.py`, `test_fanout.py`, and `test_observability_endpoint.py`, all of which import symbols WS1 changes. CI is **only** green after WS1 has:
> - (a) updated `test_api.py` / `test_events.py` for the `CognitionEvent` → `IntrospectionEvent` and `build_cognition_event` → `build_introspection_event` renames (today `test_api.py` does `from backend.schema import CognitionEvent` and constructs it; `test_events.py` imports `build_cognition_event`);
> - (b) dropped the Phoenix assertions from `test_fanout.py` / `test_observability_endpoint.py`;
> - (c) deleted `backend/tests/test_coherence_eval.py` and `backend/tests/test_phoenix_eval_features.py`.
>
> This is a hard ordering constraint, not a "note it and coordinate" hedge: Task 11's dry-run (Step 1) verifies the renamed symbols exist (`IntrospectionEvent` / `build_introspection_event`) and that the Phoenix-only test files are gone BEFORE asserting the subset green, and fails loudly with a "WS1 has not landed" message if they are absent. The two Phoenix test files above are NOT in the `--ignore` list (they are non-torch); once WS1 deletes them they simply do not collect.
>
> The 10 ignored files are the tests that require the ML stack — torch directly, or transitively via numpy / `backend.science.persona` (e.g. `test_harmfulness_pipeline.py` imports numpy and `backend.validation.harmfulness_pipeline` → `backend.science.persona` at module level, and imports torch inside a function body): `test_agent_persist`, `test_attribution`, `test_fit`, `test_generate`, `test_gpu_service`, `test_harmfulness_pipeline`, `test_judge`, `test_loop`, `test_persona`, `test_prompt_tracker`.

- [ ] Step 1: Write the CI local dry-run test. This runs the EXACT commands CI will run, locally, so the workflow is proven before pushing.
```bash
cat > /tmp/ws4_check_ci.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
[ -f .github/workflows/ci.yml ] || { echo "MISSING FILE: .github/workflows/ci.yml"; exit 1; }
# YAML must parse
uv run python -c "import sys,yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" \
  && echo "YAML OK" || { echo "YAML PARSE FAILED"; fail=1; }
# Workflow must reference the lint + test steps
grep -q "ruff check backend" .github/workflows/ci.yml || { echo "NO RUFF STEP"; fail=1; }
grep -q "pytest backend/tests" .github/workflows/ci.yml || { echo "NO PYTEST STEP"; fail=1; }
grep -q "setup-uv" .github/workflows/ci.yml || { echo "NO setup-uv"; fail=1; }
# WS1-landed gate: the non-GPU subset includes tests that import WS1-renamed symbols
# (test_api.py / test_events.py) and Phoenix-only test files WS1 deletes. Assert WS1
# has landed before asserting the subset green, so a red CI is never blamed on WS4.
grep -q "class IntrospectionEvent" backend/schema.py \
  || { echo "WS1 NOT LANDED: backend/schema.py still lacks class IntrospectionEvent (rename pending)"; fail=1; }
grep -rq "build_introspection_event" backend/ \
  || { echo "WS1 NOT LANDED: build_introspection_event not found (build_cognition_event rename pending)"; fail=1; }
[ ! -f backend/tests/test_coherence_eval.py ] \
  || { echo "WS1 NOT LANDED: backend/tests/test_coherence_eval.py still present"; fail=1; }
[ ! -f backend/tests/test_phoenix_eval_features.py ] \
  || { echo "WS1 NOT LANDED: backend/tests/test_phoenix_eval_features.py still present"; fail=1; }
# The exact lint command CI runs must pass locally
uv run ruff check backend >/dev/null && echo "LINT OK" || { echo "LINT FAILED"; fail=1; }
# The exact non-GPU pytest subset CI runs must pass locally
uv run pytest backend/tests \
  --ignore=backend/tests/test_agent_persist.py \
  --ignore=backend/tests/test_attribution.py \
  --ignore=backend/tests/test_fit.py \
  --ignore=backend/tests/test_generate.py \
  --ignore=backend/tests/test_gpu_service.py \
  --ignore=backend/tests/test_harmfulness_pipeline.py \
  --ignore=backend/tests/test_judge.py \
  --ignore=backend/tests/test_loop.py \
  --ignore=backend/tests/test_persona.py \
  --ignore=backend/tests/test_prompt_tracker.py \
  >/dev/null && echo "NON-GPU TESTS OK" || { echo "NON-GPU TESTS FAILED"; fail=1; }
[ "$fail" -eq 0 ] && echo "OK: CI commands verified locally"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_ci.sh
```
- [ ] Step 2: Run the test; expect failure (workflow file absent).
```bash
/tmp/ws4_check_ci.sh
```
Expected output:
```
MISSING FILE: .github/workflows/ci.yml
```
- [ ] Step 3: Create `.github/workflows/ci.yml`.
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    name: lint + non-GPU tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.12"

      - name: Sync (light deps + dev tools; no ML extra)
        run: uv sync

      - name: Lint (ruff)
        run: uv run ruff check backend

      - name: Test (non-GPU subset; torch-dependent tests are GPU-gated)
        run: |
          uv run pytest backend/tests \
            --ignore=backend/tests/test_agent_persist.py \
            --ignore=backend/tests/test_attribution.py \
            --ignore=backend/tests/test_fit.py \
            --ignore=backend/tests/test_generate.py \
            --ignore=backend/tests/test_gpu_service.py \
            --ignore=backend/tests/test_harmfulness_pipeline.py \
            --ignore=backend/tests/test_judge.py \
            --ignore=backend/tests/test_loop.py \
            --ignore=backend/tests/test_persona.py \
            --ignore=backend/tests/test_prompt_tracker.py
```
- [ ] Step 4: Run the test; expect success (commands pass locally — note: torch IS installed in this dev env, so the non-GPU subset runs even though CI runs it without torch; the `--ignore` list still excludes the GPU-gated files).
```bash
/tmp/ws4_check_ci.sh
```
Expected output:
```
YAML OK
LINT OK
NON-GPU TESTS OK
OK: CI commands verified locally
```
(If the WS1-landed gate above fails — `IntrospectionEvent`/`build_introspection_event` missing, or a Phoenix test file still present — STOP: WS1 has not landed and Task 11 is out of sequence. Do not commit a workflow asserted against a pre-WS1 tree; resume Task 11 after WS1 merges. The WS4 deliverable is the workflow + the command shape, which is required to be green only on the post-WS1 tree the gate confirms.)
- [ ] Step 5: Commit.
```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for ruff lint and the non-GPU pytest subset"
```

---

### Task 12: Add GitHub PR + issue templates

Files:
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Test: shell grep assertions + YAML front-matter parse

Interfaces:
- Consumes: the CONTRIBUTING conventions (lint/tests/no-secrets), the SECURITY pointer.
- Produces: GitHub community templates. No other task depends on these.

- [ ] Step 1: Write the failing templates test.
```bash
cat > /tmp/ws4_check_templates.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
for f in .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/bug_report.md .github/ISSUE_TEMPLATE/feature_request.md; do
  [ -f "$f" ] || { echo "MISSING FILE: $f"; fail=1; }
done
grep -q "ruff" .github/PULL_REQUEST_TEMPLATE.md || { echo "PR template missing lint checkbox"; fail=1; }
grep -q "non-GPU" .github/PULL_REQUEST_TEMPLATE.md || { echo "PR template missing test checkbox"; fail=1; }
grep -qi "secret" .github/PULL_REQUEST_TEMPLATE.md || { echo "PR template missing secrets checkbox"; fail=1; }
# Issue templates need YAML front matter (name/about)
for f in .github/ISSUE_TEMPLATE/bug_report.md .github/ISSUE_TEMPLATE/feature_request.md; do
  head -1 "$f" | grep -q -- "---" || { echo "$f missing front matter"; fail=1; }
  grep -q "^name:" "$f" || { echo "$f missing name:"; fail=1; }
  grep -q "^about:" "$f" || { echo "$f missing about:"; fail=1; }
done
[ "$fail" -eq 0 ] && echo "OK: templates present"
exit "$fail"
EOF
chmod +x /tmp/ws4_check_templates.sh
```
- [ ] Step 2: Run the test; expect failure (files absent).
```bash
/tmp/ws4_check_templates.sh
```
Expected output:
```
MISSING FILE: .github/PULL_REQUEST_TEMPLATE.md
MISSING FILE: .github/ISSUE_TEMPLATE/bug_report.md
MISSING FILE: .github/ISSUE_TEMPLATE/feature_request.md
...
```
- [ ] Step 3a: Create `.github/PULL_REQUEST_TEMPLATE.md`.
```markdown
## Summary

<!-- What does this PR change and why? -->

## Checklist

- [ ] `uv run ruff check backend` passes.
- [ ] The non-GPU pytest subset passes (`uv run pytest backend/tests` with the GPU-gated files
      excluded — see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- [ ] No secrets are committed (nothing in `config.yaml`; secrets only in `.env` / env).
- [ ] `backend/app.py` still imports no torch (orchestration backend stays CPU-only).
- [ ] Docs/`CHANGELOG.md` updated if behavior or config changed.
- [ ] Commit messages follow Conventional Commits.

## Notes for reviewers

<!-- Anything that needs special attention, follow-ups, or related issues. -->
```
- [ ] Step 3b: Create `.github/ISSUE_TEMPLATE/bug_report.md`.
```markdown
---
name: Bug report
about: Report a problem with GlassBox
title: "[bug] "
labels: bug
---

## What happened

<!-- A clear description of the bug. -->

## Expected behavior

## Steps to reproduce

1.
2.
3.

## Environment

- GlassBox version / commit:
- Mode: fallback / real (output of `curl -s localhost:8000/api/health`):
- Installed extras: `uv sync` only / `uv sync --extra ml`:
- OS + Python version:

## Logs / output

<!-- Paste relevant logs. DO NOT paste secrets (API keys, POD_TOKEN, etc.). -->
```
- [ ] Step 3c: Create `.github/ISSUE_TEMPLATE/feature_request.md`.
```markdown
---
name: Feature request
about: Suggest an enhancement for GlassBox
title: "[feature] "
labels: enhancement
---

## Problem / motivation

<!-- What are you trying to do that GlassBox doesn't support today? -->

## Proposed solution

## Alternatives considered

## Additional context

<!-- Note if this overlaps an existing tracked issue under docs/issues/. -->
```
- [ ] Step 4: Run the test; expect success.
```bash
/tmp/ws4_check_templates.sh
```
Expected output:
```
OK: templates present
```
- [ ] Step 5: Commit.
```bash
git add .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE
git commit -m "chore: add GitHub PR and issue templates"
```

---

### Task 13: Final cross-reference sweep (no dangling links, no forbidden strings)

Files:
- Modify: none expected (verification task; fix only if a check fails)
- Test: shell assertions across all WS4 docs

Interfaces:
- Consumes: every file produced in Tasks 2-12.
- Produces: a verified, internally-consistent doc set ready to publish.

> This task closes WS4: it proves every relative link in the new docs resolves, and that no forbidden hackathon/Phoenix/cognition_event strings leaked into any new doc. If a check fails, fix the offending file (Edit), re-run, then commit under this task.

- [ ] Step 1: Write the final sweep test.
```bash
cat > /tmp/ws4_final_sweep.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fail=0
docs=(README.md ARCHITECTURE.md CONTRIBUTING.md SECURITY.md CHANGELOG.md docs/deployment.md
      docs/issues/01-arbitrary-hf-model-sae.md docs/issues/02-multi-provider-agent.md
      docs/issues/03-managed-gpu-endpoint.md docs/issues/04-serialize-probe-calibrator.md)
# 1. No forbidden strings in any new doc.
#    cognition_event is forbidden EVERYWHERE (the rename is complete).
#    Arize/Phoenix are forbidden in the user-facing docs, but CHANGELOG.md is
#    explicitly allowed to NAME the removed integration ("Arize Phoenix removed
#    entirely") — a changelog documenting a removal must say what was removed.
for d in "${docs[@]}"; do
  if grep -q -- "cognition_event" "$d"; then echo "FORBIDDEN 'cognition_event' in $d"; fail=1; fi
  if [ "$d" = "CHANGELOG.md" ]; then continue; fi   # CHANGELOG may name Arize/Phoenix as removed
  for s in "Arize" "Phoenix"; do
    if grep -q -- "$s" "$d"; then echo "FORBIDDEN '$s' in $d"; fail=1; fi
  done
done
# 2. Relative .md links resolve. Extract (path.md) targets and check existence.
for d in "${docs[@]}"; do
  base=$(dirname "$d")
  grep -oE '\]\(([^)]+\.md)\)' "$d" | sed -E 's/^\]\(//; s/\)$//' | while read -r link; do
    case "$link" in
      http*) continue ;;
    esac
    tgt="$base/$link"
    # normalize ./ and ../
    if [ ! -f "$(cd "$base" && python3 -c "import os,sys;print(os.path.normpath(sys.argv[1]))" "$link" 2>/dev/null)" ] && [ ! -f "$tgt" ]; then
      echo "DANGLING LINK in $d -> $link"
    fi
  done
done
# 3. No reference to deleted hackathon docs
for d in "${docs[@]}"; do
  if grep -qE "PLAN\.md|LANES\.md|superpowers" "$d"; then echo "REFS DELETED DOC in $d"; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "SWEEP-FLAGS-DONE"
exit "$fail"
EOF
chmod +x /tmp/ws4_final_sweep.sh
```
- [ ] Step 2: Run the sweep. Expect it to print any `DANGLING LINK` lines (subshell warnings) and then either pass or list failures.
```bash
/tmp/ws4_final_sweep.sh
```
Expected output (clean):
```
SWEEP-FLAGS-DONE
```
If `DANGLING LINK ...` or `FORBIDDEN ...` or `REFS DELETED DOC ...` lines appear, proceed to Step 3; otherwise skip to Step 5.
- [ ] Step 3: For each flagged line, open the file with Read and fix the offending link/string with Edit (e.g. correct a relative path, or remove a stray Phoenix mention). Repeat until clean.
- [ ] Step 4: Re-run the sweep; expect the clean output.
```bash
/tmp/ws4_final_sweep.sh
```
Expected output:
```
SWEEP-FLAGS-DONE
```
- [ ] Step 5: Verify the working tree against the file map: every WS4 file exists and the deleted ones are gone.
```bash
ls README.md ARCHITECTURE.md CONTRIBUTING.md SECURITY.md CHANGELOG.md \
   docs/deployment.md docs/issues/0*.md \
   .github/workflows/ci.yml .github/PULL_REQUEST_TEMPLATE.md \
   .github/ISSUE_TEMPLATE/bug_report.md .github/ISSUE_TEMPLATE/feature_request.md \
   && echo "ALL WS4 FILES PRESENT"
[ ! -e PLAN.md ] && [ ! -e LANES.md ] && [ ! -d docs/superpowers ] && echo "HACKATHON DOCS GONE"
```
Expected output (file listing, then):
```
ALL WS4 FILES PRESENT
HACKATHON DOCS GONE
```
- [ ] Step 6: Commit (empty commit if no fixes were needed, to mark the sweep as a discrete reviewable checkpoint; otherwise commit the fixes).
```bash
git add -A
git commit -m "docs: final cross-reference sweep for OSS doc set" --allow-empty
```

---

## Execution order

Tasks 2-12 are independent and may run in any order (each is its own test cycle), with ONE
hard cross-workstream constraint: **Task 11 (CI) must run only after WS1 has landed** — it asserts
the non-GPU pytest subset is green, and that subset includes `test_api.py`/`test_events.py` (which
import WS1-renamed symbols) and the Phoenix test files WS1 deletes. Task 11's Step-1 dry-run gates
on `IntrospectionEvent` / `build_introspection_event` existing and the two Phoenix test files being
gone, and stops if WS1 has not landed. Task 1 (remove hackathon docs) deletes `docs/superpowers/`,
which contains this plan file — run it **last** if executing the plan from disk (otherwise the
executor loses the plan mid-run). Task 13 is the closing verification and must run after Tasks 2-12.
Recommended order (with Task 11 deferred until WS1 is merged):
**2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 (after WS1) → 12 → 13 → 1.**
