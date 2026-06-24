# WS3 — Probe Pipeline OSS (BYO key + GPU self-host) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

Goal: Open-source the probe pipeline by threading a user-supplied Anthropic key through the existing client-injection points, genericizing the agent/judge medical prompts to a configurable domain, shipping a Dockerfile + docker-compose so users self-host the GPU service, hardening pod auth to fail closed, and removing the committed `glassbox-dev-secret` default from scripts and docs.

Architecture: The two-process split is unchanged — a CPU orchestration backend (`backend/app.py`, never imports torch) talks over HTTP to a GPU pod service (`backend/gpu_service.py`, owns torch + model + SAE + probes). Both processes build one `AppConfig` at startup and thread the narrowest sub-config into each module. WS3 touches only the pod-side agent/judge modules, the pod auth gate, the pod HTTP client, the GPU service Docker packaging, and the secret-default cleanup in scripts/docs.

Tech Stack: Python 3.12, FastAPI, pydantic v2 + pydantic-settings, httpx, anthropic SDK, torch/transformers/sae-lens (ML extra), Docker + docker-compose with the NVIDIA Container Toolkit, pytest.

## Global Constraints

- Python >= 3.12 (matches .python-version).
- Base deps (pyproject.toml): fastapi>=0.138.0, sentry-sdk[fastapi]>=2.63.0, httpx>=0.27.0.
- ML extra (uv sync --extra ml): torch>=2.4, transformers>=4.50, sae-lens>=6.0, scikit-learn>=1.5, accelerate>=0.34, anthropic>=0.40.
- pyproject.toml is the canonical dependency source; backend/requirements.txt is being retired.
- License: MIT.
- Naming/copy: KEEP the product name "GlassBox"; REMOVE medical-specific positioning (reposition as a general-purpose LLM interpretability/observability tool); medical survives only as ONE clearly-labeled optional example. Rename the per-turn event from cognition_event to introspection_event (class CognitionEvent -> IntrospectionEvent).
- Secrets (ANTHROPIC_API_KEY, SENTRY_DSN, POD_TOKEN, HF_TOKEN) load ONLY from env / .env, NEVER from config.yaml.
- COMMIT RULE (CRITICAL): commit messages MUST NOT add Claude as a co-author. No "Co-Authored-By: Claude" trailer anywhere. Use Conventional Commits style.
- Tests: keep the existing backend pytest suite green; CI runs the non-GPU subset; backend/engine.py and backend/science/sae.py are GPU-gated (untested without weights).
- Two processes: orchestration backend (backend/app.py, NEVER imports torch) and GPU pod service (backend/gpu_service.py, owns torch). AppConfig is built once per process and threaded via FastAPI app.state.

---

## File Structure

- `backend/agent/prompts.py` — **MODIFY**: drop module-level `config.AGENT_MAX_QUESTIONS` read; make `SYSTEM`, `SPEC_TOOL`, and `judge_*` take a `builder: ProbeBuilderConfig` + a neutral configurable `domain` string instead of hardcoded "medical-chat LLM".
- `backend/agent/tools.py` — **MODIFY**: drop module-level `config` reads; `dispatch`/`_dispatch`/`_finalize`/`_persist_artifact` take `builder: ProbeBuilderConfig` + `model: ModelConfig` via the per-job `ctx`.
- `backend/agent/interp_agent.py` — **MODIFY**: `run_interp_agent` takes `builder: ProbeBuilderConfig`, `anthropic_api_key: str`, and a `domain: str`; threads them into prompts/tools; keeps `client=`/`generate_fn=` injection points.
- `backend/science/concept_synth.py` — **MODIFY**: `judge_filter` takes `builder: ProbeBuilderConfig` + `anthropic_api_key: str`; keeps `client=` injection.
- `backend/gpu_service.py` — **MODIFY**: harden `_require_auth` to fail closed; thread `builder`/`anthropic_api_key`/`domain` into `_launch_agent` → `run_interp_agent`.
- `backend/pod_client.py` — **MODIFY** (shared with WS0): unchanged auth semantics, but documented that an empty `POD_TOKEN` now means "loopback-only pod".
- `Dockerfile` — **CREATE**: GPU image for `backend.gpu_service:app` (CUDA base, ML extras, model/SAE download cache).
- `docker-compose.yml` — **CREATE**: one `gpu` service reserving an NVIDIA GPU, env-file wired, port 8000.
- `.dockerignore` — **CREATE**: keep the build context small (no `.venv`, `frontend/node_modules`, caches, artifacts churn).
- `scripts/run_with_pod.sh` — **MODIFY**: remove `glassbox-dev-secret` default; leave `POD_TOKEN` unset by default.
- `scripts/pod_tailscale_bootstrap.sh` — **MODIFY**: remove `glassbox-dev-secret` default.
- `backend/batch_medqa_observability.py` — **MODIFY**: remove `glassbox-dev-secret` from the docstring example.
- `.env.example` — **MODIFY** (shared with WS0/WS1): remove the `glassbox-dev-secret` default; add a strong-token generation hint; add `GLASSBOX_DOMAIN`.
- `docs/tailscale-pod-runbook.md` — **MODIFY (body only)**: replace `glassbox-dev-secret` with a generate-your-own instruction. The header demotion to an optional appendix is owned by WS4 (`docs/deployment.md` is WS4's canonical deploy doc) — WS3 does NOT touch the header.
- `docs/probe-training.md` — **MODIFY**: replace `glassbox-dev-secret` in the deploy snippet.
- `docs/_snippets/docker-gpu-quickstart.md` — **CREATE**: a reusable Docker self-host quickstart section fragment (BYO key + token generation) that WS4's canonical `docs/deployment.md` includes. NOT a standalone canonical deploy doc (avoids competing with WS4's `docs/deployment.md`).
- `backend/tests/test_gpu_service.py` — **MODIFY**: update auth tests for fail-closed; add loopback-allow + non-loopback-reject tests.
- `backend/tests/test_agent_prompts.py` — **CREATE**: domain genericization + builder-threading tests for `prompts.py`.
- `backend/tests/test_interp_agent.py` — **CREATE**: `run_interp_agent` signature + key/domain threading test.
- `backend/tests/test_concept_synth_judge.py` — **CREATE**: `judge_filter` builder/key threading test.
- `backend/tests/test_secrets_cleanup.py` — **CREATE**: a guard test that fails if `glassbox-dev-secret` ever reappears in tracked scripts/docs.

> **DEPENDENCY ON WS0 (READ FIRST):** This plan consumes the `AppConfig` contract (`docs/superpowers/plans/2026-06-22-appconfig-contract.md`), which is **implemented by the concrete plan file `docs/superpowers/plans/2026-06-22-ws0-config-hygiene-rebrand.md` (WS0)** — that plan is the owner that lands `backend/config.py` with `AppConfig`, `ProbeBuilderConfig`, `ModelConfig`, the `load_config()` loader, `app.state.config` wiring, and adds `pydantic-settings>=2.0` + `pydantic>=2.9` + `pyyaml>=6.0` to `pyproject.toml` (WS0 Task 1). **WS3 cannot run Tasks 2–6 until WS0 has landed** — pre-WS0 the real `backend/config.py` is flat globals with NO `ProbeBuilderConfig`/`ModelConfig`, so any `from .config import ProbeBuilderConfig` ImportErrors and any image built without WS0's `pydantic-settings` fails at runtime. There is no "bridge" that makes these tasks runnable before WS0; the dependency is hard.
>
> **OVERLAP WITH WS0 (READ BEFORE EXECUTING TASKS 2–6):** WS0 Task 14 (`ws0-config-hygiene-rebrand.md`, lines 2155–2298) ALREADY performs the agent/judge refactor — it rewrites `agent/prompts.py`, `agent/tools.py`, `agent/interp_agent.py`, and `concept_synth.judge_filter` to the contract signatures, neutralizes the medical prompt text, and migrates `backend/tests/test_judge.py`, `backend/tests/test_loop.py`, and `backend/tests/test_agent_persist.py` to the new signatures. **WS3 Tasks 2–6 therefore EXTEND/RE-VERIFY rather than introduce these signatures.** When WS0 has already landed, an executor MUST re-read each module + the three migrated tests before editing, and treat the WS3 edits below as the delta over WS0's state (chiefly: the configurable `domain` seam, the `_resolve_domain` env hook, and the WS3-owned `test_agent_prompts.py`/`test_agent_tools.py`/`test_concept_synth_judge.py`/`test_interp_agent.py`). The three existing tests are migrated by WS0; if for any reason WS0's migration is incomplete when a WS3 task runs, that task's steps below re-assert the migration so the existing pytest suite stays green at the WS3 task boundary. Tasks 1 and 7–12 (auth hardening, Docker, secret cleanup, docs, guard test) do NOT depend on WS0 and may land first.

> **FILES SHARED WITH OTHER WORKSTREAMS:** `.env.example` (WS0 + WS1 also edit), `backend/pod_client.py` (WS0 rewrites its `config.` reads to `pod`/`pod_token`), `backend/gpu_service.py` (WS0 adds `app.state.config`; WS1 removes nothing here; WS2 derives `d_sae`). Where a task edits one of these, its Interfaces block flags it so execution can be sequenced. WS3's auth + agent-threading edits to `gpu_service.py` are localized to `_require_auth`, `_launch_agent`, and `/api/track`, minimizing conflict.

---

### Task 1: Harden pod auth to fail closed

Hardens `gpu_service._require_auth` so that a non-loopback bind with no `POD_TOKEN` set is rejected at request time (currently it returns/allows when the token is empty). Loopback binds (`127.0.0.1`/`::1`) still serve without a token so the local-only / SSH-tunnel workflow keeps working. This task does NOT depend on WS0 — it reads `config.POD_TOKEN` exactly as the code does today; WS0 later swaps that to `cfg.pod_token`.

Files:
- Modify: `backend/gpu_service.py` (lines 30-36 `_require_auth`)
- Test: `backend/tests/test_gpu_service.py` (lines 8-13 autouse fixture, lines 97-108 `test_auth_rejects_bad_token`)

Interfaces:
- Consumes: `config.POD_TOKEN` (str; today `backend/config.py:109`). SHARED FILE: WS0 will later rename this read to `request.app.state.config.pod_token` — keep the new `_require_auth` body small so that rename is a one-line follow-up.
- Consumes: `fastapi.Request` (already imported in `gpu_service.py:10`); the request's `request.client.host` and `request.url.hostname` for loopback detection.
- Produces: `_require_auth(request: Request) -> None` — raises `HTTPException(401)` on bad/missing token for any non-loopback request; raises `HTTPException(401, detail="POD_TOKEN required for non-loopback bind")` when token unset AND request is non-loopback; returns None (allows) when token set + header matches, OR when token unset + request is loopback.
- Produces: `_is_loopback(request: Request) -> bool` — True when the request arrived on a loopback interface.

Steps:

- [ ] Step 1: Replace the auth-test block in `backend/tests/test_gpu_service.py` (lines 97-108, `test_auth_rejects_bad_token`) with the fail-closed tests below. The TestClient connects from `testclient` (a non-loopback host string), so an unset token must now 401. The autouse fixture (lines 8-13) already clears `POD_TOKEN` to `""`; keep it. Write these tests:

```python
def test_auth_rejects_bad_token(monkeypatch):
    _install_stubs(monkeypatch)
    monkeypatch.setattr(gpu_service.config, "POD_TOKEN", "secret")
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": []})
    assert r.status_code == 401
    r = client.post(
        "/turn",
        json={"messages": []},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200


def test_auth_fails_closed_when_token_unset_and_not_loopback(monkeypatch):
    """No POD_TOKEN + a non-loopback request must be rejected (was: allowed)."""
    _install_stubs(monkeypatch)
    monkeypatch.setattr(gpu_service.config, "POD_TOKEN", "")
    # Force non-loopback: TestClient defaults to client host "testclient", which is
    # not 127.0.0.1, so _is_loopback() is False here.
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": []})
    assert r.status_code == 401
    assert "POD_TOKEN" in r.json()["detail"]


def test_auth_allows_loopback_without_token(monkeypatch):
    """No POD_TOKEN but a loopback request still serves (local/tunnel workflow)."""
    _install_stubs(monkeypatch)
    monkeypatch.setattr(gpu_service.config, "POD_TOKEN", "")
    monkeypatch.setattr(gpu_service, "_is_loopback", lambda request: True)
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": []})
    assert r.status_code == 200


def test_auth_token_set_still_required_on_loopback(monkeypatch):
    """When a token IS set, even loopback requests must present it."""
    _install_stubs(monkeypatch)
    monkeypatch.setattr(gpu_service.config, "POD_TOKEN", "secret")
    monkeypatch.setattr(gpu_service, "_is_loopback", lambda request: True)
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": []})
    assert r.status_code == 401
    r = client.post("/turn", json={"messages": []}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
```

- [ ] Step 2: Run the tests and confirm the new ones fail (current `_require_auth` allows when token is empty, and there is no `_is_loopback`).

```bash
uv run pytest backend/tests/test_gpu_service.py -q 2>&1 | tail -20
```

Expected: `test_auth_fails_closed_when_token_unset_and_not_loopback` FAILS with `assert 200 == 401`; `test_auth_allows_loopback_without_token` and `test_auth_token_set_still_required_on_loopback` FAIL with `AttributeError: <module 'backend.gpu_service'> does not have the attribute '_is_loopback'`.

- [ ] Step 3: Replace `_require_auth` in `backend/gpu_service.py` (lines 30-36) with the fail-closed implementation plus `_is_loopback`:

```python
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(request: Request) -> bool:
    """True when the request arrived over a loopback interface.

    Self-host runs behind a token; the local SSH-tunnel / single-box workflow binds the pod
    to 127.0.0.1 and may omit the token. We allow the no-token path ONLY for loopback so a
    public bind without a token can never serve unauthenticated.
    """
    client = request.client
    host = client.host if client is not None else None
    return host in _LOOPBACK_HOSTS


def _require_auth(request: Request) -> None:
    """Fail closed. A token, when set, is mandatory for every request. When no token is set,
    only loopback requests are served — a non-loopback request is rejected so a public bind
    can never run unauthenticated."""
    token = config.POD_TOKEN
    if not token:
        if _is_loopback(request):
            return
        raise HTTPException(
            status_code=401,
            detail="POD_TOKEN required for non-loopback bind — set POD_TOKEN on the pod",
        )
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="unauthorized")
```

- [ ] Step 4: Run the tests and confirm all pass.

```bash
uv run pytest backend/tests/test_gpu_service.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `10 passed` (the original 7 minus the replaced `test_auth_rejects_bad_token`, plus the 4 new auth tests = 10), no failures.

- [ ] Step 5: Commit.

```bash
git add backend/gpu_service.py backend/tests/test_gpu_service.py
git commit -m "feat(pod): harden _require_auth to fail closed on non-loopback binds"
```

---

### Task 2: Configurable domain string for the agent system prompt

Replaces the hardcoded "medical-chat LLM" / "medical-chat questions" phrasing in `backend/agent/prompts.py` with a neutral default that is parameterized by a `domain` string and a `ProbeBuilderConfig`. Removes the module-level `config.AGENT_MAX_QUESTIONS` read (it becomes a parameter). This is the genericization called out in the design doc §3 WS3 bullet and the AppConfig contract §3.2 (`agent/prompts` → "functions take `builder: ProbeBuilderConfig`").

Files:
- Create: `backend/tests/test_agent_prompts.py`
- Modify: `backend/agent/prompts.py` (lines 1-23 module header + `SYSTEM`; lines 36-39 `SPEC_TOOL.questions.description`)

Interfaces:
- Consumes (AppConfig contract): `ProbeBuilderConfig` from `backend.config` with field `agent_max_questions: int` (default 12). DEPENDS ON WS0 having landed `backend/config.py`.
- Produces: `DEFAULT_DOMAIN: str = "AI assistant"` (module constant — the neutral default).
- Produces: `system_prompt(builder: ProbeBuilderConfig, domain: str = DEFAULT_DOMAIN) -> str` — returns the agent system prompt with `{domain}` and `{builder.agent_max_questions}` interpolated. Replaces the module-level `SYSTEM` constant.
- Produces: `spec_tool(builder: ProbeBuilderConfig, domain: str = DEFAULT_DOMAIN) -> dict` — returns the `submit_spec` tool schema with the question cap + domain interpolated. Replaces the module-level `SPEC_TOOL` constant.
- Produces (unchanged signatures): `judge_schema(n: int) -> dict`, `judge_prompt(spec: dict, responses: list[str]) -> str`.

Steps:

- [ ] Step 1: Create `backend/tests/test_agent_prompts.py` with the full test:

```python
"""WS3: the agent prompts must be domain-neutral by default and accept a configurable
domain string + ProbeBuilderConfig instead of reading module-level config globals."""
from __future__ import annotations

from backend.agent import prompts
from backend.config import ProbeBuilderConfig


def test_default_domain_is_neutral_not_medical():
    builder = ProbeBuilderConfig()
    sys = prompts.system_prompt(builder)
    assert "medical" not in sys.lower()
    assert prompts.DEFAULT_DOMAIN.lower() != "medical-chat llm"
    # The neutral default domain must appear in the rendered prompt.
    assert prompts.DEFAULT_DOMAIN in sys


def test_system_prompt_interpolates_custom_domain():
    builder = ProbeBuilderConfig()
    sys = prompts.system_prompt(builder, domain="legal contract assistant")
    assert "legal contract assistant" in sys
    assert "medical-chat" not in sys


def test_system_prompt_uses_builder_question_cap():
    builder = ProbeBuilderConfig(agent_max_questions=5)
    sys = prompts.system_prompt(builder)
    # Assert on the cap-bearing phrase, not a bare "5": the prompt also has hardcoded step
    # numbers ("1." .. "5."), so a bare `"5" in sys` passes even if the cap is ignored. The
    # f-string interpolates the cap as "up to {n} questions", so check that exact phrase.
    assert "up to 5 questions" in sys
    # And confirm a different cap actually changes the rendered phrase.
    sys7 = prompts.system_prompt(ProbeBuilderConfig(agent_max_questions=7))
    assert "up to 7 questions" in sys7
    assert "up to 5 questions" not in sys7


def test_spec_tool_carries_domain_and_cap():
    builder = ProbeBuilderConfig(agent_max_questions=7)
    tool = prompts.spec_tool(builder, domain="customer-support assistant")
    assert tool["name"] == "submit_spec"
    desc = tool["input_schema"]["properties"]["questions"]["description"]
    assert "customer-support assistant" in desc
    assert "7" in desc
    assert "medical" not in desc.lower()


def test_judge_schema_unchanged():
    sch = prompts.judge_schema(3)
    assert sch["schema"]["properties"]["scores"]["minItems"] == 3
    assert sch["schema"]["properties"]["scores"]["maxItems"] == 3


def test_judge_prompt_numbers_responses():
    spec = {"trait_name": "sycophancy", "judge_rubric": "1-5 rubric"}
    out = prompts.judge_prompt(spec, ["aaa", "bbb"])
    assert "sycophancy" in out
    assert "0. aaa" in out and "1. bbb" in out
```

- [ ] Step 2: Run the test and confirm failure (no `system_prompt`/`spec_tool`/`DEFAULT_DOMAIN` yet).

```bash
uv run pytest backend/tests/test_agent_prompts.py -q 2>&1 | tail -20
```

Expected: `AttributeError: module 'backend.agent.prompts' has no attribute 'system_prompt'` (and `DEFAULT_DOMAIN`, `spec_tool`).

- [ ] Step 3: Rewrite `backend/agent/prompts.py` to the parameterized, domain-neutral form (replace the whole file):

```python
"""System prompt, the submit_spec tool schema, and judge templates for the
Interpretability Agent. Follows Persona Vectors (arXiv 2507.21509) generate_trait.

Domain-neutral by default: the agent monitors behaviors in a configurable target assistant
(`domain`), not a hardcoded medical chat model. The question cap comes from
ProbeBuilderConfig, threaded in by the caller instead of read from a module global."""
from __future__ import annotations

from ..config import ProbeBuilderConfig

DEFAULT_DOMAIN = "AI assistant"


def system_prompt(builder: ProbeBuilderConfig, domain: str = DEFAULT_DOMAIN) -> str:
    """The interpretability-agent system prompt, parameterized by the target `domain` and the
    builder's question cap. `domain` describes the assistant under inspection (e.g. "AI
    assistant", "legal contract assistant"); it is derived from the model's system prompt /
    a configured domain field by the caller."""
    n = builder.agent_max_questions
    return f"""You are an interpretability researcher replicating the Persona Vectors method.
Given a natural-language request to monitor a behavior in a {domain}, you:
1. Call submit_spec to define the trait: a crisp definition, a contrastive system-prompt
   pair (pos elicits the trait, neg suppresses it / behaves neutrally), up to {n} questions
   for the {domain} where the trait could surface, and a 1-5 judge rubric.
2. Call generate_contrastive to produce paired responses + activations.
3. Call judge_filter to keep only responses whose behavior matched the intended side.
4. Call fit_and_validate to train a probe and measure held-out AUROC vs a baseline.
5. Call finalize with a one- or two-sentence pipeline summary in plain language. Deployment is
   decided automatically by the AUROC gate; your summary should explain what the probe learned
   about the representation (separability, contrast quality, limitations). When the gate passes,
   affirm what the direction captures; when it fails, note what blocked deployment without
   dismissing the Persona Vectors approach — the metric outcome is separate from whether the
   concept is worth monitoring.
Call exactly one tool per step, in order. Do not skip steps."""


def spec_tool(builder: ProbeBuilderConfig, domain: str = DEFAULT_DOMAIN) -> dict:
    """The submit_spec tool schema, with the question cap + domain interpolated."""
    n = builder.agent_max_questions
    return {
        "name": "submit_spec",
        "description": "Define the contrastive trait spec (Persona Vectors generate_trait).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "trait_name": {"type": "string"},
                "definition": {"type": "string", "description": "What counts as the trait; what doesn't."},
                "pos_prompt": {"type": "string", "description": "System prompt that elicits the trait."},
                "neg_prompt": {"type": "string", "description": "System prompt that suppresses it / neutral."},
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Up to {n} questions for the {domain} (extra entries are truncated).",
                },
                "judge_rubric": {"type": "string", "description": "1-5 rubric for trait expression."},
            },
            "required": ["trait_name", "definition", "pos_prompt", "neg_prompt", "questions", "judge_rubric"],
            "additionalProperties": False,
        },
    }


def judge_schema(n: int) -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "minItems": n,
                    "maxItems": n,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }


def judge_prompt(spec: dict, responses: list[str]) -> str:
    numbered = "\n".join(f"{i}. {r[:800]}" for i, r in enumerate(responses))
    return (
        f"Trait: {spec['trait_name']}\nRubric: {spec['judge_rubric']}\n\n"
        f"Score how strongly each response expresses the trait (1-5). "
        f"Return exactly {len(responses)} scores in order.\n\nResponses:\n{numbered}"
    )
```

- [ ] Step 4: Run the test and confirm all pass.

```bash
uv run pytest backend/tests/test_agent_prompts.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `6 passed`, no failures.

- [ ] Step 5: Commit.

```bash
git add backend/agent/prompts.py backend/tests/test_agent_prompts.py
git commit -m "feat(agent): genericize agent prompts to a configurable domain (was medical-chat)"
```

---

### Task 3: Thread builder/model config through agent tools

Removes the module-level `config.AGENT_MAX_QUESTIONS`, `config.TRACK_AUROC_TAU`, and `config.LAYER` reads in `backend/agent/tools.py` and replaces them with values pulled from the per-job `ctx` dict (`ctx["builder"]`, `ctx["model"]`), per the AppConfig contract §3.2 (`agent/tools` → "take `builder: ProbeBuilderConfig` + `model: ModelConfig` as needed"). The `TOOLS` list, which referenced the module-level `prompts.SPEC_TOOL`, becomes a `tools_list(builder, domain)` builder since `SPEC_TOOL` is now a function (Task 2). DEPENDS ON Task 2 + WS0 (for `ProbeBuilderConfig`/`ModelConfig`).

> SIGNATURE NOTE (WS3 vs WS0): the contract row says `agent/tools` functions "take `builder` + `model` as needed". WS0 Task 14 threads `model` as an explicit parameter (`_persist_artifact(ctx, fit, model)`). WS3 instead reads `ctx["model"]` inside `_persist_artifact(ctx, fit)` and keeps the two-arg shape — this is the WS3-chosen form (the per-job `ctx` carries `builder`/`model`/`domain`, populated by `run_interp_agent` Task 5). When WS0 has landed, the executor re-reads `tools.py` and reconciles to the WS3 `ctx`-based form below (so `_persist_artifact` stays `(ctx, fit)` and reads `ctx["model"].layer`). Either way the existing `test_agent_persist.py` is migrated to put `'model': ModelConfig()` into its `ctx` (see Files + Step 5), because both forms reach `model.layer` only when a `ModelConfig` is present.

Files:
- Create: `backend/tests/test_agent_tools.py`
- Modify: `backend/agent/tools.py` (lines 1-21 imports + `TOOLS`; lines 36-43 `submit_spec` cap read; lines 62-70 `fit_and_validate` tau read; lines 76-94 `_finalize` tau read; lines 97-119 `_persist_artifact` layer read)
- Modify (existing-test migration): `backend/tests/test_agent_persist.py` (lines 45-47). The current `test_persist_artifact_round_trip_scores_graded` builds a `ctx` dict with NO `'model'` key and calls `tools._persist_artifact(ctx, fit)`. After this task `_persist_artifact` reads `ctx["model"].layer`, so that call raises `KeyError: 'model'`. This existing test MUST be migrated here (or by WS0 Task 14; if WS0 already migrated it, re-confirm the `ctx` has `'model'` and skip). FAILING-TEST-FIRST: Step 4 below runs `test_agent_persist.py` red against the old `ctx`, then green after adding `'model': ModelConfig()`.

Interfaces:
- Consumes (Task 2): `prompts.spec_tool(builder, domain) -> dict`.
- Consumes (AppConfig contract): `ProbeBuilderConfig` (fields `agent_max_questions: int`, `auroc_threshold: float`), `ModelConfig` (field `layer: int`). DEPENDS ON WS0.
- Consumes (existing): `ctx` dict keys `tracker_id`, `request`, `client`, `generate_fn`, `spec`, `rows`, `fit`. WS3 ADDS keys `builder: ProbeBuilderConfig`, `model: ModelConfig`, `domain: str` (populated by `run_interp_agent` in Task 5).
- Produces: `tools_list(builder: ProbeBuilderConfig, domain: str) -> list[dict]` — the five-tool list with the rendered `submit_spec` schema. Replaces the module-level `TOOLS`.
- Produces (signatures unchanged, behavior reads from ctx): `dispatch(name, tool_input, ctx) -> str`, `_dispatch(name, tool_input, ctx) -> str`, `_finalize(ctx, verdict) -> str`, `_persist_artifact(ctx, fit) -> None`, `_proj_calibration(rows, direction) -> tuple[float, float]`.

Steps:

- [ ] Step 1: Create `backend/tests/test_agent_tools.py` with the full test (uses fakes for `concept_synth` and `persona` so no torch/model is needed):

```python
"""WS3: agent tools read builder/model from ctx, not from config module globals."""
from __future__ import annotations

import backend.agent.tools as tools
from backend.config import ModelConfig, ProbeBuilderConfig


def test_tools_list_renders_spec_tool_with_domain():
    builder = ProbeBuilderConfig(agent_max_questions=4)
    lst = tools.tools_list(builder, domain="tax-advice assistant")
    names = [t["name"] for t in lst]
    assert names[0] == "submit_spec"
    assert {"generate_contrastive", "judge_filter", "fit_and_validate", "finalize"} <= set(names)
    desc = lst[0]["input_schema"]["properties"]["questions"]["description"]
    assert "tax-advice assistant" in desc and "4" in desc


def test_submit_spec_truncates_to_builder_cap(monkeypatch):
    updates = {}
    monkeypatch.setattr(tools.cs, "update_job", lambda tid, **kw: updates.update(kw))
    ctx = {
        "tracker_id": "t1",
        "builder": ProbeBuilderConfig(agent_max_questions=2),
        "model": ModelConfig(),
        "domain": "AI assistant",
        "spec": None,
    }
    tool_input = {
        "trait_name": "sycophancy",
        "questions": ["q1", "q2", "q3", "q4"],
        "definition": "d", "pos_prompt": "p", "neg_prompt": "n", "judge_rubric": "r",
    }
    out = tools.dispatch("submit_spec", tool_input, ctx)
    assert len(ctx["spec"]["questions"]) == 2
    assert "2 questions" in out


def test_fit_and_validate_reports_builder_tau(monkeypatch):
    monkeypatch.setattr(tools.cs, "update_job", lambda tid, **kw: None)
    monkeypatch.setattr(
        tools.cs, "fit_and_validate",
        lambda rows: {"status": "ok", "auroc": 0.9, "baseline_auroc": 0.6, "n_kept": 10},
    )
    ctx = {
        "tracker_id": "t1",
        "builder": ProbeBuilderConfig(auroc_threshold=0.8),
        "model": ModelConfig(),
        "rows": [{"label": 1}],
    }
    out = tools.dispatch("fit_and_validate", {}, ctx)
    assert "0.80" in out  # tau formatted from builder.auroc_threshold


def test_persist_artifact_uses_model_layer(monkeypatch, tmp_path):
    import json

    captured = {}

    class FakeDir:
        def __truediv__(self, name):
            captured["name"] = name
            return tmp_path / name

    monkeypatch.setattr(tools.persona, "ARTIFACT_DIR", FakeDir())
    monkeypatch.setattr(tools, "_proj_calibration", lambda rows, direction: (0.0, 1.0))

    import torch

    fit = {"direction": torch.tensor([1.0, 2.0]), "threshold": 0.5, "auroc": 0.9}
    ctx = {
        "tracker_id": "probe1",
        "request": "watch sycophancy",
        "spec": {"trait_name": "sycophancy"},
        "model": ModelConfig(layer=23),
        "rows": [],
    }
    tools._persist_artifact(ctx, fit)
    written = json.loads((tmp_path / "probe1.json").read_text())
    assert written["layer"] == 23
```

- [ ] Step 2: Run the test and confirm failure (no `tools_list`; `dispatch` still reads `config.*`).

```bash
uv run pytest backend/tests/test_agent_tools.py -q 2>&1 | tail -20
```

Expected: `AttributeError: module 'backend.agent.tools' has no attribute 'tools_list'`.

- [ ] Step 3: Rewrite `backend/agent/tools.py` to read from `ctx` (replace the whole file):

```python
"""The five tools the Interpretability Agent calls, plus a dispatch() that executes
them against a per-job context dict and keeps the job record in sync.

The per-job `ctx` carries the config the tools need: ctx["builder"] (ProbeBuilderConfig),
ctx["model"] (ModelConfig), ctx["domain"] (str). No module-level config globals are read."""
from __future__ import annotations

from ..science import concept_synth as cs
from ..science import persona
from . import prompts


def tools_list(builder, domain: str) -> list[dict]:
    """The five-tool schema list with submit_spec rendered for the builder's question cap
    + the target domain."""
    return [
        prompts.spec_tool(builder, domain),
        {"name": "generate_contrastive", "description": "Run the model under the pos/neg prompts; capture activations.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "judge_filter", "description": "Keep only responses whose behavior matched the intended side.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "fit_and_validate", "description": "Train the probe; measure held-out AUROC vs a baseline.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "finalize", "description": "Record the plain-language pipeline summary; deployment is decided by the AUROC gate.",
         "input_schema": {"type": "object", "properties": {"verdict": {"type": "string"}},
                          "required": ["verdict"], "additionalProperties": False}},
    ]


def dispatch(name: str, tool_input: dict, ctx: dict) -> str:
    tid = ctx["tracker_id"]
    try:
        return _dispatch(name, tool_input, ctx)
    except Exception as e:  # noqa: BLE001 - surface in job record + agent tool_result
        msg = str(e) or type(e).__name__
        cs.update_job(tid, status="error", error=f"{name}: {msg}")
        return f"ERROR: {msg}"


def _dispatch(name: str, tool_input: dict, ctx: dict) -> str:
    tid = ctx["tracker_id"]
    builder = ctx["builder"]
    if name == "submit_spec":
        ctx["spec"] = tool_input
        cap = builder.agent_max_questions
        if cap:
            ctx["spec"]["questions"] = ctx["spec"]["questions"][:cap]
        cs.update_job(tid, status="designing", trait_name=tool_input["trait_name"],
                      progress={"step": "spec", "pct": 20})
        return f"Spec stored: {len(ctx['spec']['questions'])} questions. Call generate_contrastive."
    if name == "generate_contrastive":
        cs.update_job(tid, status="generating", progress={"step": "generating", "pct": 40})

        def on_progress(done: int, total: int) -> None:
            pct = 40 + int(15 * done / max(total, 1))
            cs.update_job(tid, progress={"step": "generating", "pct": min(pct, 59)})

        ctx["rows"] = cs.generate_contrastive(
            ctx["spec"], generate_fn=ctx["generate_fn"], on_progress=on_progress
        )
        return f"Generated {len(ctx['rows'])} responses. Call judge_filter."
    if name == "judge_filter":
        cs.update_job(tid, status="judging", progress={"step": "judging", "pct": 60})
        ctx["rows"] = cs.judge_filter(
            ctx["spec"], ctx["rows"], builder, ctx["anthropic_api_key"], client=ctx["client"]
        )
        cs.update_job(tid, n_kept=len(ctx["rows"]))
        if not ctx["rows"]:
            return "Kept 0 clean rows after judge — call finalize explaining insufficient contrast."
        return f"Kept {len(ctx['rows'])} clean rows. Call fit_and_validate."
    if name == "fit_and_validate":
        cs.update_job(tid, status="fitting", progress={"step": "fitting", "pct": 80})
        ctx["fit"] = cs.fit_and_validate(ctx["rows"])
        f = ctx["fit"]
        cs.update_job(tid, auroc=f["auroc"], baseline_auroc=f["baseline_auroc"], n_kept=f["n_kept"])
        if f["status"] != "ok":
            return "insufficient_data: too few clean rows or a class collapsed. Call finalize explaining this."
        return (f"held-out AUROC={f['auroc']:.2f} (baseline {f['baseline_auroc']:.2f}), "
                f"tau={builder.auroc_threshold:.2f}. Call finalize with your verdict.")
    if name == "finalize":
        return _finalize(ctx, tool_input["verdict"])
    return f"unknown tool: {name}"


def _finalize(ctx: dict, verdict: str) -> str:
    tid = ctx["tracker_id"]
    builder = ctx["builder"]
    fit = ctx.get("fit") or {}
    deployed = fit.get("status") == "ok" and (fit.get("auroc") or 0.0) >= builder.auroc_threshold
    if deployed:
        persona._trackers[tid] = {
            "dir": fit["direction"],
            "calibrator": fit["calibrator"],
            "threshold": fit["threshold"],
            "meta": {"user_defined": True, "auroc": fit["auroc"],
                     "request": ctx["request"], "reliability": "synthetic-validated"},
        }
        try:
            _persist_artifact(ctx, fit)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] artifact persist failed for {tid}: {e}")
    cs.update_job(tid, status="ready" if deployed else "rejected", verdict=verdict,
                  progress={"step": "done", "pct": 100})
    return "deployed as a live guardrail." if deployed else "not deployed (gate not met)."


def _persist_artifact(ctx: dict, fit: dict) -> None:
    """Serialize a deployed probe to backend/science/artifacts/<id>.json."""
    import json

    tid = ctx["tracker_id"]
    spec = ctx.get("spec") or {}
    direction = fit["direction"].detach().cpu()
    center, scale = _proj_calibration(ctx["rows"], direction)
    artifact = {
        "id": tid,
        "concept": spec.get("trait_name") or tid,
        "description": ctx["request"],
        "direction": direction.tolist(),
        "threshold": float(fit["threshold"]),
        "projection_center": center,
        "projection_scale": scale,
        "alert_direction": "high",
        "direction_method": "diff_of_means",
        "layer": ctx["model"].layer,
        "auroc": fit.get("auroc"),
        "user_defined": True,
    }
    (persona.ARTIFACT_DIR / f"{tid}.json").write_text(json.dumps(artifact, indent=2))


def _proj_calibration(rows: list[dict], direction) -> tuple[float, float]:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

    X = torch.stack([r["act_resp"].float().cpu() for r in rows])
    proj = (X @ direction.float()).numpy().reshape(-1, 1)
    y = np.asarray([int(r["label"]) for r in rows], dtype="int64")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(proj, y)
    w = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    if w > 1e-8:
        return -b / w, 1.0 / w
    return float(proj.mean()), float(proj.std()) or 1.0
```

- [ ] Step 4: Run the test and confirm all pass.

```bash
uv run pytest backend/tests/test_agent_tools.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `4 passed`, no failures.

- [ ] Step 5: Migrate the existing `backend/tests/test_agent_persist.py` so it still passes under the `ctx["model"]` read. First confirm it now breaks (run before editing): `_persist_artifact(ctx, fit)` reads `ctx["model"].layer`, but the test's `ctx` has no `'model'` key, so it raises `KeyError: 'model'`. (If WS0 Task 14 already migrated this file, it will already pass — verify the `ctx` has `'model'` and skip the edit.)

```bash
uv run pytest backend/tests/test_agent_persist.py -q 2>&1 | tail -20
```

Expected (pre-migration, if not already migrated by WS0): `KeyError: 'model'` in `test_persist_artifact_round_trip_scores_graded`.

Then edit `backend/tests/test_agent_persist.py`:

Add the config import at the top (with the other imports):

```python
from backend.config import ModelConfig
```

Add `'model': ModelConfig()` to the `ctx` dict (lines 45-46) so `_persist_artifact` can read `ctx["model"].layer`:

```python
    ctx = {"tracker_id": "over-conf-test", "request": "flag over-confidence",
           "spec": {"trait_name": "over_confidence"}, "model": ModelConfig(), "rows": rows}
```

Then assert the written artifact carries the model's layer (proves the layer is sourced from `ctx["model"]`, not a config global) — add after the existing `assert len(data["direction"]) == 8`:

```python
    assert data["layer"] == ModelConfig().layer
```

- [ ] Step 6: Re-run the migrated existing test plus the new test; confirm both pass.

```bash
uv run pytest backend/tests/test_agent_tools.py backend/tests/test_agent_persist.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `6 passed` (4 new + the 2 in the migrated `test_agent_persist.py`), no failures.

- [ ] Step 7: Commit.

```bash
git add backend/agent/tools.py backend/tests/test_agent_tools.py backend/tests/test_agent_persist.py
git commit -m "refactor(agent): read builder/model from ctx instead of config globals in tools"
```

---

### Task 4: Thread the Anthropic key + builder through judge_filter

Updates `backend/science/concept_synth.judge_filter` to take `builder: ProbeBuilderConfig` + `anthropic_api_key: str` as explicit parameters (keeping the `client=` injection point), per the AppConfig contract §3.2 (`science/concept_synth.judge_filter` → `judge_filter(spec, rows, builder, anthropic_api_key, *, client=None)`). Removes the module-level `config.ANTHROPIC_API_KEY`, `config.JUDGE_MODEL`, `config.JUDGE_BATCH_SIZE` reads in that function. DEPENDS ON WS0 (`ProbeBuilderConfig`).

Files:
- Create: `backend/tests/test_concept_synth_judge.py`
- Modify: `backend/science/concept_synth.py` (lines 9 `from .. import config`; lines 208-228 `judge_filter`)
- Modify (existing-test migration): `backend/tests/test_judge.py` (line 30). The current `test_judge.py:30` calls `cs.judge_filter(SPEC, _rows(), client=FakeClient([5, 2, 1, 4]))` with only `spec` + `rows`. After this task `builder` + `anthropic_api_key` are REQUIRED positionals, so that call raises `TypeError: judge_filter() missing 2 required positional arguments`. This existing test MUST be migrated here (or by WS0 Task 14; if WS0 already migrated it, re-confirm the call matches and skip). FAILING-TEST-FIRST: Step 2 below runs `test_judge.py` red against the old call before the migration, then green after — the existing test is migrated, not merely supplemented by the new `test_concept_synth_judge.py`.

Interfaces:
- Consumes (AppConfig contract): `ProbeBuilderConfig` (fields `judge_model: str`, `judge_batch_size: int`); `anthropic_api_key: str` scalar. DEPENDS ON WS0.
- Consumes (unchanged): `_judge_batch(spec, batch, *, client, model) -> list[int]`; `prompts.judge_schema`, `prompts.judge_prompt`.
- Produces: `judge_filter(spec: dict, rows: list[dict], builder: ProbeBuilderConfig, anthropic_api_key: str, *, client=None) -> list[dict]`. Caller in `agent/tools.py` (Task 3) already passes `(ctx["spec"], ctx["rows"], builder, ctx["anthropic_api_key"], client=ctx["client"])`.
- NOTE: `concept_synth.py` line 9 `from .. import config` may still be used by other functions in the file (`fit_and_validate`, `generate_contrastive` do NOT read config; verify the import is still needed elsewhere — if `judge_filter` was the only consumer, drop the import; WS0's broader refactor will have removed it. If WS0 already removed the import, skip that edit and keep only the signature change).

Steps:

- [ ] Step 1: Create `backend/tests/test_concept_synth_judge.py` with the full test (a fake Anthropic client, no network):

```python
"""WS3: judge_filter takes builder + anthropic_api_key explicitly, keeps client= injection."""
from __future__ import annotations

import json

from backend.config import ProbeBuilderConfig
from backend.science import concept_synth as cs


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResp:
    def __init__(self, scores):
        self.content = [_FakeBlock(json.dumps({"scores": scores}))]


class _FakeMessages:
    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp(self._scores)


class _FakeClient:
    def __init__(self, scores):
        self.messages = _FakeMessages(scores)


def test_judge_filter_keeps_clean_rows_with_injected_client():
    rows = [
        {"response": "very positive", "intended_label": 1},
        {"response": "very negative", "intended_label": 0},
    ]
    client = _FakeClient([5, 1])  # pos scores 5 (>=4 kept), neg scores 1 (<=2 kept)
    # Sentinel judge_model that is NOT the ProbeBuilderConfig default, so the model assertion
    # below actually proves the builder value (not a config global / default) drove the call.
    builder = ProbeBuilderConfig(judge_model="sentinel-judge-model", judge_batch_size=8)
    assert ProbeBuilderConfig().judge_model != "sentinel-judge-model"  # guard the sentinel
    spec = {"trait_name": "t", "judge_rubric": "r"}
    kept = cs.judge_filter(spec, rows, builder, "sk-fake", client=client)
    assert len(kept) == 2
    assert kept[0]["label"] == 1 and kept[1]["label"] == 0
    # builder.judge_model was used, not a config global
    assert client.messages.calls[0]["model"] == "sentinel-judge-model"


def test_judge_filter_drops_ambiguous():
    rows = [
        {"response": "meh", "intended_label": 1},   # score 3 -> dropped (not >=4)
        {"response": "ok", "intended_label": 0},    # score 3 -> dropped (not <=2)
    ]
    client = _FakeClient([3, 3])
    builder = ProbeBuilderConfig()
    spec = {"trait_name": "t", "judge_rubric": "r"}
    kept = cs.judge_filter(spec, rows, builder, "sk-fake", client=client)
    assert kept == []
```

- [ ] Step 2: Run the test and confirm failure (current `judge_filter` signature is `(spec, rows, *, client=None, judge_model=None)`, so the positional `builder` + `anthropic_api_key` raise `TypeError`).

```bash
uv run pytest backend/tests/test_concept_synth_judge.py -q 2>&1 | tail -20
```

Expected: `TypeError: judge_filter() takes 2 positional arguments but 4 were given`.

- [ ] Step 3: Edit `backend/science/concept_synth.py` `judge_filter` (lines 208-228) to the new signature:

```python
def judge_filter(
    spec: dict,
    rows: list[dict],
    builder,
    anthropic_api_key: str,
    *,
    client=None,
) -> list[dict]:
    """Score each response 1-5; keep unambiguous positives (>=4) and negatives (<=2).

    `builder` is a ProbeBuilderConfig (judge_model + judge_batch_size). `anthropic_api_key`
    is used only to construct the default client; pass `client=` to inject a pre-built one
    (the agent injects the pod-shared client so this never re-reads a config global)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_api_key)
    model = builder.judge_model
    batch_size = max(1, builder.judge_batch_size)

    all_scores: list[int] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        all_scores.extend(_judge_batch(spec, batch, client=client, model=model))

    kept: list[dict] = []
    for row, score in zip(rows, all_scores):
        if row["intended_label"] == 1 and score >= 4:
            kept.append({**row, "label": 1})
        elif row["intended_label"] == 0 and score <= 2:
            kept.append({**row, "label": 0})
    return kept
```

- [ ] Step 4: Run the new test and confirm all pass.

```bash
uv run pytest backend/tests/test_concept_synth_judge.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `2 passed`, no failures.

- [ ] Step 5: Migrate the existing `backend/tests/test_judge.py` to the new signature. First confirm it now breaks (run before editing): the call `cs.judge_filter(SPEC, _rows(), client=FakeClient([5, 2, 1, 4]))` raises `TypeError` for the missing required `builder` + `anthropic_api_key`. (If WS0 Task 14 already migrated this file, it will already pass — verify and skip the edit.)

```bash
uv run pytest backend/tests/test_judge.py -q 2>&1 | tail -20
```

Expected (pre-migration, if not already migrated by WS0): `TypeError: judge_filter() missing 2 required positional arguments: 'builder' and 'anthropic_api_key'`.

Then edit `backend/tests/test_judge.py`:

Add the config import at the top (with `from backend.science import concept_synth as cs`):

```python
from backend.config import ProbeBuilderConfig
```

Change the `judge_filter` call (line 30) to pass the now-required positionals (the canned `FakeClient` already bypasses the network, so `judge_model`/`batch_size` come from the default `ProbeBuilderConfig()`):

```python
    rows = cs.judge_filter(
        SPEC, _rows(), ProbeBuilderConfig(), "sk-fake", client=FakeClient([5, 2, 1, 4])
    )
```

- [ ] Step 6: Re-run the migrated existing test plus the new test; confirm both pass.

```bash
uv run pytest backend/tests/test_concept_synth_judge.py backend/tests/test_judge.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `3 passed` (2 new + the 1 migrated judge test), no failures.

- [ ] Step 7: Commit.

```bash
git add backend/science/concept_synth.py backend/tests/test_concept_synth_judge.py backend/tests/test_judge.py
git commit -m "refactor(science): judge_filter takes builder + anthropic_api_key explicitly"
```

---

### Task 5: Thread builder/key/domain through run_interp_agent

Updates `backend/agent/interp_agent.run_interp_agent` to take `builder: ProbeBuilderConfig`, `anthropic_api_key: str`, and `domain: str`, per the AppConfig contract §3.2 (`agent/interp_agent.run_interp_agent` → `run_interp_agent(tracker_id, builder, anthropic_api_key, *, client=None, generate_fn=None)`). Threads them into the system prompt (Task 2), the tools list (Task 3), and the per-job `ctx` (so `tools.dispatch` and `judge_filter` get them). DEPENDS ON Tasks 2, 3, 4 and WS0.

Files:
- Create: `backend/tests/test_interp_agent.py`
- Modify: `backend/agent/interp_agent.py` (lines 1-59 — whole module)
- Modify (existing-test migration): `backend/tests/test_loop.py` (line 33-34 `judge_filter` monkeypatch lambda; line 47 `run_interp_agent` call). The current `test_loop.py:47` calls `interp_agent.run_interp_agent(tid, client=client, generate_fn=_fake_generate)` with only `tracker_id` — after this task that raises `TypeError` (missing required `builder` + `anthropic_api_key`). And `test_loop.py:33-34` monkeypatches `cs.judge_filter` with `lambda spec, rows, **kw: ...`, which accepts only 2 positionals — after Task 3/4 the agent calls `cs.judge_filter(ctx['spec'], ctx['rows'], builder, ctx['anthropic_api_key'], client=...)` (4 positionals) → `TypeError`. Both MUST be migrated here (or by WS0 Task 14; if WS0 already migrated them, re-confirm they match the WS3 keyword-only signature and skip). FAILING-TEST-FIRST: the existing test is *migrated*, not merely left alone — Step 4 below runs it red against the old call, then green after the migration.

Interfaces:
- Consumes (Task 2): `prompts.system_prompt(builder, domain) -> str`, `prompts.DEFAULT_DOMAIN`.
- Consumes (Task 3): `tools.tools_list(builder, domain) -> list[dict]`, `tools.dispatch(name, input, ctx) -> str`.
- Consumes (AppConfig contract): `ProbeBuilderConfig` (field `agent_model: str`); `anthropic_api_key: str`. DEPENDS ON WS0.
- Produces: `run_interp_agent(tracker_id: str, builder: ProbeBuilderConfig, anthropic_api_key: str, *, domain: str = prompts.DEFAULT_DOMAIN, model: ModelConfig | None = None, client=None, generate_fn=None) -> dict | None`. The pod caller (`gpu_service._launch_agent`, Task 6) supplies `builder`, `anthropic_api_key`, `domain`, and `model` from `app.state.config`.
- CONTRACT NOTE (no unauthorized deviation): the AppConfig contract §3.2 fixes the **positional/required** surface as `run_interp_agent(tracker_id, builder, anthropic_api_key, *, client=None, generate_fn=None)`. WS3 preserves that surface exactly — `tracker_id`, `builder`, `anthropic_api_key` stay the three required positionals and `client`/`generate_fn` stay keyword-only — and ADDS two **keyword-only, optional** params, `domain` and `model`, after the `*`. Adding keyword-only optional params is backward-compatible and does NOT change any caller that uses the contract signature, so it is not a second deviation from the contract's "every other path is fixed" rule (which constrains the positional/required surface, the one the contract row pins). The contract §3.2 row + §6 have been amended in `2026-06-22-appconfig-contract.md` to record `domain`/`model` as authorized keyword-only additions (see that file). Rationale: `tools._persist_artifact` genuinely needs `model.layer` (Task 3) and `judge_filter` needs `anthropic_api_key`, and the pod (which holds the full `AppConfig`) is the natural place to inject both. `run_interp_agent` puts `model` (defaulting to `ModelConfig()`) and `anthropic_api_key` into the per-job `ctx` so the tools read them from there.
- ALIGNMENT WITH WS0: WS0 Task 14 threads `model` to `_persist_artifact` as an explicit argument and keeps `run_interp_agent` at the contract signature. WS3's keyword-only `model` param is the WS3 superset that ALSO carries the `domain` seam; when WS0 has landed, reconcile to this single keyword-only-`model`/`domain` form (the executor re-reads `interp_agent.py` first and replaces WS0's body with the version below, keeping WS0's contract positionals intact).

Steps:

- [ ] Step 1: Create `backend/tests/test_interp_agent.py` with the full test (a fake Claude client that immediately stops, so no tools run; verifies signature + threading):

```python
"""WS3: run_interp_agent takes builder + anthropic_api_key + domain (+ model) and threads
them into the system prompt, tools, and ctx. No real Claude / torch."""
from __future__ import annotations

import backend.agent.interp_agent as ia
from backend.config import ModelConfig, ProbeBuilderConfig
from backend.science import concept_synth as cs


class _StopResp:
    stop_reason = "end_turn"
    content = []


class _Messages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StopResp()


class _Client:
    def __init__(self):
        self.messages = _Messages()


def test_run_interp_agent_threads_builder_and_domain(monkeypatch):
    tid = cs.create_job("watch sycophancy")
    client = _Client()
    # Use a sentinel model id that is NOT the ProbeBuilderConfig default, so the assertion
    # below fails if the builder value is ignored (i.e. it actually proves threading).
    builder = ProbeBuilderConfig(agent_model="sentinel-agent-model")
    assert ProbeBuilderConfig().agent_model != "sentinel-agent-model"  # guard the sentinel
    out = ia.run_interp_agent(
        tid, builder, "sk-fake", domain="legal assistant",
        model=ModelConfig(), client=client,
    )  # domain/model/client are keyword-only per the WS3 signature
    # The system prompt sent to Claude must carry the configured domain, not "medical".
    sys = client.messages.calls[0]["system"]
    assert "legal assistant" in sys
    assert "medical" not in sys.lower()
    # The agent_model from the builder must drive the request (sentinel, not the default).
    assert client.messages.calls[0]["model"] == "sentinel-agent-model"
    assert out is not None


def test_run_interp_agent_requires_known_tracker():
    import pytest

    with pytest.raises(ValueError, match="unknown tracker_id"):
        ia.run_interp_agent(
            "nope", ProbeBuilderConfig(), "sk-fake",
            domain="AI assistant", model=ModelConfig(), client=_Client(),
        )
```

- [ ] Step 2: Run the test and confirm failure (current signature is `run_interp_agent(tracker_id, *, client=None, generate_fn=None)`).

```bash
uv run pytest backend/tests/test_interp_agent.py -q 2>&1 | tail -20
```

Expected: `TypeError: run_interp_agent() takes 1 positional argument but 3 positional arguments (and 2 keyword-only) were given` (or similar arity error).

- [ ] Step 3: Rewrite `backend/agent/interp_agent.py` (replace the whole file):

```python
"""The Interpretability Agent's Claude tool-use loop. Blocking (gemma + sync Claude);
run via asyncio.to_thread so it never blocks the FastAPI event loop.

WS3: config is threaded in (builder/model/anthropic_api_key/domain), not read from globals,
so a self-host user's own key + domain drive the run. The client= / generate_fn= injection
points are preserved for a future per-request / multi-provider path."""
from __future__ import annotations

from ..config import ModelConfig, ProbeBuilderConfig
from ..science import concept_synth as cs
from . import prompts, tools

_MAX_TURNS = 12


def run_interp_agent(
    tracker_id: str,
    builder: ProbeBuilderConfig,
    anthropic_api_key: str,
    *,
    domain: str = prompts.DEFAULT_DOMAIN,
    model: ModelConfig | None = None,
    client=None,
    generate_fn=None,
) -> dict | None:
    job = cs.get_job(tracker_id)
    if job is None:
        raise ValueError(f"unknown tracker_id: {tracker_id}")
    if model is None:
        model = ModelConfig()
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_api_key)

    # generate_fn may be None; the None→engine default happens inside
    # concept_synth.generate_contrastive, keeping agent/ free of any torch/engine import.
    ctx = {"tracker_id": tracker_id, "request": job["request"],
           "client": client, "generate_fn": generate_fn,
           "builder": builder, "model": model, "domain": domain,
           "anthropic_api_key": anthropic_api_key,
           "spec": None, "rows": None, "fit": None}
    system = prompts.system_prompt(builder, domain)
    tool_schemas = tools.tools_list(builder, domain)
    messages = [{"role": "user", "content": f"Monitor request: {job['request']}"}]

    for _ in range(_MAX_TURNS):
        resp = client.messages.create(
            model=builder.agent_model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            tools=tool_schemas,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = tools.dispatch(block.name, block.input, ctx)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        if not results:
            break
        messages.append({"role": "user", "content": results})
        if cs.get_job(tracker_id)["status"] in ("ready", "rejected"):
            break

    job = cs.get_job(tracker_id)
    if job is not None and job["status"] not in ("ready", "rejected", "error"):
        step = (job.get("progress") or {}).get("step", job["status"])
        cs.update_job(
            tracker_id,
            status="error",
            error=f"agent did not finalize within MAX_TURNS (last step: {step})",
        )
    return cs.get_job(tracker_id)
```

- [ ] Step 4: Run the test and confirm all pass.

```bash
uv run pytest backend/tests/test_interp_agent.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `2 passed`, no failures.

- [ ] Step 5: Migrate the existing `backend/tests/test_loop.py` so the integration loop test still runs under the new signatures. First confirm it breaks (run it before editing): the `run_interp_agent(tid, client=..., generate_fn=...)` call now raises `TypeError` for the missing required `builder`/`anthropic_api_key`, and once the agent reaches `judge_filter` the 2-arg monkeypatch lambda raises `TypeError` for the extra positionals. (If WS0 Task 14 already migrated this file, it will already pass — verify the call matches the WS3 keyword-only form below; if it does, skip the edit.)

```bash
uv run pytest backend/tests/test_loop.py -q 2>&1 | tail -20
```

Expected (pre-migration, if not already migrated by WS0): `TypeError: run_interp_agent() missing 2 required positional arguments: 'builder' and 'anthropic_api_key'`.

Then edit `backend/tests/test_loop.py`:

Add the config import at the top (with the other imports):

```python
from backend.config import ModelConfig, ProbeBuilderConfig
```

Change the `judge_filter` monkeypatch (lines 33-34) so the lambda accepts the new positionals `builder` + `anthropic_api_key`:

```python
    # judge_filter now takes (spec, rows, builder, anthropic_api_key, *, client=None);
    # accept the new positionals so the agent's 4-arg call doesn't TypeError.
    monkeypatch.setattr(
        cs, "judge_filter",
        lambda spec, rows, builder=None, anthropic_api_key=None, **kw: [
            {**r, "label": r["intended_label"]} for r in rows
        ],
    )
```

Change the `run_interp_agent` call (line 47) to pass the now-required `builder` + `anthropic_api_key` (and the keyword-only `model`):

```python
    job = interp_agent.run_interp_agent(
        tid, ProbeBuilderConfig(), "sk-fake",
        model=ModelConfig(), client=client, generate_fn=_fake_generate,
    )
```

- [ ] Step 6: Re-run the migrated existing test and the new test together; confirm both pass.

```bash
uv run pytest backend/tests/test_interp_agent.py backend/tests/test_loop.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `3 passed` (2 new + the 1 migrated loop test), no failures.

- [ ] Step 7: Commit.

```bash
git add backend/agent/interp_agent.py backend/tests/test_interp_agent.py backend/tests/test_loop.py
git commit -m "feat(agent): thread builder/key/domain into run_interp_agent (BYO key)"
```

---

### Task 6: Wire the pod to launch the agent with config + a configurable domain

Updates `gpu_service._launch_agent` (lines 275-288) and the `/api/track` endpoint (lines 302-319) so the agent receives `builder`, `anthropic_api_key`, `domain`, and `model` from the process-level config. WS0 stores `AppConfig` on `app.state.config` and replaces `config.ANTHROPIC_API_KEY` here; WS3 ADDS the `domain` derivation and the threaded `run_interp_agent` call. The domain defaults to `prompts.DEFAULT_DOMAIN`, overridable by a `GLASSBOX_DOMAIN` env var (a pragmatic v1 seam until WS0/WS2 add a first-class `model.domain` field). **HARD DEPENDENCY: this task requires WS0 to have landed `backend/config.py` (with `ProbeBuilderConfig`/`ModelConfig`) and `pydantic-settings` in `pyproject.toml`.** The Step 3 body imports `from .config import ModelConfig, ProbeBuilderConfig`; pre-WS0 the real `backend/config.py` is flat globals with NO such classes, so this import ImportErrors — there is no working "bridge before WS0". DEPENDS ON Task 5 + WS0.

Files:
- Modify: `backend/gpu_service.py` (lines 275-288 `_launch_agent`; lines 302-319 `/api/track`; line 194 `anthropic_configured` read)
- Test: `backend/tests/test_api_track.py` is the orchestration-side proxy test (no pod config) — unaffected. The pod-side `/api/track` is covered by the gpu_service suite; add one threading test here.

Interfaces:
- Consumes (Task 5): `run_interp_agent(tracker_id, builder, anthropic_api_key, *, domain=..., model=None, client=None, generate_fn=None)` — `domain` and `model` are keyword-only (per Task 5), so `_launch_agent` MUST pass them by keyword, never positionally.
- Consumes (Task 2): `prompts.DEFAULT_DOMAIN`.
- Consumes (AppConfig contract / WS0): `config.AppConfig` on `app.state.config` with `.probes.builder` (ProbeBuilderConfig), `.anthropic_api_key`, `.model` (ModelConfig). SHARED FILE: WS0 lands `app.state.config` on `gpu_service`. HARD DEPENDENCY ON WS0 — there is no pre-WS0 bridge: the Step 3 `/api/track` body imports `from .config import ModelConfig, ProbeBuilderConfig`, which does not exist until WS0 lands, and `pyproject.toml` only ships `pydantic-settings` after WS0 Task 1. The `# WS0: replace with app.state.config` comments in Step 3 mark where the still-flat `config.ANTHROPIC_API_KEY` read and the `ProbeBuilderConfig()`/`ModelConfig()` construction become `request.app.state.config.*` reads ONCE WS0 has landed — they are NOT a way to run this task before WS0. The threading test below stubs `_launch_agent` itself, so the test does not exercise the `ProbeBuilderConfig`/`ModelConfig` import path, but the production `/api/track` body still requires WS0 to import at module load.
- Produces: `_launch_agent(tracker_id: str, *, builder, anthropic_api_key: str, domain: str, model) -> None`.
- Produces: `_resolve_domain() -> str` — reads `os.getenv("GLASSBOX_DOMAIN")` else `prompts.DEFAULT_DOMAIN`.

Steps:

- [ ] Step 1: Append a threading test to `backend/tests/test_gpu_service.py` (the autouse fixture already clears `POD_TOKEN`, so loopback would apply; this test stubs `run_interp_agent` and `_attempt_load`).

> POST-WS0 RECONCILIATION (this task hard-depends on WS0): WS0 Task 14 removes `from . import config` from `gpu_service` and moves the key read to `request.app.state.config.anthropic_api_key`, and rewrites `test_gpu_service.py` to set `gpu_service.app.state.config = AppConfig()` and patch via the sub-config (e.g. `gpu_service.app.state.config.pod_token = ...`). When WS0 has landed, write THIS test in that post-WS0 style instead of the flat-config form shown below: replace `monkeypatch.setattr(gpu_service.config, "ANTHROPIC_API_KEY", "sk-pod-key")` with `gpu_service.app.state.config.anthropic_api_key = "sk-pod-key"`, and rely on WS0's autouse `_state_config` fixture rather than `gpu_service.config`. The flat-`config` version below is shown only to make the assertion intent legible; since the whole task requires WS0, the executor MUST adapt it to the post-WS0 access path so the test is actually green. Do NOT ship the `gpu_service.config.*` form after WS0 — `gpu_service.config` no longer exists then.

```python
def test_track_launches_agent_with_config(monkeypatch):
    """POST /api/track must launch the agent with builder/key/domain threaded, not globals."""
    _install_stubs(monkeypatch)
    monkeypatch.setattr(gpu_service.config, "ANTHROPIC_API_KEY", "sk-pod-key")
    monkeypatch.setattr(gpu_service, "_is_loopback", lambda request: True)

    captured = {}

    def fake_launch(tracker_id, *, builder, anthropic_api_key, domain, model):
        captured.update(
            tracker_id=tracker_id, anthropic_api_key=anthropic_api_key, domain=domain
        )

    monkeypatch.setattr(gpu_service, "_launch_agent", fake_launch)
    monkeypatch.setenv("GLASSBOX_DOMAIN", "tax-advice assistant")

    client = TestClient(gpu_service.app)
    r = client.post("/api/track", json={"request": "watch for sycophancy"})
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert captured["anthropic_api_key"] == "sk-pod-key"
    assert captured["domain"] == "tax-advice assistant"
    assert captured["tracker_id"].startswith("watch-for-sycophancy-")
```

- [ ] Step 2: Run the test and confirm failure (current `_launch_agent(tracker_id)` takes no keyword args; `_resolve_domain` does not exist; `/api/track` calls `_launch_agent(tracker_id)`).

```bash
uv run pytest backend/tests/test_gpu_service.py::test_track_launches_agent_with_config -q 2>&1 | tail -20
```

Expected: `TypeError: fake_launch() missing ... keyword-only argument` OR `_launch_agent() got an unexpected keyword argument 'builder'` — because `/api/track` still calls `_launch_agent(tracker_id)` positionally without the new kwargs.

- [ ] Step 3: Edit `backend/gpu_service.py`. Add `import os` near the top imports (after `import time`), add `_resolve_domain`, and replace `_launch_agent` (lines 275-288) and the `_launch_agent` call inside `/api/track` (line 318):

Add near the existing imports (top of file, with `import asyncio` / `import time`):

```python
import os
```

Replace `_launch_agent` (lines 275-288) with:

```python
def _resolve_domain() -> str:
    """The target-assistant domain the probe-builder prompts describe. v1 reads GLASSBOX_DOMAIN;
    WS0/WS2 promote this to a first-class model config field. Neutral default, never medical."""
    from .agent import prompts

    return os.getenv("GLASSBOX_DOMAIN") or prompts.DEFAULT_DOMAIN


def _launch_agent(tracker_id: str, *, builder, anthropic_api_key: str, domain: str, model) -> None:
    """Run the blocking interp-agent pipeline (pod-local gemma + Claude) off the event loop."""
    from .agent.interp_agent import run_interp_agent
    from .science import concept_synth as cs

    def _run():
        try:
            run_interp_agent(
                tracker_id, builder, anthropic_api_key, domain=domain, model=model
            )
        except Exception as e:  # noqa: BLE001 - surface failure in the job record
            msg = str(e) or f"{type(e).__name__} during probe build"
            print(f"[gpu_service] interp agent failed ({tracker_id}): {msg}")
            cs.update_job(tracker_id, status="error", error=msg)

    asyncio.create_task(asyncio.to_thread(_run))
```

Replace the body of `/api/track` (lines 302-319) so it builds the config args and calls the new `_launch_agent`:

```python
@app.post("/api/track", dependencies=[Depends(_require_auth)])
async def track(body: dict) -> dict:
    """Submit a natural-language monitoring request. Trains a persona-vector probe in the
    background ON THIS POD (local gemma generation + Claude design/judge) and, if it clears the
    AUROC gate, registers it into THIS process's live `persona._trackers`."""
    from .agent.prompts import DEFAULT_DOMAIN  # noqa: F401 — domain resolved below
    from .config import ModelConfig, ProbeBuilderConfig
    from .science import concept_synth as cs

    if STATE["mode"] != "real":
        raise HTTPException(status_code=503, detail="model not loaded")
    anthropic_api_key = config.ANTHROPIC_API_KEY  # WS0: replace with request.app.state.config.anthropic_api_key
    if not anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set on pod")
    request = (body.get("request") or body.get("concept") or body.get("name") or "").strip()
    if not request:
        raise HTTPException(status_code=400, detail="request is required")
    tracker_id = cs.create_job(request)
    # WS0: pull builder/model from request.app.state.config.probes.builder / .model.
    _launch_agent(
        tracker_id,
        builder=ProbeBuilderConfig(),
        anthropic_api_key=anthropic_api_key,
        domain=_resolve_domain(),
        model=ModelConfig(),
    )
    return {"tracker_id": tracker_id, "status": "pending"}
```

- [ ] Step 4: Run the full gpu_service suite and confirm all pass.

```bash
uv run pytest backend/tests/test_gpu_service.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `11 passed` (10 from Task 1 + the 1 new threading test), no failures.

- [ ] Step 5: Commit.

```bash
git add backend/gpu_service.py backend/tests/test_gpu_service.py
git commit -m "feat(pod): launch interp agent with threaded config + configurable domain"
```

---

### Task 7: Remove the committed glassbox-dev-secret default from scripts + batch script

Removes the hardcoded `glassbox-dev-secret` POD token default from `scripts/run_with_pod.sh`, `scripts/pod_tailscale_bootstrap.sh`, and the docstring of `backend/batch_medqa_observability.py`. After this change, `POD_TOKEN` is empty unless the operator sets it; the local/tunnel workflow still works because Task 1 made the pod allow loopback without a token, and `run_with_pod.sh` connects over a loopback tunnel. This task does NOT depend on WS0.

> **OWNERSHIP (resolves the WS1 collision on the batch script):** WS3 owns the `glassbox-dev-secret` removal across all script/doc surfaces; `backend/batch_medqa_observability.py` is also listed in WS1's File list (`2026-06-22-ws1-observability.md` line 43) which removes the same example token while it drops Phoenix prints + neutralizes the medical docstring. The two edits target DIFFERENT content in that file (WS3: the `POD_TOKEN` example token on line 12; WS1: Phoenix prints + medical framing), so they do not textually collide. To stay safe against ordering, Step 3 below is written idempotently (replace-if-present); if WS1 already removed the token, Step 4's grep simply confirms it is gone and the commit is a no-op for that file.

Files:
- Modify: `scripts/run_with_pod.sh` (line 8)
- Modify: `scripts/pod_tailscale_bootstrap.sh` (line 32)
- Modify: `backend/batch_medqa_observability.py` (line 12 docstring) — SHARED with WS1 (disjoint content; idempotent edit).
- Test: verification is a `grep` proving the strings are gone from these three files.

Interfaces:
- Consumes: nothing (pure cleanup). `run_with_pod.sh` still exports `POD_URL`; it stops exporting a token default.
- Produces: no token default in source. `scripts/run_with_pod.sh` keeps `POD_TOKEN` as a pass-through (`export POD_TOKEN="${POD_TOKEN:-}"`) so a token set in the environment / `.env` still flows.

Steps:

- [ ] Step 1: Edit `scripts/run_with_pod.sh` line 8 from:

```bash
export POD_TOKEN="${POD_TOKEN:-glassbox-dev-secret}"
```

to:

```bash
# POD_TOKEN flows through from your environment / .env. Empty is fine for a loopback tunnel;
# set a strong token (openssl rand -hex 32) when binding the pod to a non-loopback address.
export POD_TOKEN="${POD_TOKEN:-}"
```

- [ ] Step 2: Edit `scripts/pod_tailscale_bootstrap.sh` line 32 from:

```bash
  export POD_TOKEN="${POD_TOKEN:-glassbox-dev-secret}"
```

to:

```bash
  # Set POD_TOKEN before running this script (e.g. POD_TOKEN=$(openssl rand -hex 32)).
  export POD_TOKEN="${POD_TOKEN:-}"
```

- [ ] Step 3: Edit `backend/batch_medqa_observability.py` line 12 (the docstring example) **idempotently** — replace the token only if it is still present (WS1 may have already removed it while neutralizing the medical docstring). From:

```python
  export POD_URL=http://localhost:8001 POD_TOKEN=glassbox-dev-secret
```

to:

```python
  export POD_URL=http://localhost:8001 POD_TOKEN=$(openssl rand -hex 32)
```

Idempotent application (safe whether or not WS1 already removed it):

```bash
if grep -q 'POD_TOKEN=glassbox-dev-secret' backend/batch_medqa_observability.py; then
  sed -i '' 's/POD_TOKEN=glassbox-dev-secret/POD_TOKEN=$(openssl rand -hex 32)/g' backend/batch_medqa_observability.py
fi
```

- [ ] Step 4: Verify the secret string is gone from all three files.

```bash
grep -rn "glassbox-dev-secret" scripts/run_with_pod.sh scripts/pod_tailscale_bootstrap.sh backend/batch_medqa_observability.py; echo "exit=$?"
```

Expected: no matches printed and `exit=1` (grep found nothing).

- [ ] Step 5: Commit.

```bash
git add scripts/run_with_pod.sh scripts/pod_tailscale_bootstrap.sh backend/batch_medqa_observability.py
git commit -m "chore(security): remove committed glassbox-dev-secret pod-token default from scripts"
```

---

### Task 8: Clean the secret default + add domain hint to .env.example

Removes the `glassbox-dev-secret` default from `.env.example`, documents generating a strong token, and adds the `GLASSBOX_DOMAIN` variable Task 6 consumes.

> **OWNERSHIP (resolves the WS1 collision):** The `glassbox-dev-secret` removal from `.env.example` is assigned to **WS3 (this task)** as the single owner — WS3 owns the pod-token secret cleanup end to end (scripts in Task 7, `.env.example` here, docs in Task 11). **WS1 must NOT also remove the `POD_TOKEN` default**; WS1 keeps only its own `.env.example` edits (rename `SENTRY_ORG`/`SENTRY_PROJECT` → `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG`, remove `PHOENIX_COLLECTOR_ENDPOINT`). WS1's plan (`2026-06-22-ws1-observability.md`, File list line 48) lists the `glassbox-dev-secret` removal too; that line is superseded by this ownership note — WS1 leaves the `POD_TOKEN` line for WS3. The two workstreams touch DISJOINT lines, so order between WS1 and WS3 no longer matters for this file. WS0 only introduces `config.yaml`/`config.example.yaml`; it does not edit the `POD_TOKEN` line.
>
> **IDEMPOTENT EDIT:** Because either WS1 (under its old plan) or WS0 may have already mutated the `POD_TOKEN` line by the time this task runs, the edit is written as grep-then-replace-if-present, with a branch that only ADDS `GLASSBOX_DOMAIN` when the secret is already gone. It must never depend on `glassbox-dev-secret` sitting on an exact line number.

Files:
- Modify: `.env.example` (the `POD_TOKEN` line, wherever it is; add a `GLASSBOX_DOMAIN` line in the Runtime block). SHARED with WS0/WS1 — WS3 owns the `POD_TOKEN`/`GLASSBOX_DOMAIN` lines only.
- Test: verification is a `grep` proving the secret is gone and the domain hint is present.

Interfaces:
- Consumes: nothing.
- Produces: a `.env.example` with `POD_TOKEN=` (empty) plus a generation hint, and a commented `GLASSBOX_DOMAIN=` example. SHARED FILE flag above.

Steps:

- [ ] Step 1: Replace the `POD_TOKEN` line **idempotently** (works whether or not a prior workstream already removed the `glassbox-dev-secret` default). The aim is: a `POD_TOKEN=` line preceded by the three-line hint comment, and no `glassbox-dev-secret` anywhere.

```bash
# If the committed secret default is still present, rewrite the whole POD_TOKEN line + hint.
if grep -q 'POD_TOKEN=glassbox-dev-secret' .env.example; then
  python3 - <<'PY'
import pathlib, re
p = pathlib.Path(".env.example")
hint = (
    "# Required when the pod binds to a non-loopback address (Docker self-host). Generate one with:\n"
    "#   openssl rand -hex 32\n"
    "# Empty is allowed ONLY for a loopback-bound pod reached over an SSH tunnel.\n"
    "POD_TOKEN="
)
text = p.read_text()
text = re.sub(r'^POD_TOKEN=glassbox-dev-secret.*$', hint, text, count=1, flags=re.M)
p.write_text(text)
PY
elif ! grep -qE '^POD_TOKEN=' .env.example; then
  # WS0/WS1 already dropped the POD_TOKEN line entirely — re-add an empty, hinted one.
  printf '%s\n' \
    '# Required when the pod binds to a non-loopback address (Docker self-host). Generate one with:' \
    '#   openssl rand -hex 32' \
    '# Empty is allowed ONLY for a loopback-bound pod reached over an SSH tunnel.' \
    'POD_TOKEN=' >> .env.example
fi
# else: a bare `POD_TOKEN=` line already exists (secret already removed) — leave it, just add domain in Step 2.
```

- [ ] Step 2: Add the `GLASSBOX_DOMAIN` variable **only if it is not already present** (idempotent — a re-run or a prior workstream must not duplicate it). Append to the Runtime block:

```bash
if ! grep -q '^GLASSBOX_DOMAIN=' .env.example; then
  printf '%s\n' \
    '' \
    '# Optional: describes the assistant the probe-builder monitors (drives the agent prompt copy).' \
    '# Neutral default is "AI assistant"; set a domain to make custom probes domain-aware, e.g.:' \
    '#   GLASSBOX_DOMAIN=legal contract assistant' \
    'GLASSBOX_DOMAIN=' >> .env.example
fi
```

- [ ] Step 3: Verify the secret default is gone and the domain hint is present.

```bash
grep -n "glassbox-dev-secret" .env.example; echo "secret_exit=$?"; grep -n "GLASSBOX_DOMAIN" .env.example
```

Expected: first grep prints nothing with `secret_exit=1`; second grep prints the `GLASSBOX_DOMAIN` lines.

- [ ] Step 4: Confirm the file still parses as a dotenv-style file (no syntax surprises from the edit) by sourcing it in a subshell.

```bash
bash -c 'set -a; . ./.env.example; set +a; echo "POD_TOKEN=[$POD_TOKEN] GLASSBOX_DOMAIN=[$GLASSBOX_DOMAIN]"'
```

Expected: `POD_TOKEN=[] GLASSBOX_DOMAIN=[]` (both empty, no errors).

- [ ] Step 5: Commit.

```bash
git add .env.example
git commit -m "chore(security): drop glassbox-dev-secret from .env.example; add GLASSBOX_DOMAIN"
```

---

### Task 9: Dockerfile + .dockerignore for the GPU service

Creates a `Dockerfile` that builds the GPU pod service (`backend.gpu_service:app`) with the ML extras, plus a `.dockerignore` to keep the build context small. The image binds to `0.0.0.0` (a non-loopback address), so `POD_TOKEN` is mandatory at runtime — Task 1's fail-closed gate enforces it. Writing the Dockerfile + the structural pytest does NOT depend on WS0. But the **built image only runs successfully once WS0 has landed**: `uv sync --extra ml` installs against `pyproject.toml`, and `backend.gpu_service` at startup does `from .config import load_config` / constructs `ProbeBuilderConfig`/`ModelConfig`, which require WS0 Task 1's addition of `pydantic-settings>=2.0` + `pydantic>=2.9` + `pyyaml>=6.0` to the base deps and WS0's `backend/config.py`. So sequence a real `docker compose up` / runtime smoke after WS0; the structural CI test below is the only WS0-independent gate.

Files:
- Create: `Dockerfile`
- Create: `.dockerignore`
- Test: verification is `docker build` syntax validation via the buildx dry-run (or a hadolint-style structural grep when Docker is unavailable in CI). NOTE: the structural test does not import `backend.config`, so it passes pre-WS0; a runtime image boot requires WS0.

Interfaces:
- Consumes (runtime): `backend.gpu_service:app`; env `POD_TOKEN`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `HF_HOME`, `GLASSBOX_DOMAIN`; `pyproject.toml` ML extra. **RUNTIME DEPENDS ON WS0:** the image needs `pydantic-settings>=2.0` (+ `pydantic>=2.9`, `pyyaml>=6.0`) in `pyproject.toml`, which WS0 Task 1 adds — without it, `from .config import load_config` / `ProbeBuilderConfig` fail at container startup. Build the image / run `docker compose up` only after WS0 has landed. (Do NOT add these deps from WS3 to "ship first" — WS0 owns `pyproject.toml`'s dependency block; duplicating the add here would collide with WS0 Task 1.)
- Produces: an image whose default command is `uvicorn backend.gpu_service:app --host 0.0.0.0 --port 8000`.

Steps:

- [ ] Step 1: Write a structural test that asserts the Dockerfile exists and has the load-bearing lines (this is the unit "test" for an infra file — it fails before the file exists). Create `backend/tests/test_dockerfile.py`:

```python
"""WS3: the GPU-service Dockerfile must exist and pin the load-bearing build/run steps."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_exists_and_runs_gpu_service():
    df = (ROOT / "Dockerfile").read_text()
    # CUDA-capable base so torch finds a GPU at runtime.
    assert "nvidia/cuda" in df or "pytorch/pytorch" in df
    # Installs the ML extra (torch/transformers/sae-lens) — not just base deps.
    assert "--extra ml" in df or "[ml]" in df
    # Binds 0.0.0.0 (non-loopback) → Task 1's fail-closed auth requires POD_TOKEN.
    assert "backend.gpu_service:app" in df
    assert "0.0.0.0" in df
    assert "8000" in df


def test_dockerignore_excludes_heavy_dirs():
    di = (ROOT / ".dockerignore").read_text()
    for pat in (".venv", "node_modules", "__pycache__"):
        assert pat in di
```

- [ ] Step 2: Run the test and confirm failure (no Dockerfile / .dockerignore yet).

```bash
uv run pytest backend/tests/test_dockerfile.py -q 2>&1 | tail -20
```

Expected: `FileNotFoundError: ... 'Dockerfile'`.

- [ ] Step 3: Create `Dockerfile`:

```dockerfile
# GlassBox GPU pod service — self-host on your own NVIDIA GPU.
# Build:  docker build -t glassbox-gpu .
# Run:    docker run --gpus all --env-file .env -p 8000:8000 glassbox-gpu
# The pod binds 0.0.0.0, so POD_TOKEN is MANDATORY (auth fails closed on non-loopback binds).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 (matches .python-version) + git for SAELens, plus the uv installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Resolve deps first (cache layer) using only the manifest, then the source.
COPY pyproject.toml uv.lock* /app/
RUN uv venv /opt/venv --python 3.12 \
    && uv sync --extra ml --no-install-project

COPY . /app
RUN uv sync --extra ml

EXPOSE 8000

# Model + SAE weights download on first request (cached to HF_HOME volume). Bind 0.0.0.0
# so the host port-forward reaches it; auth fails closed without POD_TOKEN.
CMD ["uvicorn", "backend.gpu_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] Step 4: Create `.dockerignore`:

```
.git
.venv
**/__pycache__
**/*.pyc
.pytest_cache
frontend/node_modules
frontend/dist
**/.cache
*.log
.env
backend/science/probe_jobs
docs/superpowers
```

- [ ] Step 5: Run the structural test and confirm pass; also confirm the Dockerfile parses with buildx (if Docker is present — skip the build, just validate the syntax via `--call=check`). If Docker is unavailable, the structural pytest is the gate.

```bash
uv run pytest backend/tests/test_dockerfile.py -q 2>&1 | grep -iE "passed|failed"
docker buildx build --call=check . 2>&1 | tail -5 || echo "docker unavailable — structural test is the gate"
```

Expected: `2 passed`; the `docker buildx ... --call=check` either reports `Check complete, no warnings found` or prints `docker unavailable — structural test is the gate`.

- [ ] Step 6: Commit.

```bash
git add Dockerfile .dockerignore backend/tests/test_dockerfile.py
git commit -m "feat(deploy): add GPU-service Dockerfile + .dockerignore for self-host"
```

---

### Task 10: docker-compose for the GPU service

Creates a `docker-compose.yml` that runs the Task 9 image as one `gpu` service: reserves an NVIDIA GPU, wires `.env`, persists the HuggingFace cache to a named volume, and publishes port 8000. This replaces the manual `nohup`/Tailscale dance from the design doc.

Files:
- Create: `docker-compose.yml`
- Test: structural pytest asserting the compose file has the GPU reservation, env-file, port, and HF cache volume.

Interfaces:
- Consumes (Task 9): the `Dockerfile` build context; env vars `POD_TOKEN`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `GLASSBOX_DOMAIN` via `env_file: .env`.
- Produces: a `gpu` service reachable at `http://localhost:8001` on the host (mapped to the container's 8000) so it lines up with the default `POD_URL=http://localhost:8001`.

Steps:

- [ ] Step 1: Create `backend/tests/test_compose.py`:

```python
"""WS3: docker-compose must reserve a GPU, wire .env, map the port, and persist the HF cache."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compose_structure():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "env_file" in text
    assert ".env" in text
    # Host 8001 -> container 8000 so the default POD_URL (localhost:8001) lines up.
    assert "8001:8000" in text
    # NVIDIA GPU reservation (compose device-request form).
    assert "driver: nvidia" in text
    # HF weights cache persisted across container restarts.
    assert "huggingface" in text


def test_compose_parses_as_yaml():
    import yaml

    data = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "services" in data
    assert "gpu" in data["services"]
```

- [ ] Step 2: Run the test and confirm failure (no compose file yet).

```bash
uv run pytest backend/tests/test_compose.py -q 2>&1 | tail -20
```

Expected: `FileNotFoundError: ... 'docker-compose.yml'`.

- [ ] Step 3: Create `docker-compose.yml`:

```yaml
# GlassBox GPU pod service — self-host quickstart.
#   1. cp .env.example .env  and set ANTHROPIC_API_KEY, HF_TOKEN, and a strong POD_TOKEN
#      (openssl rand -hex 32).
#   2. docker compose up --build
#   3. Point the orchestration backend at it: POD_URL=http://localhost:8001
# Requires the NVIDIA Container Toolkit on the host.
services:
  gpu:
    build:
      context: .
      dockerfile: Dockerfile
    image: glassbox-gpu
    env_file: .env
    ports:
      - "8001:8000"   # host 8001 -> container 8000 (matches the default POD_URL)
    volumes:
      - hf-cache:/workspace/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  hf-cache:
```

- [ ] Step 4: Run the structural test and validate the compose file with the Docker CLI if present.

```bash
uv run pytest backend/tests/test_compose.py -q 2>&1 | grep -iE "passed|failed"
docker compose -f docker-compose.yml config >/dev/null 2>&1 && echo "compose valid" || echo "docker unavailable — structural test is the gate"
```

Expected: `2 passed`; then either `compose valid` or `docker unavailable — structural test is the gate`.

- [ ] Step 5: Commit.

```bash
git add docker-compose.yml backend/tests/test_compose.py
git commit -m "feat(deploy): add docker-compose for self-hosting the GPU service"
```

---

### Task 11: Docker quickstart snippet for the deploy doc + secret cleanup in the runbook/probe docs

WS3 owns the GPU-service Docker packaging (Tasks 9–10) and the `glassbox-dev-secret` secret cleanup. **WS4 owns the canonical self-host/deployment doc and the Tailscale runbook header demotion** — so this task does NOT create a competing canonical doc and does NOT rewrite the runbook header. Instead it (a) supplies a self-contained **Docker quickstart snippet** (`docs/_snippets/docker-gpu-quickstart.md`) that WS4's `docs/deployment.md` includes verbatim in its "GPU pod via Docker" section, and (b) removes the remaining `glassbox-dev-secret` occurrences from `docs/tailscale-pod-runbook.md` and `docs/probe-training.md`. This task does NOT depend on WS0.

> **OWNERSHIP (resolves the WS4 collision):** There is exactly ONE canonical self-host/deployment doc: **`docs/deployment.md`, owned by WS4** (`2026-06-22-ws4-oss-infra.md` Task 8, lines 783–999 — it creates `docs/deployment.md` and demotes `docs/tailscale-pod-runbook.md`'s header to point at `deployment.md`). WS3 originally planned a second canonical doc (`docs/self-host-gpu.md`) and a conflicting runbook-header rewrite pointing at it; both are REMOVED to avoid two competing canonical docs and two conflicting header rewrites of the same runbook line 1.
>
> - WS3 → supplies the Dockerfile (Task 9), the docker-compose (Task 10), and a reusable Docker quickstart snippet (this task). WS3 removes the committed secret from the runbook body + probe-training doc.
> - WS4 → owns `docs/deployment.md` (the canonical guide; it pulls in the WS3 snippet's commands for the Docker GPU-pod section) AND the single rewrite of the `docs/tailscale-pod-runbook.md` header demoting it to an optional appendix that links to `docs/deployment.md`.
> - The runbook header (line 1) is rewritten by WS4 ONLY. WS3 must NOT touch the runbook header — it edits only the `glassbox-dev-secret` occurrences in the runbook body.
>
> If WS3 ships before WS4: the snippet file simply exists for WS4 to consume later; the runbook body is secret-free and its header is left for WS4 to demote. If WS4 ships first: `docs/deployment.md` already exists; WS3 only writes the snippet (idempotently) and cleans the secret. Either order is safe.

Files:
- Create: `docs/_snippets/docker-gpu-quickstart.md` (the reusable Docker quickstart that WS4's `docs/deployment.md` includes; NOT a standalone canonical doc).
- Modify: `docs/tailscale-pod-runbook.md` (body only — the `glassbox-dev-secret` occurrences at the `POD_TOKEN` table row + the `export POD_TOKEN=...` lines). DO NOT edit the header (WS4 owns that).
- Modify: `docs/probe-training.md` (the `glassbox-dev-secret` deploy snippet).
- Test: verification is a `grep` proving the secret is gone from both docs, plus the snippet file existing with the Docker quickstart.

Interfaces:
- Consumes (Task 9, 10): the Dockerfile + compose quickstart commands.
- Produces: `docs/_snippets/docker-gpu-quickstart.md` (a self-contained Docker quickstart section for WS4's `docs/deployment.md` to include) and a secret-free Tailscale runbook body + probe-training doc. Does NOT produce a canonical deploy doc and does NOT touch the runbook header — both are WS4's.

Steps:

- [ ] Step 1: Create `docs/_snippets/docker-gpu-quickstart.md` — the reusable Docker quickstart WS4 includes in `docs/deployment.md`'s "GPU pod via Docker" section. It is a section fragment (starts at a `##`-level heading, no top-level `#` title) so it slots into the canonical doc without introducing a competing H1:

````markdown
## GPU pod via Docker (quickstart)

Self-host the GPU pod service (`backend.gpu_service`) on your own NVIDIA GPU with Docker. Bring
your own Anthropic API key.

### Prerequisites

- An NVIDIA GPU + driver, and the **NVIDIA Container Toolkit** installed on the host.
- Docker with Compose v2 (`docker compose`).
- An **Anthropic API key** (the probe builder + auto-interp labels call Claude).
- A **HuggingFace token** if the model/SAE you load is gated.

### 1. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Required? | Notes |
|----------|-----------|-------|
| `ANTHROPIC_API_KEY` | Yes (for probe builder / auto-interp) | Your own key. |
| `POD_TOKEN` | **Yes** | The container binds `0.0.0.0`; auth **fails closed** without a token. Generate one: `openssl rand -hex 32`. |
| `HF_TOKEN` | If the model is gated | HuggingFace download token. |
| `GLASSBOX_DOMAIN` | Optional | Describes the assistant your probes monitor (default `AI assistant`). |

Never put secrets in `config.yaml` — they load only from `.env` / the environment.

### 2. Bring up the GPU service

```bash
docker compose up --build
```

The model + SAE download on first request and cache to the `hf-cache` volume. Watch the logs;
the service is ready when `/health` reports `"mode": "real"`:

```bash
curl -s http://localhost:8001/health
# {"mode":"real","model_loaded":true,"sae_loaded":true,...}
```

### 3. Point the orchestration backend at it

On the machine running `backend.app`, set:

```bash
export POD_URL=http://localhost:8001      # or http://<gpu-host>:8001
export POD_TOKEN=<the same token you set in .env>
./scripts/run_with_pod.sh
```

The backend proxies `/turn` and `/api/track` to the pod over HTTP with the bearer token.

### Auth model (fail closed)

- **Token set** → every request must send `Authorization: Bearer <POD_TOKEN>`.
- **Token unset + loopback request** (e.g. an SSH tunnel to `127.0.0.1`) → served (local-only).
- **Token unset + non-loopback request** → **rejected** (`401`). A public bind can never serve
  unauthenticated.

> Behind a restrictive firewall that blocks outbound SSH to your GPU host? See the optional
> appendix [`docs/tailscale-pod-runbook.md`](tailscale-pod-runbook.md).
````

- [ ] Step 2: Remove every `glassbox-dev-secret` from the **body** of `docs/tailscale-pod-runbook.md` (the `POD_TOKEN` table row + the `export POD_TOKEN=...` lines). **Do NOT edit the header (line 1) — WS4 owns the header demotion.** Idempotent (replace-if-present):

```bash
if grep -q "glassbox-dev-secret" docs/tailscale-pod-runbook.md; then
  sed -i '' 's/POD_TOKEN=glassbox-dev-secret/POD_TOKEN="$POD_TOKEN"  # set yourself: openssl rand -hex 32/g; s/| `glassbox-dev-secret` |/| _(generate: `openssl rand -hex 32`)_ |/g' docs/tailscale-pod-runbook.md
fi
```

- [ ] Step 3: Remove `glassbox-dev-secret` from `docs/probe-training.md`'s deploy snippet. Idempotent:

```bash
if grep -q "glassbox-dev-secret" docs/probe-training.md; then
  sed -i '' 's/POD_TOKEN=glassbox-dev-secret/POD_TOKEN="$POD_TOKEN"/g' docs/probe-training.md
fi
```

- [ ] Step 4: Verify the secret is gone from both docs and the snippet exists with the Docker quickstart.

```bash
grep -rn "glassbox-dev-secret" docs/tailscale-pod-runbook.md docs/probe-training.md; echo "secret_exit=$?"
test -f docs/_snippets/docker-gpu-quickstart.md && grep -q "docker compose up" docs/_snippets/docker-gpu-quickstart.md && echo "snippet OK"
```

Expected: first grep prints nothing with `secret_exit=1`; then `snippet OK`.

- [ ] Step 5: Commit.

```bash
git add docs/_snippets/docker-gpu-quickstart.md docs/tailscale-pod-runbook.md docs/probe-training.md
git commit -m "docs(deploy): add Docker GPU quickstart snippet; remove glassbox-dev-secret from runbook/probe docs"
```

---

### Task 12: Guard test against the secret default reappearing

Adds a regression test that scans tracked scripts and docs for the `glassbox-dev-secret` literal and fails if it ever reappears, so a future copy-paste cannot reintroduce the committed default. Excludes the design-spec files under `docs/superpowers/specs/` (they legitimately reference the old default as the thing being removed).

Files:
- Create: `backend/tests/test_secrets_cleanup.py`
- Test: the test is its own verification.

Interfaces:
- Consumes: the repo tree (read-only).
- Produces: `test_no_committed_pod_token_default()` — fails if `glassbox-dev-secret` appears in any tracked file outside the allowed spec/plan paths.

Steps:

- [ ] Step 1: Create `backend/tests/test_secrets_cleanup.py`:

```python
"""WS3 regression guard: the committed dev pod-token default must never reappear.

The design spec + plan files under docs/superpowers/ legitimately reference the old default
as the thing being removed, so they are excluded. Everything else (scripts, .env.example,
runtime docs, source) must be clean."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEEDLE = "glassbox-dev-secret"
ALLOWED_PREFIXES = ("docs/superpowers/",)


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_committed_pod_token_default():
    offenders = []
    for rel in _tracked_files():
        if rel.startswith(ALLOWED_PREFIXES):
            continue
        path = ROOT / rel
        try:
            text = path.read_text(errors="ignore")
        except (OSError, IsADirectoryError):
            continue
        if NEEDLE in text:
            offenders.append(rel)
    assert not offenders, (
        f"'{NEEDLE}' must not be committed outside docs/superpowers/; found in: {offenders}"
    )
```

- [ ] Step 2: Run the test and confirm it passes (Tasks 7, 8, 11 already removed every offender; this proves the cleanup is complete and locks it in).

```bash
uv run pytest backend/tests/test_secrets_cleanup.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `1 passed`. (If it FAILS, the failure message lists any file still containing the literal — fix that file, re-run, then proceed.)

- [ ] Step 3: Sanity-check the guard actually catches a regression by temporarily reintroducing the needle in a scratch-tracked file, confirming failure, then removing it. (Optional belt-and-braces; the assert message above is the contract.)

```bash
printf 'POD_TOKEN=glassbox-dev-secret\n' > scripts/_guard_probe.sh && git add scripts/_guard_probe.sh
uv run pytest backend/tests/test_secrets_cleanup.py -q 2>&1 | grep -iE "failed|_guard_probe"
git rm -f scripts/_guard_probe.sh
```

Expected: the pytest line reports `1 failed` and names `scripts/_guard_probe.sh`; the final `git rm` removes the probe file.

- [ ] Step 4: Confirm the guard passes again now that the probe file is gone.

```bash
uv run pytest backend/tests/test_secrets_cleanup.py -q 2>&1 | grep -iE "passed|failed"
```

Expected: `1 passed`.

- [ ] Step 5: Commit.

```bash
git add backend/tests/test_secrets_cleanup.py
git commit -m "test(security): guard against the glassbox-dev-secret default reappearing"
```

---

### Task 13: Full-suite green check

Runs the entire non-GPU backend suite to confirm WS3's changes (auth fail-closed, agent/judge config threading, Docker/compose infra tests, secret cleanup) keep the existing 30-file suite green. The GPU-gated `engine`/`sae` paths are not exercised (no weights); WS3 only stubbed them.

> **EXISTING TESTS THAT CHANGE BEHAVIOR UNDER WS3 (migrated, not just added):** three existing tests are *migrated* to the new signatures by WS3 — `backend/tests/test_judge.py` (Task 4 Step 5: `judge_filter` now takes `builder` + `anthropic_api_key`), `backend/tests/test_loop.py` (Task 5 Step 5: `run_interp_agent` required positionals + the `judge_filter` monkeypatch lambda), and `backend/tests/test_agent_persist.py` (Task 3 Step 5: `ctx` gains `'model': ModelConfig()`). The signature changes in Tasks 3/4/5 WOULD break these three tests if left unmigrated; the migrations are part of those tasks, so the full-suite gate below MUST account for them. (If WS0 Task 14 already migrated all three, the migrations in Tasks 3/4/5 are no-ops and the suite is green for the same reason.) The gate is NOT blind to these — they are listed in the expected output.

Files:
- Test: the whole `backend/tests/` suite.

Interfaces:
- Consumes: every WS3 task's output, including the three migrated existing tests (`test_judge.py`, `test_loop.py`, `test_agent_persist.py`).
- Produces: a green suite (the milestone safety net per design doc §8).

Steps:

- [ ] Step 1: Run the full suite.

```bash
uv run pytest backend/tests -q 2>&1 | tail -25
```

Expected: all tests pass. WS3 ADDED test files: `test_agent_prompts.py` (6), `test_agent_tools.py` (4), `test_concept_synth_judge.py` (2), `test_interp_agent.py` (2), `test_dockerfile.py` (2), `test_compose.py` (2), `test_secrets_cleanup.py` (1), and the modified `test_gpu_service.py` (11). WS3 MIGRATED existing test files (changed to track the new signatures, must still pass): `test_judge.py` (judge_filter gains `builder` + `anthropic_api_key`), `test_loop.py` (run_interp_agent required positionals + the judge_filter monkeypatch lambda), `test_agent_persist.py` (ctx gains `'model': ModelConfig()`, asserts `data["layer"] == ModelConfig().layer`). No failures.
>
> Two expected non-failure conditions, both ordering-driven, NOT silent breakage:
> 1. If WS0 has NOT yet landed `backend/config.py` with `ProbeBuilderConfig`/`ModelConfig` (or `pydantic-settings` in `pyproject.toml`), every WS3 file that imports `backend.config` — the four added agent/judge test files AND the three migrated existing tests — errors on import. That is the hard WS0 dependency: land WS0 first, then re-run. This is collection-time ImportError, distinct from an assertion failure.
> 2. If WS0 Task 14 already migrated `test_judge.py`/`test_loop.py`/`test_agent_persist.py`, the WS3 migrations in Tasks 3/4/5 were no-ops; the three tests pass for the same reason. Either way, after WS0 + WS3 the three migrated tests are green.
>
> If the run shows a TypeError/KeyError in `test_judge.py`/`test_loop.py`/`test_agent_persist.py` (not an ImportError), a signature migration in Task 3/4/5 was skipped — go back and apply that task's "migrate existing test" step before re-running.

- [ ] Step 2: Confirm no `Co-Authored-By: Claude` trailer slipped into the WS3 commits.

```bash
git log --format='%H %s%n%b' origin/main..HEAD 2>/dev/null | grep -i "co-authored-by: claude"; echo "trailer_exit=$?"
```

Expected: no output and `trailer_exit=1` (grep found no Claude co-author trailer in the WS3 commits).

- [ ] Step 3: Commit nothing if the suite is already clean (this task is a verification gate). If any cross-workstream import ordering surfaced a fix, commit it:

```bash
git status --porcelain
# If empty, no commit needed — WS3 is complete.
```

Expected: clean working tree (`git status --porcelain` prints nothing), confirming WS3 is fully landed and green.
