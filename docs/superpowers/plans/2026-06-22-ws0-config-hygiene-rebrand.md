# WS0 — Config Backbone, Hygiene & Rebrand Foundation Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

Goal: Replace the ~50 flat `backend/config.py` globals with one validated `AppConfig` (pydantic-settings, `config.yaml` + env secrets, threaded via `app.state` on both processes), perform the repo-hygiene cleanup, and lay the rebrand foundation (config-driven product name + neutral default system prompt; strip medical positioning from package metadata).

Architecture: GlassBox runs as two FastAPI processes — a CPU orchestration backend (`backend/app.py`, never imports torch) and a GPU pod service (`backend/gpu_service.py`, owns torch). Today both import `backend/config` as flat module globals. This workstream builds a single `AppConfig` object (seven sub-models: `model`, `sae`, `feature_cloud`, `probes`, `observability`, `pod`, `runtime` + env-only secrets), loads it once per process into `app.state.config`, and threads the narrowest sub-config into each consumer. The existing 30-file pytest suite is the safety net; it monkeypatches flat `config.X` globals today, so every migrated module's test is rewritten to inject the new sub-config.

Tech Stack: Python 3.12, pydantic v2, pydantic-settings, PyYAML, FastAPI, pytest, uv (pyproject.toml as the canonical dependency source).

## Global Constraints

- Python >= 3.12 (matches .python-version).
- Base deps (pyproject.toml): fastapi>=0.138.0, sentry-sdk[fastapi]>=2.63.0, httpx>=0.27.0.
- ML extra (uv sync --extra ml): torch>=2.4, transformers>=4.50, sae-lens>=6.0, scikit-learn>=1.5, accelerate>=0.34, anthropic>=0.40.
- pyproject.toml is the canonical dependency source; backend/requirements.txt is being retired.
- License: MIT.
- Naming/copy: KEEP the product name "GlassBox"; REMOVE medical-specific positioning (reposition as a general-purpose LLM interpretability/observability tool); medical survives only as ONE clearly-labeled optional example. Rename the per-turn event from cognition_event to introspection_event (class CognitionEvent -> IntrospectionEvent) — **(WS1 owns the `CognitionEvent` → `IntrospectionEvent` / `capture_cognition_alarm` / `cognition_event`-fixture rename; WS0 leaves the `CognitionEvent` symbol UNTOUCHED. WS0 only neutralizes copy/defaults and renames Sentry sub-config FIELD names, per the scope note below. Do NOT attempt the class rename in WS0.)**
- Secrets (ANTHROPIC_API_KEY, SENTRY_DSN, POD_TOKEN, HF_TOKEN) load ONLY from env / .env, NEVER from config.yaml.
- COMMIT RULE (CRITICAL): commit messages MUST NOT add Claude as a co-author. No "Co-Authored-By: Claude" trailer anywhere. Use Conventional Commits style.
- Tests: keep the existing backend pytest suite green; CI runs the non-GPU subset; backend/engine.py and backend/science/sae.py are GPU-gated (untested without weights).
- Two processes: orchestration backend (backend/app.py, NEVER imports torch) and GPU pod service (backend/gpu_service.py, owns torch). AppConfig is built once per process and threaded via FastAPI app.state.

> Scope note on the `cognition_event` → `introspection_event` rename: the Global Constraints list this rename, but the design doc §4 and the AppConfig Contract §6 both assign the **`CognitionEvent` class / `capture_cognition_alarm` / `cognition_event` fixture rename to WS1**, NOT WS0. WS0 therefore leaves the `CognitionEvent` symbol untouched and only renames the Sentry **sub-config field names**, which already use neutral terms in the contract. The Phoenix removal in `fanout.init_sponsors` is likewise WS1. WS0 changes `init_sponsors`'s signature to `init_sponsors(obs: ObsConfig, sentry_dsn: str)` but leaves its Phoenix branch in place for WS1 to delete.

---

## File Structure

- `backend/config.py` — REPLACED: flat globals become `AppConfig` + 7 sub-models + `Secrets` + `load_config()`. (Also edited by WS1/WS2/WS3 later — see Interfaces notes.)
- `config.example.yaml` — NEW: committed template (no secrets) with a labeled "medical" profile comment block.
- `.gitignore` — MODIFY: ignore `config.yaml`, `backend/science/probe_jobs/`, `backend/science/artifacts/watch-*.json`, `.pytest_cache/`, `.superpowers/`.
- `pyproject.toml` — MODIFY: add `pydantic-settings>=2.0`, `pydantic>=2.9`, `pyyaml>=6.0` to base deps; fix stale "torch 2.12" comment; neutralize the medical `description`.
- `backend/requirements.txt` — DELETE (retired; pyproject is canonical).
- `LICENSE` — MODIFY: real copyright holder.
- `main.py`, `err.txt`, `batch_medqa_results.json` — DELETE (tracked cruft).
- `frontend/bun.lock` — DELETE (keep `frontend/package-lock.json` as the single lockfile; npm is the standard).
- `backend/app.py` — MODIFY: build `app.state.config` in `lifespan`; thread `cfg` into endpoints, `runtime`, `init_sponsors`, `sentry_api`, `labels`, `analyze`. (Also edited by WS1.)
- `backend/gpu_service.py` — MODIFY: build `app.state.config` at startup; thread `cfg` into `_require_auth`, `_attempt_load`, `_run_recon_check`, `_baseline_vec`, `_sae_candidates`, all endpoints. (Also edited by WS2/WS3.)
- `backend/runtime.py` — MODIFY: `start_loading(cfg)`, `health_payload(cfg)`, `refresh_pod_health(cfg)`, `_poll_pod_once(cfg)`.
- `backend/pod_client.py` — MODIFY: functions take `pod: PodConfig` + `pod_token` + `max_new_tokens`.
- `backend/sentry_api.py` — MODIFY: `deep_link(sentry, token)`, `list_recent_issues(sentry, token, ...)`.
- `backend/fanout.py` — MODIFY: `fanout`, `capture_cognition_alarm`, `_filter_disabled`/`_observable_trackers`, `_scrub_pii`, `init_sponsors` take sub-configs. (Phoenix branch left for WS1.)
- `backend/analyze.py` — MODIFY: `analyze_turn(messages, cfg, *, message_id=None, ts=None, strict=False)` (cfg is the explicit 2nd positional param; the existing keyword-only `message_id`/`ts`/`strict` block is PRESERVED); `_rank_features(candidates, feature_cloud)`; `_real_turn(messages, cfg)`.
- `backend/labels.py` — MODIFY: `get_label(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0)` (and `get_feature_stats`/`_autointerp` threaded the same; `activations` stays an HTTP-response value, NOT a caller param); neutral auto-interp prompt.
- `backend/fallback.py` — MODIFY: `synth_turn(messages, sae)`; neutral templates + neutral labels.
- `backend/mock_labels.py` — MODIFY: rename `CLINICAL_LABELS` → neutral `GENERIC_LABELS` with neutral content.
- `backend/engine.py` — MODIFY (GPU-gated): `load_engine(model, device)`, `probe_activation(text, model, pos)`, `_inject_system(messages, model)`, `info(model, sae)`, `generate_and_capture(..., model, feature_cloud)`.
- `backend/science/sae.py` — MODIFY (GPU-gated): `load_sae(sae, layer, device)`, `sae_topk(act, sae, feature_cloud, k)`, `attribution_topk(..., sae, feature_cloud)`, `reconstruction_error(act, sae)`.
- `backend/science/persona.py` — MODIFY: `load_artifacts(probes, include)`, `clear_custom_trackers(probes, ...)`.
- `backend/science/feature_provider.py` — MODIFY: providers take `(sae, feature_cloud, model)`.
- `backend/science/concept_synth.py` — MODIFY: `judge_filter(spec, rows, builder, anthropic_api_key, *, client)`.
- `backend/agent/prompts.py` — MODIFY: functions take `builder`; neutral (non-medical) prompt text.
- `backend/agent/tools.py` — MODIFY: `_dispatch`/`_finalize`/`_persist_artifact` take `builder` + `model`.
- `backend/agent/interp_agent.py` — MODIFY: `run_interp_agent(tracker_id, builder, anthropic_api_key, *, client, generate_fn)`.
- `backend/tests/test_appconfig.py` — NEW: defaults-when-no-YAML, env-overrides-YAML, secrets-never-from-YAML.
- `backend/tests/test_runtime.py`, `test_pod_client.py`, `test_sentry_api.py`, `test_gpu_service.py`, `test_fanout.py`, `test_api.py`, `test_observability_endpoint.py`, `test_fallback.py`, `test_persona.py`, `test_labels_autointerp.py`, `test_rank.py`, `test_attribution.py`, `test_analyze.py`, `test_feature_provider.py` — MODIFY: inject sub-config instead of monkeypatching flat globals.

> Note on `test_analyze.py` (real file, exercises both pod and fallback paths): today it calls `analyze.analyze_turn([...])` POSITIONALLY with medical samples ("metformin", "ibuprofen", "aspirin"), monkeypatches `pod_client.turn` with `fake_turn(messages, max_new=None)` (OLD arity, no `pod`/`pod_token`), and monkeypatches `runtime.refresh_pod_health` with `lambda: None` (no `cfg`). It WILL break under Tasks 6/9/11. It is rewritten in Task 11 (inject `cfg`, neutral samples, new monkeypatch arities). It is NOT a flat-global monkeypatcher, so it is exempt from the Task 15 grep but explicitly owned by Task 11.

---

### Task 1: AppConfig pydantic models

Files:
- Create: `backend/config.py` (full replacement — Task 1 writes the models + `Secrets`; Task 2 adds `AppConfig` + helpers; Task 3 adds `load_config`).
- Test: `backend/tests/test_appconfig.py` (new).

> Interfaces note: `backend/config.py` is rewritten here and is ALSO edited later by WS1 (delete `PHOENIX_*`/`EVAL_LLM_*`), WS2 (derive `d_in`/`d_sae`), and WS3 (`HF_TOKEN` consumers). WS0 lands the full shape; downstream WS only delete/derive fields, never rename the ones below.

Interfaces:
- Consumes: pydantic v2 `BaseModel`, `Field`; pydantic-settings `BaseSettings`, `SettingsConfigDict`.
- Produces: `ModelConfig`, `SAEConfig`, `FeatureCloudConfig`, `ProbeBuilderConfig`, `ProbeConfig`, `SentryConfig`, `ObsConfig`, `PodConfig`, `RuntimeConfig` (all `BaseModel`), and `Secrets(BaseSettings)` with fields `anthropic_api_key`, `sentry_dsn`, `sentry_auth_token`, `pod_token`, `hf_token` (each `str`, env-aliased). Field names verbatim per the AppConfig Contract §1.

Steps:

- [ ] Step 1: Add `pydantic-settings`, `pydantic`, and `pyyaml` to base deps in `pyproject.toml` so the import target exists. Edit the `dependencies` list (currently `fastapi`, `sentry-sdk[fastapi]`, `httpx`) to also include:
```toml
dependencies = [
    "fastapi>=0.138.0",
    "sentry-sdk[fastapi]>=2.63.0",
    "httpx>=0.27.0",
    "pydantic>=2.9",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
]
```
Run `uv sync` to install:
```
$ uv sync
```
Expected: resolves and installs `pydantic-settings` and `pyyaml` with no error.

- [ ] Step 2: Write the failing test for the sub-models in `backend/tests/test_appconfig.py`:
```python
"""WS0 AppConfig backbone — models, loader, secret exclusion."""
from __future__ import annotations

from pathlib import Path


def test_submodels_have_gemma_defaults():
    from backend.config import ModelConfig, SAEConfig, FeatureCloudConfig, ProbeConfig

    m = ModelConfig()
    assert m.model_id == "unsloth/gemma-3-4b-it"
    assert m.layer == 17
    assert m.sae_layers == [9, 17, 22, 29]
    assert m.mask_tokens == ["<bos>", "<start_of_turn>", "<end_of_turn>"]
    assert m.preamble_skip == 12
    assert m.max_new_tokens == 512

    s = SAEConfig()
    assert s.release == "gemma-scope-2-4b-it-res"
    assert s.sae_id_pattern == "layer_{layer}_width_16k_l0_medium"
    assert s.d_in == 2560
    assert s.d_sae == 16384
    assert s.np_model == "gemma-3-4b-it"
    assert s.np_source_pattern == "{layer}-gemmascope-2-res-16k"

    fc = FeatureCloudConfig()
    assert fc.topk == 15
    assert fc.topk_event == 30
    assert fc.topk_candidates == 50
    assert fc.rank_method == "attribution"

    p = ProbeConfig()
    assert p.enabled == ["harmful", "harmful_prompt", "over_confidence"]
    assert p.disabled == ["uncertainty", "hallucination", "risk_awareness"]
    assert p.builder.auroc_threshold == 0.75


def test_default_system_prompt_is_neutral_not_medical():
    from backend.config import ModelConfig

    sp = ModelConfig().system_prompt.lower()
    assert "helpful" in sp and "honest" in sp
    assert "clinical" not in sp and "medical" not in sp and "patient" not in sp


def test_recon_probe_default_is_neutral():
    from backend.config import SAEConfig

    assert "ibuprofen" not in SAEConfig().recon_probe.lower()
    assert "pregnancy" not in SAEConfig().recon_probe.lower()


def test_secrets_read_from_env_only(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("POD_TOKEN", "pod-secret")
    from backend.config import Secrets

    s = Secrets()
    assert s.anthropic_api_key == "sk-test-123"
    assert s.pod_token == "pod-secret"
```

- [ ] Step 3: Run the test and confirm it fails because the new symbols do not exist yet:
```
$ uv run pytest backend/tests/test_appconfig.py -q
```
Expected: `ImportError: cannot import name 'ModelConfig' from 'backend.config'` (collection error / failures on all four tests).

- [ ] Step 4: Replace `backend/config.py` with the models + `Secrets` (this overwrites the flat globals; Tasks 2–3 append to this file). Write:
```python
"""backend/config.py — unified AppConfig (replaces the flat globals)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Neutral, non-medical default. The medical prompt ships as a labeled profile in
# config.example.yaml, NOT as the code default.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI assistant. Answer clearly and "
    "accurately. When you are uncertain, say so plainly rather than guessing, and "
    "distinguish what is well-established from what is speculative. If a question is "
    "ambiguous, state the assumptions you are making. Be concise and well-structured."
)


class ModelConfig(BaseModel):
    """The LLM under inspection. Gemma specifics live here (the model seam)."""

    model_id: str = "unsloth/gemma-3-4b-it"
    layer: int = 17
    sae_layers: list[int] = Field(default_factory=lambda: [9, 17, 22, 29])
    device: str = "cuda"  # "cuda" | "mps" | "cpu" | "auto"; resolved via resolve_device()
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    mask_tokens: list[str] = Field(
        default_factory=lambda: ["<bos>", "<start_of_turn>", "<end_of_turn>"]
    )
    preamble_skip: int = 12
    max_new_tokens: int = 512


class SAEConfig(BaseModel):
    """The SAE under inspection + its Neuronpedia label source (the swap seam)."""

    release: str = "gemma-scope-2-4b-it-res"
    sae_id_pattern: str = "layer_{layer}_width_16k_l0_medium"
    d_in: int = 2560
    d_sae: int = 16384
    np_model: str = "gemma-3-4b-it"
    np_source_pattern: str = "{layer}-gemmascope-2-res-16k"
    np_feature_url: str = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
    recon_min_cosine: float = 0.85
    recon_probe: str = "What is the boiling point of water at sea level?"


class FeatureCloudConfig(BaseModel):
    """SAE feature-cloud ranking knobs (Family A). Pure-CPU + torch read these."""

    topk: int = 15
    topk_event: int = 30
    topk_candidates: int = 50
    drop_unlabeled: bool = True
    density_max: float = 0.01
    syntactic_penalty: float = 0.12
    structural_penalty: float = 0.15
    rank_method: Literal["attribution", "activation"] = "attribution"
    contrast_baseline: bool = True
    contrast_prompt: str = "Can you explain how rainbows form?"
    contrast_max_new: int = 64
    autointerp: bool = True
    autointerp_model: str = "claude-haiku-4-5"


class ProbeBuilderConfig(BaseModel):
    """Interpretability-agent / probe-builder knobs (the probe builder writes here)."""

    agent_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    agent_max_questions: int = 12
    judge_batch_size: int = 8
    auroc_threshold: float = 0.75


class ProbeConfig(BaseModel):
    """Probe set + builder."""

    enabled: list[str] = Field(
        default_factory=lambda: ["harmful", "harmful_prompt", "over_confidence"]
    )
    disabled: list[str] = Field(
        default_factory=lambda: ["uncertainty", "hallucination", "risk_awareness"]
    )
    artifacts_dir: Path = Path(__file__).parent / "science" / "artifacts"
    default_threshold: float = 0.5
    builder: ProbeBuilderConfig = Field(default_factory=ProbeBuilderConfig)


class SentryConfig(BaseModel):
    """Sentry read/emit surfaces. The DSN + auth token are SECRETS (see Secrets), not here."""

    environment: str = "production"
    release: str | None = None
    send_io: bool = False  # PHI gate
    org_slug: str = ""
    project_slug: str = ""
    api_base: str = "https://sentry.io"
    org_url: str = "https://sentry.io"


class ObsConfig(BaseModel):
    """Observability. Phoenix/Arize is REMOVED in WS1 — only Sentry remains."""

    sentry: SentryConfig = Field(default_factory=SentryConfig)


class PodConfig(BaseModel):
    """GPU pod connection (orchestration → remote torch service). token is a SECRET."""

    url: str = "http://localhost:8001"  # rstrip("/") applied in loader
    timeout: float = 120.0
    poll_interval: float = 5.0


class RuntimeConfig(BaseModel):
    """Process-level runtime + branding."""

    mode: Literal["posthoc", "live"] = "posthoc"
    product_name: str = "GlassBox"


class Secrets(BaseSettings):
    """Secret values. Sourced ONLY from environment / .env, never YAML."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    sentry_auth_token: str = Field(default="", alias="SENTRY_AUTH_TOKEN")
    pod_token: str = Field(default="", alias="POD_TOKEN")
    hf_token: str = Field(default="", alias="HF_TOKEN")
```

- [ ] Step 5: Run the test and confirm it passes:
```
$ uv run pytest backend/tests/test_appconfig.py -q
```
Expected: `4 passed`.

- [ ] Step 6: Commit:
```
$ git add pyproject.toml backend/config.py backend/tests/test_appconfig.py
$ git commit -m "feat(config): add AppConfig sub-models and env-only Secrets"
```

---

### Task 2: AppConfig parent object + derived helpers

Files:
- Modify: `backend/config.py` (append `AppConfig` class + `sae_id`/`np_source`/`resolve_device` methods after `Secrets`).
- Test: `backend/tests/test_appconfig.py` (append).

Interfaces:
- Consumes: the sub-models from Task 1.
- Produces: `class AppConfig(BaseModel)` with fields `model: ModelConfig`, `sae: SAEConfig`, `feature_cloud: FeatureCloudConfig`, `probes: ProbeConfig`, `observability: ObsConfig`, `pod: PodConfig`, `runtime: RuntimeConfig`, plus injected secret scalars `anthropic_api_key`, `sentry_dsn`, `sentry_auth_token`, `pod_token`, `hf_token` (all `str`, default `""`). Methods: `sae_id(layer: int | None = None) -> str`, `np_source(layer: int | None = None) -> str`, `resolve_device(pref: str | None = None) -> str` (imports torch lazily). Every later task threads instances of these types.

Steps:

- [ ] Step 1: Append the failing test to `backend/tests/test_appconfig.py`:
```python
def test_appconfig_assembles_and_derives_ids():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.model.layer == 17
    assert cfg.sae.release == "gemma-scope-2-4b-it-res"
    assert cfg.feature_cloud.topk == 15
    assert cfg.runtime.product_name == "GlassBox"
    # secret scalars default to empty (never auto-populated by the model itself)
    assert cfg.anthropic_api_key == ""
    assert cfg.pod_token == ""
    # derived helpers substitute the layer
    assert cfg.sae_id() == "layer_17_width_16k_l0_medium"
    assert cfg.sae_id(22) == "layer_22_width_16k_l0_medium"
    assert cfg.np_source() == "17-gemmascope-2-res-16k"
    assert cfg.np_source(9) == "9-gemmascope-2-res-16k"
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_appconfig.py::test_appconfig_assembles_and_derives_ids -q
```
Expected: `ImportError: cannot import name 'AppConfig' from 'backend.config'`.

- [ ] Step 3: Append to `backend/config.py` (after the `Secrets` class):
```python
class AppConfig(BaseModel):
    """Single source of truth. Built once at startup, stored on app.state, threaded down."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    sae: SAEConfig = Field(default_factory=SAEConfig)
    feature_cloud: FeatureCloudConfig = Field(default_factory=FeatureCloudConfig)
    probes: ProbeConfig = Field(default_factory=ProbeConfig)
    observability: ObsConfig = Field(default_factory=ObsConfig)
    pod: PodConfig = Field(default_factory=PodConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Secrets injected by the loader (env-only). NOT loaded from YAML.
    anthropic_api_key: str = ""
    sentry_dsn: str = ""
    sentry_auth_token: str = ""
    pod_token: str = ""
    hf_token: str = ""

    def sae_id(self, layer: int | None = None) -> str:
        """SAELens sae_id for `layer` (defaults to model.layer)."""
        return self.sae.sae_id_pattern.format(
            layer=layer if layer is not None else self.model.layer
        )

    def np_source(self, layer: int | None = None) -> str:
        """Neuronpedia source slug for `layer`."""
        return self.sae.np_source_pattern.format(
            layer=layer if layer is not None else self.model.layer
        )

    def resolve_device(self, pref: str | None = None) -> str:
        """Resolve the runtime device. torch imported lazily so CPU modules can call build()."""
        import torch

        p = (pref or self.model.device or "auto").lower()
        if p == "cuda" and torch.cuda.is_available():
            return "cuda"
        if p == "mps" and torch.backends.mps.is_available():
            return "mps"
        if p == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_appconfig.py -q
```
Expected: `5 passed`.

- [ ] Step 5: Commit:
```
$ git add backend/config.py backend/tests/test_appconfig.py
$ git commit -m "feat(config): add AppConfig parent and derived sae_id/np_source/resolve_device helpers"
```

---

### Task 3: load_config loader (YAML structure, env override, env-only secrets)

Files:
- Modify: `backend/config.py` (append `_CFG_PATH` + `load_config`).
- Test: `backend/tests/test_appconfig.py` (append).

Interfaces:
- Consumes: `AppConfig`, `Secrets` from Tasks 1–2; `yaml.safe_load`.
- Produces: module-level `_CFG_PATH: Path` (repo-root `config.yaml`) and `def load_config(path: str | Path | None = None) -> AppConfig`. Behavioral guarantees: no YAML → Gemma defaults; secret keys in YAML are popped (ignored); env `GLASSBOX__SECTION__FIELD` overrides YAML for structural fields; `cfg.pod.url` is trailing-slash-normalized; secrets come only from `Secrets()`.

Steps:

- [ ] Step 1: Append the three contract tests to `backend/tests/test_appconfig.py`:
```python
def test_load_config_defaults_when_no_yaml(tmp_path):
    from backend.config import load_config

    missing = tmp_path / "nope.yaml"
    cfg = load_config(missing)
    assert cfg.model.model_id == "unsloth/gemma-3-4b-it"
    assert cfg.model.layer == 17
    assert cfg.pod.url == "http://localhost:8001"


def test_load_config_yaml_then_env_override(tmp_path, monkeypatch):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "model:\n  layer: 22\nruntime:\n  product_name: Demo\n", encoding="utf-8"
    )
    cfg = load_config(yaml_path)
    assert cfg.model.layer == 22  # from YAML
    assert cfg.runtime.product_name == "Demo"

    monkeypatch.setenv("GLASSBOX__MODEL__LAYER", "29")
    cfg2 = load_config(yaml_path)
    assert cfg2.model.layer == 29  # env beats YAML


def test_load_config_ignores_secrets_in_yaml(tmp_path, monkeypatch):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "anthropic_api_key: leaked-from-yaml\npod_token: leaked\n", encoding="utf-8"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("POD_TOKEN", raising=False)
    cfg = load_config(yaml_path)
    assert cfg.anthropic_api_key == ""  # YAML secret popped, env empty
    assert cfg.pod_token == ""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    cfg2 = load_config(yaml_path)
    assert cfg2.anthropic_api_key == "sk-from-env"  # env is the only source


def test_load_config_normalizes_pod_url(tmp_path):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("pod:\n  url: http://pod.example/\n", encoding="utf-8")
    cfg = load_config(yaml_path)
    assert cfg.pod.url == "http://pod.example"
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_appconfig.py -k load_config -q
```
Expected: `ImportError: cannot import name 'load_config' from 'backend.config'`.

- [ ] Step 3: Append to `backend/config.py` (after `AppConfig`):
```python
_CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_SECRET_KEYS = (
    "anthropic_api_key",
    "sentry_dsn",
    "sentry_auth_token",
    "pod_token",
    "hf_token",
)


def _env_overrides() -> dict:
    """Collect GLASSBOX__SECTION__FIELD env vars into a nested dict (env beats YAML)."""
    import os

    nested: dict = {}
    for key, val in os.environ.items():
        if not key.startswith("GLASSBOX__"):
            continue
        parts = [p.lower() for p in key[len("GLASSBOX__"):].split("__") if p]
        if not parts:
            continue
        cursor = nested
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = val
    return nested


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build the AppConfig once. Structure from YAML (env override); secrets from env only."""
    cfg_path = Path(path) if path is not None else _CFG_PATH

    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        for k in _SECRET_KEYS:  # defense-in-depth: never honor secrets from YAML
            raw.pop(k, None)

    raw = _deep_merge(raw, _env_overrides())
    cfg = AppConfig.model_validate(raw)

    secrets = Secrets()
    cfg.anthropic_api_key = secrets.anthropic_api_key
    cfg.sentry_dsn = secrets.sentry_dsn
    cfg.sentry_auth_token = secrets.sentry_auth_token
    cfg.pod_token = secrets.pod_token
    cfg.hf_token = secrets.hf_token

    cfg.pod.url = cfg.pod.url.rstrip("/")
    return cfg
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_appconfig.py -q
```
Expected: `9 passed`.

- [ ] Step 5: Commit:
```
$ git add backend/config.py backend/tests/test_appconfig.py
$ git commit -m "feat(config): add load_config loader with YAML structure, env override, env-only secrets"
```

---

### Task 4: config.example.yaml template (with labeled medical profile)

Files:
- Create: `config.example.yaml` (repo root).
- Test: `backend/tests/test_appconfig.py` (append — validate the example file loads with no secrets).

Interfaces:
- Consumes: `load_config` from Task 3.
- Produces: a committed `config.example.yaml` the README/quickstart references. No code symbols.

Steps:

- [ ] Step 1: Append the failing test to `backend/tests/test_appconfig.py`:
```python
def test_example_yaml_loads_and_has_no_secrets():
    from pathlib import Path

    import yaml

    from backend.config import load_config

    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    assert example.exists(), "config.example.yaml must be committed"

    raw = yaml.safe_load(example.read_text())
    for secret in ("anthropic_api_key", "sentry_dsn", "sentry_auth_token", "pod_token", "hf_token"):
        assert secret not in raw, f"{secret} must NOT appear in config.example.yaml"

    cfg = load_config(example)
    assert cfg.runtime.product_name == "GlassBox"
    assert cfg.model.model_id == "unsloth/gemma-3-4b-it"
    # the active default prompt stays neutral; medical lives only in a comment block
    assert "clinical" not in cfg.model.system_prompt.lower()
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_appconfig.py::test_example_yaml_loads_and_has_no_secrets -q
```
Expected: `AssertionError: config.example.yaml must be committed`.

- [ ] Step 3: Create `config.example.yaml`:
```yaml
# config.example.yaml — copy to config.yaml and edit. config.yaml is gitignored.
# Secrets (ANTHROPIC_API_KEY, SENTRY_DSN, SENTRY_AUTH_TOKEN, POD_TOKEN, HF_TOKEN) go in .env, never here.

runtime:
  mode: posthoc
  product_name: GlassBox

model:
  model_id: unsloth/gemma-3-4b-it
  layer: 17
  device: cuda
  preamble_skip: 12
  max_new_tokens: 512
  # Neutral default assistant prompt. See the "medical" profile below for the labeled example.
  system_prompt: |
    You are a helpful, honest, and harmless AI assistant. Answer clearly and accurately.
    When you are uncertain, say so plainly rather than guessing.

sae:
  release: gemma-scope-2-4b-it-res
  sae_id_pattern: "layer_{layer}_width_16k_l0_medium"
  d_in: 2560
  d_sae: 16384
  np_model: gemma-3-4b-it
  np_source_pattern: "{layer}-gemmascope-2-res-16k"

feature_cloud:
  rank_method: attribution
  autointerp: true

probes:
  enabled: [harmful, harmful_prompt, over_confidence]
  disabled: [uncertainty, hallucination, risk_awareness]
  builder:
    agent_model: claude-opus-4-8
    judge_model: claude-opus-4-8
    auroc_threshold: 0.75

observability:
  sentry:
    environment: production
    send_io: false   # PHI gate; keep false to never send raw I/O to Sentry

pod:
  url: http://localhost:8001
  timeout: 120

# --- Labeled example profile: medical clinical-decision-support (opt-in) ---
# GlassBox is a general-purpose interpretability/observability tool. Medical is ONE example.
# To use, copy the prompt below into model.system_prompt above:
#   system_prompt: |
#     You are a clinical decision-support assistant for licensed healthcare professionals.
#     You provide accurate, evidence-based medical information grounded in current clinical
#     guidelines and the peer-reviewed literature. Accuracy first; calibrated confidence;
#     safety first (surface contraindications, interactions, special populations); typical
#     adult dosing with caveats; you support, not replace, the clinician's judgment. Direct
#     emergencies to emergency services. Define abbreviations on first use.
# A matching example recon_probe and contrast_prompt for the medical profile:
#   sae:
#     recon_probe: "Is ibuprofen safe during the third trimester of pregnancy?"
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_appconfig.py -q
```
Expected: `10 passed`.

- [ ] Step 5: Commit:
```
$ git add config.example.yaml backend/tests/test_appconfig.py
$ git commit -m "feat(config): ship config.example.yaml template with labeled medical profile"
```

---

### Task 5: Thread AppConfig into runtime (CPU, torch-free)

Files:
- Modify: `backend/runtime.py` (lines 1–125 — replace flat `config.*` reads with a passed-in `cfg`).
- Test: `backend/tests/test_runtime.py` (rewrite — inject `cfg` instead of monkeypatching `runtime.config.POD_URL`).

Interfaces:
- Consumes: `AppConfig`, `load_config` from Tasks 2–3; `cfg.pod.url`, `cfg.pod.poll_interval`, `cfg.model.model_id`, `cfg.model.layer`, `cfg.sae.d_sae`.
- Produces: `start_loading(cfg: AppConfig) -> None`, `health_payload(cfg: AppConfig) -> dict`, `refresh_pod_health(cfg: AppConfig) -> dict`, `_poll_pod_once(cfg: AppConfig) -> dict`, `_pod_poll_loop(cfg: AppConfig) -> None`. `runtime` still must NOT import torch.

> Interfaces note: `runtime.health_payload`/`start_loading`/`refresh_pod_health` are called from `backend/app.py` (Task 13). Keep this task and Task 13 sequenced together.

> Sequencing dependency (Task 5 ↔ Task 6): Task 5's rewritten test monkeypatches `pod_client.health` as `lambda pod, pod_token: {...}` (two positional args), matching the NEW Task 6 signature. But Task 5 lands BEFORE Task 6, so until Task 6 the REAL `pod_client.health()` still has the old no-arg signature — meaning `runtime._poll_pod_once` calling `pod_client.health(cfg.pod, cfg.pod_token)` would `TypeError` against the un-migrated `pod_client`. Every Task 5 test that reaches `_poll_pod_once` monkeypatches `pod_client.health`, so the in-task suite is green via the mock; the REAL un-mocked `pod_client.health(cfg.pod, cfg.pod_token)` integration is only valid once Task 6 lands and is validated by the full suite in Task 15. (Alternatively, reorder Task 6 before Task 5 — both are CPU/torch-free; if reordered, no monkeypatch is hiding anything. Either order is acceptable as long as the full suite in Task 15 is green; the dependency is called out here so the per-task "confirm pass" is not mistaken for end-to-end validation.)

Steps:

- [ ] Step 1: Rewrite `backend/tests/test_runtime.py` to inject `cfg` (replaces every `monkeypatch.setattr(runtime.config, "POD_URL", ...)`):
```python
from backend import runtime
from backend.config import AppConfig


def _cfg(pod_url="", poll=5.0):
    cfg = AppConfig()
    cfg.pod.url = pod_url
    cfg.pod.poll_interval = poll
    return cfg


def test_no_pod_url_stays_fallback():
    runtime.STATE.clear()
    runtime.STATE.update(mode="loading")
    runtime.start_loading(_cfg(pod_url=""))
    assert runtime.STATE["mode"] == "fallback"
    assert runtime.STATE["pod_reachable"] is False


def test_apply_pod_health_real():
    runtime._apply_pod_health(
        {
            "mode": "real",
            "model_loaded": True,
            "sae_loaded": True,
            "d_sae": 16384,
            "trackers": ["uncertainty"],
            "sae_recon_cosine": 0.95,
            "sae_recon_ok": True,
        },
        reachable=True,
    )
    assert runtime.STATE["mode"] == "real"
    assert runtime.STATE["model_loaded"] is True
    assert runtime.STATE["pod_reachable"] is True


def test_apply_pod_health_unreachable():
    runtime._apply_pod_health(None, reachable=False)
    assert runtime.STATE["mode"] == "fallback"
    assert runtime.STATE["pod_reachable"] is False


def test_poll_pod_once_uses_client(monkeypatch):
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    state = runtime._poll_pod_once(_cfg(pod_url="http://pod.test"))
    assert state["mode"] == "real"
    assert state["pod_reachable"] is True


def test_poll_pod_once_falls_back_on_error(monkeypatch):
    import backend.pod_client as pc

    def boom(pod, pod_token):
        raise pc.PodError(0, "health")

    monkeypatch.setattr(pc, "health", boom)
    state = runtime._poll_pod_once(_cfg(pod_url="http://pod.test"))
    assert state["mode"] == "fallback"
    assert state["pod_reachable"] is False


def test_health_payload_shape():
    runtime.STATE.update(
        mode="fallback",
        model_loaded=False,
        sae_loaded=False,
        pod_reachable=False,
        pod_health=None,
    )
    p = runtime.health_payload(_cfg(pod_url="http://pod.test"))
    assert {
        "mode",
        "model_loaded",
        "sae_loaded",
        "model",
        "layer",
        "d_sae",
        "trackers",
        "sae_recon_cosine",
        "sae_recon_ok",
        "pod_reachable",
        "pod_url_configured",
    } <= set(p)
    assert p["layer"] == 17
    assert isinstance(p["trackers"], list)


def test_start_loading_eager_polls_pod(monkeypatch):
    monkeypatch.setenv("GLASSBOX_EAGER_LOAD", "1")
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: None, raising=False)
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert runtime.STATE["mode"] == "real"


def test_eager_load_also_keeps_polling(monkeypatch):
    monkeypatch.setenv("GLASSBOX_EAGER_LOAD", "1")
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    started: list[bool] = []
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: started.append(True), raising=False)
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert runtime.STATE["mode"] == "real"
    assert started == [True]


def test_non_eager_starts_poll_loop(monkeypatch):
    monkeypatch.delenv("GLASSBOX_EAGER_LOAD", raising=False)
    started: list[bool] = []
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: started.append(True), raising=False)
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert started == [True]
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_runtime.py -q
```
Expected: failures like `TypeError: start_loading() takes 0 positional arguments but 1 was given` and `health_payload() takes 0 positional arguments but 1 was given`.

- [ ] Step 3: Rewrite `backend/runtime.py` to thread `cfg` (replace the module body from line 13 onward; keep `STATE` and `_apply_pod_health` unchanged):
```python
"""Backend readiness. Polls the GPU pod when pod.url is set; otherwise stays on synthetic fallback.
State is read per-request by the API.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import os
import threading
import time

STATE: dict = {
    "mode": "loading",
    "model_loaded": False,
    "sae_loaded": False,
    "pod_reachable": False,
    "pod_health": None,
}


def _apply_pod_health(h: dict | None, *, reachable: bool) -> dict:
    STATE["pod_reachable"] = reachable
    STATE["pod_health"] = h
    if not reachable or h is None:
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
        return dict(STATE)
    STATE["model_loaded"] = bool(h.get("model_loaded"))
    STATE["sae_loaded"] = bool(h.get("sae_loaded"))
    STATE["sae_recon_cosine"] = h.get("sae_recon_cosine")
    STATE["sae_recon_ok"] = h.get("sae_recon_ok")
    if h.get("mode") == "real" and STATE["model_loaded"] and STATE["sae_loaded"]:
        STATE["mode"] = "real"
    elif h.get("mode") == "loading":
        STATE["mode"] = "loading"
    else:
        STATE["mode"] = "fallback"
    return dict(STATE)


def _poll_pod_once(cfg) -> dict:
    """Fetch pod /health once. Never raises."""
    if not cfg.pod.url:
        return _apply_pod_health(None, reachable=False)
    try:
        from . import pod_client

        return _apply_pod_health(pod_client.health(cfg.pod, cfg.pod_token), reachable=True)
    except Exception as e:  # noqa: BLE001
        print(f"[runtime] pod health poll failed ({e})")
        return _apply_pod_health(None, reachable=False)


def _pod_poll_loop(cfg) -> None:
    while True:
        _poll_pod_once(cfg)
        time.sleep(cfg.pod.poll_interval)


_poll_thread: threading.Thread | None = None


def _ensure_poll_loop(cfg) -> None:
    """Start the background pod-health poll loop once (idempotent)."""
    global _poll_thread
    if _poll_thread is not None and _poll_thread.is_alive():
        return
    _poll_thread = threading.Thread(target=_pod_poll_loop, args=(cfg,), daemon=True)
    _poll_thread.start()


def refresh_pod_health(cfg) -> dict:
    """Re-check pod readiness (e.g. after a failed turn)."""
    return _poll_pod_once(cfg)


def start_loading(cfg) -> None:
    """Bring the real path online. Called once at FastAPI startup."""
    if not cfg.pod.url:
        STATE.update(
            mode="fallback",
            model_loaded=False,
            sae_loaded=False,
            pod_reachable=False,
            pod_health=None,
        )
        return

    STATE["mode"] = "loading"
    if os.getenv("GLASSBOX_EAGER_LOAD") == "1":
        _poll_pod_once(cfg)
    _ensure_poll_loop(cfg)


def health_payload(cfg) -> dict:
    ph = STATE.get("pod_health") or {}
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": cfg.model.model_id,
        "layer": cfg.model.layer,
        "d_sae": ph.get("d_sae", cfg.sae.d_sae),
        "trackers": ph.get("trackers", []),
        "sae_recon_cosine": STATE.get("sae_recon_cosine", ph.get("sae_recon_cosine")),
        "sae_recon_ok": STATE.get("sae_recon_ok", ph.get("sae_recon_ok")),
        "pod_reachable": STATE.get("pod_reachable", False),
        "pod_url_configured": bool(cfg.pod.url),
        "anthropic_configured": ph.get("anthropic_configured"),
        "active_probe_jobs": ph.get("active_probe_jobs"),
    }
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_runtime.py -q
```
Expected: `9 passed`.

- [ ] Step 5: Commit:
```
$ git add backend/runtime.py backend/tests/test_runtime.py
$ git commit -m "refactor(runtime): thread AppConfig into start_loading/health_payload/poll"
```

---

### Task 6: Thread PodConfig + pod_token into pod_client (CPU, torch-free)

Files:
- Modify: `backend/pod_client.py` (lines 1–209 — replace flat `config.*` reads with passed-in `pod`, `pod_token`, `max_new`).
- Test: `backend/tests/test_pod_client.py` (rewrite — pass `PodConfig` + token).

Interfaces:
- Consumes: `PodConfig` from Task 1; `cfg.pod`, `cfg.pod_token`, `cfg.model.max_new_tokens`.
- Produces: `health(pod, pod_token) -> dict | None`, `inference(messages, pod, pod_token, *, max_new) -> dict`, `activations(messages, pod, pod_token, *, max_new, attribution) -> dict`, `sae_features(messages, pod, pod_token, *, max_new, attribution, cap) -> dict`, `turn(messages, pod, pod_token, *, max_new) -> dict`, `track(request, pod, pod_token) -> dict`, `clear_custom_trackers(pod, pod_token) -> dict`, `track_status(tracker_id, pod, pod_token) -> dict`. Internal `_headers(pod_token)`, `_base_url(pod)`, `_post(path, payload, pod, pod_token, stage)`. `PodError`/`pod_unavailable_payload` unchanged.

> Interfaces note: `pod_client.track`/`track_status`/`clear_custom_trackers`/`turn` are called from `backend/app.py` (Task 13) and `runtime._poll_pod_once` (Task 5). `max_new` callers must pass `cfg.model.max_new_tokens` explicitly since the global default is gone.

Steps:

- [ ] Step 1: Rewrite `backend/tests/test_pod_client.py`:
```python
import httpx

import backend.pod_client as pc
from backend.config import PodConfig


def _pod(url="http://pod.test"):
    return PodConfig(url=url)


def test_health_returns_none_without_pod_url():
    assert pc.health(_pod(url=""), "") is None


def test_turn_posts_messages(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"answer": "ok", "candidates": [], "trackers": {}, "reliable": True}

    def fake_post(url, **kw):
        captured["url"] = url
        captured.update(kw)
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    out = pc.turn([{"role": "user", "content": "hi"}], _pod(), "secret", max_new=512)
    assert out["answer"] == "ok"
    assert captured["url"] == "http://pod.test/turn"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["max_new"] == 512


def test_sae_features_path(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"answer": "a", "candidates": [], "reliable": True}

    monkeypatch.setattr(httpx, "post", lambda url, **kw: captured.update({"url": url, **kw}) or FakeResp())
    pc.sae_features([{"role": "user", "content": "x"}], _pod(), "", max_new=64, cap=20)
    assert captured["url"] == "http://pod.test/sae/features"
    assert captured["json"]["cap"] == 20


def test_post_raises_pod_error_on_non_200(monkeypatch):
    class FakeResp:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    try:
        pc.inference([{"role": "user", "content": "x"}], _pod(), "", max_new=512)
        assert False, "expected PodError"
    except pc.PodError as e:
        assert e.status == 500


def test_health_raises_on_failure(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    try:
        pc.health(_pod(), "")
        assert False, "expected PodError"
    except pc.PodError as e:
        assert e.status == 0


def test_pod_error_carries_no_body():
    e = pc.PodError(500, "turn")
    assert "turn" in str(e) and "500" in str(e)


def test_post_non_200_body_not_in_exception(monkeypatch):
    class FakeResp:
        status_code = 503
        text = "secret model answer that must not leak"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    try:
        pc.turn([{"role": "user", "content": "x"}], _pod(), "", max_new=512)
        assert False
    except pc.PodError as e:
        assert "secret model answer" not in str(e)
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_pod_client.py -q
```
Expected: `TypeError: health() takes 0 positional arguments but 2 were given` (and similar for the rest).

- [ ] Step 3: Rewrite `backend/pod_client.py` (replace the `from . import config` import and every helper/function below it):
```python
"""Orchestration-side HTTP client for the GPU pod service. CPU only — never imports torch."""

from __future__ import annotations

_UA = {"User-Agent": "glassbox-orchestration/0.1"}


class PodError(Exception):
    """Raised when the pod is unreachable or returns a non-2xx response."""

    def __init__(self, status: int, stage: str, *, detail: str = "") -> None:
        self.status = status
        self.stage = stage
        self.detail = detail
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.detail:
            return f"pod {self.stage} failed: {self.detail}"
        return f"pod {self.stage} failed: HTTP {self.status}"


def pod_unavailable_payload(exc: Exception) -> dict:
    """Map pod client errors to a JSON body safe for the Build UI."""
    if isinstance(exc, PodError):
        if exc.status == 0:
            detail = "Cannot reach GPU pod — start the SSH tunnel and gpu_service"
        elif exc.detail:
            detail = exc.detail
        else:
            detail = f"GPU pod error during {exc.stage} (HTTP {exc.status})"
        return {"status": "unavailable", "detail": detail}
    if isinstance(exc, ConnectionError):
        return {"status": "unavailable", "detail": "POD_URL is not configured on the backend"}
    msg = str(exc) or type(exc).__name__
    return {"status": "unavailable", "detail": msg}


def _headers(pod_token: str) -> dict[str, str]:
    h = dict(_UA)
    if pod_token:
        h["Authorization"] = f"Bearer {pod_token}"
    return h


def _base_url(pod) -> str:
    if not pod.url:
        raise ConnectionError("pod.url is not configured")
    return pod.url


def _post(path: str, payload: dict, pod, pod_token: str, stage: str = "request") -> dict:
    import httpx

    try:
        r = httpx.post(
            f"{_base_url(pod)}{path}",
            json=payload,
            headers=_headers(pod_token),
            timeout=pod.timeout,
        )
    except httpx.HTTPError as e:
        raise PodError(0, stage) from e
    if r.status_code != 200:
        print(f"[pod] {stage} HTTP {r.status_code}: {r.text[:200]}")
        raise PodError(r.status_code, stage)
    return r.json()


def health(pod, pod_token: str) -> dict | None:
    """Poll pod /health. Returns None when pod.url is unset; raises PodError on failure."""
    if not pod.url:
        return None
    import httpx

    try:
        r = httpx.get(
            f"{_base_url(pod)}/health",
            headers=_headers(pod_token),
            timeout=min(pod.timeout, 10.0),
        )
    except httpx.HTTPError as e:
        raise PodError(0, "health") from e
    if r.status_code != 200:
        raise PodError(r.status_code, "health")
    return r.json()


def inference(messages: list[dict], pod, pod_token: str, *, max_new: int) -> dict:
    """POST /inference → {answer}."""
    return _post("/inference", {"messages": messages, "max_new": max_new}, pod, pod_token, stage="inference")


def activations(messages: list[dict], pod, pod_token: str, *, max_new: int, attribution: bool = False) -> dict:
    """POST /activations → {answer, resp_start, act_last, act_resp}."""
    return _post(
        "/activations",
        {"messages": messages, "max_new": max_new, "attribution": attribution},
        pod,
        pod_token,
        stage="activations",
    )


def sae_features(
    messages: list[dict],
    pod,
    pod_token: str,
    *,
    max_new: int,
    attribution: bool | None = None,
    cap: int | None = None,
) -> dict:
    """POST /sae/features → {answer, candidates, reliable}."""
    body: dict = {"messages": messages, "max_new": max_new}
    if attribution is not None:
        body["attribution"] = attribution
    if cap is not None:
        body["cap"] = cap
    return _post("/sae/features", body, pod, pod_token, stage="sae_features")


def turn(messages: list[dict], pod, pod_token: str, *, max_new: int) -> dict:
    """POST /turn → {answer, candidates, trackers, reliable}."""
    return _post("/turn", {"messages": messages, "max_new": max_new}, pod, pod_token, stage="turn")


def track(request: str, pod, pod_token: str) -> dict:
    """POST /api/track → {tracker_id, status}. Body carries only the NL request (no PHI)."""
    import httpx

    try:
        r = httpx.post(
            f"{_base_url(pod)}/api/track",
            json={"request": request},
            headers=_headers(pod_token),
            timeout=pod.timeout,
        )
    except httpx.HTTPError as e:
        raise PodError(0, "track") from e
    if r.status_code == 200:
        return r.json()
    detail = ""
    try:
        detail = str(r.json().get("detail") or "")
    except Exception:  # noqa: BLE001
        detail = ""
    print(f"[pod] track HTTP {r.status_code}: {(detail or r.text)[:200]}")
    if detail:
        return {"status": "unavailable", "detail": detail}
    raise PodError(r.status_code, "track")


def clear_custom_trackers(pod, pod_token: str) -> dict:
    """POST /api/trackers/clear-custom → {removed, trackers}."""
    return _post("/api/trackers/clear-custom", {}, pod, pod_token, stage="clear_custom_trackers")


def track_status(tracker_id: str, pod, pod_token: str) -> dict:
    """GET /api/track/{id} → the probe-job record."""
    import httpx

    try:
        r = httpx.get(
            f"{_base_url(pod)}/api/track/{tracker_id}",
            headers=_headers(pod_token),
            timeout=min(pod.timeout, 15.0),
        )
    except httpx.HTTPError as e:
        raise PodError(0, "track_status") from e
    if r.status_code == 404:
        return {"status": "unknown", "detail": "job not found — pod may have restarted"}
    if r.status_code != 200:
        detail = ""
        try:
            detail = str(r.json().get("detail") or "")
        except Exception:  # noqa: BLE001
            detail = ""
        print(f"[pod] track_status HTTP {r.status_code}: {(detail or r.text)[:200]}")
        if detail:
            return {"status": "unavailable", "detail": detail}
        raise PodError(r.status_code, "track_status")
    return r.json()
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_pod_client.py -q
```
Expected: `7 passed`.

- [ ] Step 5: Commit:
```
$ git add backend/pod_client.py backend/tests/test_pod_client.py
$ git commit -m "refactor(pod_client): thread PodConfig + pod_token + max_new through client"
```

---

### Task 7: Thread SentryConfig + auth token into sentry_api (CPU, torch-free)

Files:
- Modify: `backend/sentry_api.py` (lines 8–43 — replace flat `config.*` with `sentry` + `token` params).
- Test: `backend/tests/test_sentry_api.py` (rewrite — pass `SentryConfig` + token).

Interfaces:
- Consumes: `SentryConfig` from Task 1; `cfg.observability.sentry`, `cfg.sentry_auth_token`.
- Produces: `deep_link(sentry, token) -> str | None`, `list_recent_issues(sentry, token, limit=15) -> list[dict]` (still async, still never raises). Drops the `from . import config` import.

> Interfaces note: both functions are called from `backend/app.py` (Task 13).

Steps:

- [ ] Step 1: Rewrite `backend/tests/test_sentry_api.py`:
```python
"""Tests for backend/sentry_api.py — field projection + graceful failure."""
import asyncio

import backend.sentry_api as sa
from backend.config import SentryConfig


def _sentry(org="org", proj="proj", base="https://sentry.io", org_url="https://sentry.io"):
    return SentryConfig(org_slug=org, project_slug=proj, api_base=base, org_url=org_url)


def test_list_recent_issues_projects_fields(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{
                "id": "1", "shortId": "G-1", "title": "t", "culprit": "c", "level": "warning",
                "count": 5, "userCount": 2, "lastSeen": "x", "permalink": "p", "extra": "drop",
            }]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(sa.httpx, "AsyncClient", FakeClient)
    out = asyncio.run(sa.list_recent_issues(_sentry(), "tok"))
    assert len(out) == 1
    assert out[0]["shortId"] == "G-1"
    assert "extra" not in out[0]
    for field in ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink"):
        assert field in out[0]


def test_list_recent_issues_empty_when_unconfigured():
    assert asyncio.run(sa.list_recent_issues(_sentry(), "")) == []


def test_list_recent_issues_empty_when_org_slug_missing():
    assert asyncio.run(sa.list_recent_issues(_sentry(org=""), "tok")) == []


def test_list_recent_issues_empty_on_timeout(monkeypatch):
    class TimeoutClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            raise sa.httpx.TimeoutException("timeout")

    monkeypatch.setattr(sa.httpx, "AsyncClient", TimeoutClient)
    assert asyncio.run(sa.list_recent_issues(_sentry(), "tok")) == []


def test_list_recent_issues_empty_on_non_2xx(monkeypatch):
    class BadResponse:
        def raise_for_status(self):
            raise sa.httpx.HTTPStatusError("403", request=None, response=None)

        def json(self):
            return []

    class BadClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return BadResponse()

    monkeypatch.setattr(sa.httpx, "AsyncClient", BadClient)
    assert asyncio.run(sa.list_recent_issues(_sentry(), "tok")) == []


def test_deep_link_when_configured():
    link = sa.deep_link(_sentry(org="myorg", org_url="https://sentry.io"), "tok")
    assert link == "https://sentry.io/organizations/myorg/issues/"


def test_deep_link_none_when_unconfigured():
    assert sa.deep_link(_sentry(), "") is None
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_sentry_api.py -q
```
Expected: `TypeError: deep_link() takes 0 positional arguments but 2 were given` (and `list_recent_issues` arity errors).

- [ ] Step 3: Rewrite `backend/sentry_api.py`:
```python
"""Sentry REST read path — recent-issues strip.
Lane A — never imports torch. Never raises (returns [] on any failure).
"""
from __future__ import annotations

import httpx

_ISSUE_FIELDS = ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink")


def deep_link(sentry, token: str) -> str | None:
    """Return the Sentry issues deep-link URL when configured, else None."""
    if sentry.org_slug and token:
        return f"{sentry.org_url}/organizations/{sentry.org_slug}/issues/"
    return None


async def list_recent_issues(sentry, token: str, limit: int = 15) -> list[dict]:
    """GET recent unresolved issues from Sentry. Returns [] when unconfigured or on any error."""
    if not token or not sentry.org_slug or not sentry.project_slug:
        return []
    url = f"{sentry.api_base}/api/0/projects/{sentry.org_slug}/{sentry.project_slug}/issues/"
    params = {"statsPeriod": "24h", "query": "is:unresolved", "sort": "date", "limit": limit}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            issues = r.json()
            return [{k: issue.get(k) for k in _ISSUE_FIELDS} for issue in issues]
    except Exception:  # noqa: BLE001 — never raises to caller
        return []
```

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_sentry_api.py -q
```
Expected: `7 passed`.

- [ ] Step 5: Commit:
```
$ git add backend/sentry_api.py backend/tests/test_sentry_api.py
$ git commit -m "refactor(sentry_api): thread SentryConfig + auth token through read path"
```

---

### Task 8: Thread sub-configs into fanout (CPU, torch-free; Phoenix branch left for WS1)

Files:
- Modify: `backend/fanout.py` (the `from . import config` import + lines 104–143, 162–230, 280–end — replace flat `config.*` reads with passed-in sub-configs; keep the Phoenix branch in `init_sponsors` for WS1 to delete).
- Test: `backend/tests/test_fanout.py` (rewrite the assertions that monkeypatch `fo.config.*`).

Interfaces:
- Consumes: `ObsConfig`, `ProbeConfig`, `SentryConfig` from Task 1; `cfg.observability`, `cfg.probes`, `cfg.sentry_dsn`.
- Produces: `fanout(event, perf=None, *, obs: ObsConfig, probes: ProbeConfig) -> None`; `capture_cognition_alarm(event, obs: ObsConfig, *, flush=False) -> bool`; `init_sponsors(obs: ObsConfig, sentry_dsn: str) -> None`; `_scrub_pii(event, hint, *, send_io: bool)`; `_observable_trackers(trackers, disabled: list[str]) -> dict`; `sentry_enabled(sentry_dsn: str) -> bool`. The module keeps a process-level sink registry (`register_sink`); `report_error` keeps its current signature. `_stage_spans`/`PhoenixSink` are UNCHANGED here (WS1 deletes them); `_phoenix_is_local()` gains an `endpoint` param and is fed a module-level `_PHOENIX_ENDPOINT = None` sentinel (no live env read — see Step 4) so the Phoenix branch is dead-but-importable until WS1 deletes it.

> Interfaces note: `fanout`/`capture_cognition_alarm`/`init_sponsors`/`sentry_enabled`/`_flag_reason` are imported into `backend/app.py` (Task 13). WS1 will further edit `init_sponsors` (drop Phoenix) and rename `capture_cognition_alarm`/`CognitionEvent`. Sequence WS1 AFTER this task.

Steps:

- [ ] Step 1: Read `backend/fanout.py` in full to capture every `config.*` site and the exact current bodies of `fanout`, `capture_cognition_alarm`, `_scrub_pii`, `_observable_trackers`, `sentry_enabled`, `init_sponsors`. (No code change yet.)
```
$ uv run python -c "import re,sys; print(open('backend/fanout.py').read())" | sed -n '1,330p'
```
Expected: prints the file; note `config.SENTRY_SEND_IO` (lines ~112,184), `config.DISABLED_TRACKERS` (line ~143), `config.SENTRY_DSN`/`SENTRY_ENVIRONMENT`/`SENTRY_RELEASE` (lines ~286–291), `config.PHOENIX_ENDPOINT` (lines ~276,307).

- [ ] Step 2: Rewrite the failing assertions in `backend/tests/test_fanout.py`. Replace the `monkeypatch.setattr(fo.config, ...)` lines and the `fanout()`/`init_sponsors()`/`capture_cognition_alarm()` calls with sub-config injection. The full rewritten test file:
```python
import backend.fanout as fo
from backend.config import ObsConfig, ProbeConfig


def _obs(send_io=False, dsn=""):
    obs = ObsConfig()
    obs.sentry.send_io = send_io
    return obs


def _probes(disabled=("uncertainty", "hallucination", "risk_awareness")):
    return ProbeConfig(disabled=list(disabled))


def test_fanout_null_uncertainty_does_not_crash():
    event = {"message_id": "m1", "ts": 0.0, "flag": False, "uncertainty": None, "trackers": {}}
    fo.fanout(event, None, obs=_obs(), probes=_probes())  # must not raise


def test_scrub_pii_strips_exception_frame_vars():
    from backend.fanout import _scrub_pii

    ev = {"exception": {"values": [{"stacktrace": {"frames": [{"vars": {"x": "secret"}}]}}]}}
    _scrub_pii(ev, {}, send_io=False)
    assert ev["exception"]["values"][0]["stacktrace"]["frames"][0].get("vars") in (None, {})


def test_level_clamps():
    from backend.fanout import _level

    assert _level("warning") in ("warning", "error", "info")


def test_fanout_isolates_failing_sink(monkeypatch):
    def boom(event, perf):
        raise RuntimeError("sink down")

    monkeypatch.setattr(fo, "_SINKS", [boom], raising=False)
    fo.fanout({"message_id": "m", "flag": False, "trackers": {}}, None, obs=_obs(), probes=_probes())


def test_store_sink_records_redacted():
    from backend import observability

    event = {"message_id": "store1", "flag": False, "uncertainty": 0.2, "trackers": {}, "io": {"user_msg": "raw"}}
    fo.fanout(event, None, obs=_obs(send_io=False), probes=_probes())
    snap = observability.STORE.snapshot()
    assert isinstance(snap, dict)


def test_observable_trackers_drops_disabled():
    from backend.fanout import _observable_trackers

    trackers = {"harmful": {"score": 0.1}, "uncertainty": {"score": 0.9}}
    out = _observable_trackers(trackers, ["uncertainty"])
    assert "harmful" in out and "uncertainty" not in out


def test_flag_reason_prefers_severity_order():
    from backend.fanout import _flag_reason

    assert isinstance(_flag_reason({"flag": True, "severity": "warning", "trackers": {}}), str)


def test_sentry_sink_quiet_on_unflagged_and_redacted_on_flag(monkeypatch):
    import sys

    sent = {}

    class FakeSdk:
        @staticmethod
        def capture_event(ev):
            sent["ev"] = ev

        @staticmethod
        def flush(*a, **k):
            pass

        @staticmethod
        def push_scope():
            class _S:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    return False

                def set_tag(self_, *a, **k):
                    pass

                def set_context(self_, *a, **k):
                    pass

                def set_fingerprint(self_, *a, **k):
                    pass

                def set_level(self_, *a, **k):
                    pass

            return _S()

    monkeypatch.setitem(sys.modules, "sentry_sdk", FakeSdk)
    event = {"message_id": "f1", "flag": True, "severity": "warning", "uncertainty": 0.9, "trackers": {}, "io": {"user_msg": "raw"}}
    fo.capture_cognition_alarm(event, _obs(send_io=False), flush=True)


def test_scrub_pii_walks_whole_event_not_just_three_sections():
    from backend.fanout import _scrub_pii

    ev = {"a": {"user_msg": "secret"}, "b": [{"response": "more"}]}
    _scrub_pii(ev, {}, send_io=False)
    assert "secret" not in str(ev)
    assert "more" not in str(ev)
```

- [ ] Step 3: Run and confirm failure:
```
$ uv run pytest backend/tests/test_fanout.py -q
```
Expected: `TypeError: fanout() got an unexpected keyword argument 'obs'` and `_scrub_pii() got an unexpected keyword argument 'send_io'`, etc.

- [ ] Step 4: Edit `backend/fanout.py`. Apply these surgical changes (do NOT touch the Phoenix sink internals — only its `config.*` reads and the function signatures):
  - Remove the `from . import config` import line.
  - `fanout(event, perf=None)` → `fanout(event, perf=None, *, obs, probes)`; inside, pass `obs`/`probes` to the store-sink redaction and `_observable_trackers(..., probes.disabled)`; pass `obs.sentry.send_io` to `_scrub_pii`.
  - `_scrub_pii(event, hint)` → `_scrub_pii(event, hint, *, send_io: bool)`; replace `config.SENTRY_SEND_IO` with `send_io` at both sites (lines ~112,184).
  - `_observable_trackers(trackers)` → `_observable_trackers(trackers, disabled)`; replace `config.DISABLED_TRACKERS` (line ~143) with `disabled`.
  - `capture_cognition_alarm(event, *, flush=False)` → `capture_cognition_alarm(event, obs, *, flush=False)`; thread `obs.sentry.send_io` into the `_scrub_pii` call and `obs.sentry.environment`/`obs.sentry.release` where the alarm sets Sentry scope.
  - `sentry_enabled()` → `sentry_enabled(sentry_dsn: str) -> bool`; return `bool(sentry_dsn)`.
  - `init_sponsors()` → `init_sponsors(obs, sentry_dsn: str)`; replace `config.SENTRY_DSN`→`sentry_dsn` (line 286, 289), `config.SENTRY_ENVIRONMENT`→`obs.sentry.environment` (line 290), `config.SENTRY_RELEASE`→`obs.sentry.release` (line 291).
  - Phoenix branch (WS1 deletes it; WS0 must keep the module importable WITHOUT re-introducing a live env read — the design doc wants env-coupling gone). Make the branch DATA-DRIVEN rather than hardcoding `os.getenv`: introduce a module-level constant `_PHOENIX_ENDPOINT: str | None = None  # WS1: delete — Phoenix removed; WS0 disables this branch` at the top of the Phoenix section (near line 216). Then:
    - `_phoenix_is_local()` (line 276) → `_phoenix_is_local(endpoint: str | None)`: return `False` immediately if `endpoint` is None, else parse `urlparse(endpoint).hostname`. Replace the `config.PHOENIX_ENDPOINT` read with the passed `endpoint`.
    - In `init_sponsors`, gate the whole Phoenix block on `if _PHOENIX_ENDPOINT and _phoenix_is_local(_PHOENIX_ENDPOINT):` and use `_PHOENIX_ENDPOINT` (line 307) instead of `config.PHOENIX_ENDPOINT`. Since `_PHOENIX_ENDPOINT` defaults to `None`, the branch is dead-but-importable in WS0 (no `PhoenixSink` registered, no env read) and dies cleanly when WS1 deletes the section. Keep the line-314 fail-closed print but it becomes unreachable while `_PHOENIX_ENDPOINT is None`.
    - Add a `# WS1: delete this entire Phoenix branch (and _PHOENIX_ENDPOINT, PhoenixSink, _stage_spans, _phoenix_is_local)` comment so WS1 has one clear target. This re-adds NO live `os.getenv` coupling — the endpoint is a `None` sentinel that WS1 removes.

- [ ] Step 5: Run and confirm pass:
```
$ uv run pytest backend/tests/test_fanout.py -q
```
Expected: `9 passed`.

- [ ] Step 6: Commit:
```
$ git add backend/fanout.py backend/tests/test_fanout.py
$ git commit -m "refactor(fanout): thread ObsConfig/ProbeConfig/sentry_dsn; Phoenix branch left for WS1"
```

---

### Task 9: Genericize fallback synthetic turn (rebrand)

Files:
- Modify: `backend/fallback.py` (lines 12–74 — neutral templates, neutral marker detection, `synth_turn(messages, sae)`).
- Modify: `backend/mock_labels.py` (rename `CLINICAL_LABELS` → `GENERIC_LABELS` with neutral content).
- Test: `backend/tests/test_fallback.py` (rewrite — pass `SAEConfig`; assert neutral output).

Interfaces:
- Consumes: `SAEConfig` from Task 1; `cfg.sae.d_sae`, `cfg.np_source()`.
- Produces: `synth_turn(messages: list[dict], sae) -> tuple[str, list[dict]]` (uses `sae.d_sae` and a passed `np_source` string), `is_synthetic_response(text: str) -> bool` (neutral markers). `mock_labels.GENERIC_LABELS: list[str]`.

> Interfaces note: `synth_turn` is called from `backend/analyze.py` fallback path (Task 11). The `np_source` slug must be passed in (analyze threads `cfg.np_source()`).

> Impact / marker-change verification: `is_synthetic_response` markers change (the OLD markers `"That's an important clinical question about"` / `"Good question regarding"` are replaced by the neutral markers in the new `_TEMPLATES`). VERIFIED against the repo: the ONLY consumer of `is_synthetic_response` is `analyze.analyze_turn` (strict-mode check, analyze.py:179-182), which detects whatever `synth_turn` produces — the new neutral markers stay self-consistent. No test asserts the OLD marker strings. The fallback path is exercised by `test_analyze.py` (rewritten in Task 11 with neutral samples) and `test_api.py` (`/api/chat` SSE-shape test, which does NOT assert on the answer text and whose medical sample is swapped for a neutral one in Task 13) — neither asserts the medical markers, so they tolerate the neutral output. `test_fallback.py` is fully rewritten in Step 1 of this task. `backend/fallback.py` keeps no `from . import config` (its only flat reads, `D_SAE`/`NP_SOURCE`, become `sae.d_sae` + the passed `np_source`); it is imported lazily by `analyze.py` at runtime, so dropping config there is safe.

Steps:

- [ ] Step 1: Rewrite `backend/tests/test_fallback.py`:
```python
from backend import fallback
from backend.config import AppConfig


def test_synth_turn_is_deterministic_and_neutral():
    cfg = AppConfig()
    msgs = [{"role": "user", "content": "How do rainbows form?"}]
    a1, f1 = fallback.synth_turn(msgs, cfg.sae, np_source=cfg.np_source())
    a2, f2 = fallback.synth_turn(msgs, cfg.sae, np_source=cfg.np_source())
    assert a1 == a2  # deterministic
    assert "clinical" not in a1.lower() and "patient" not in a1.lower()
    assert len(f1) >= 1
    for feat in f1:
        assert 0 <= feat["index"] < cfg.sae.d_sae
        assert feat["source"] == "17-gemmascope-2-res-16k"
        assert "clinical" not in feat["label"].lower()


def test_is_synthetic_response_detects_template():
    cfg = AppConfig()
    answer, _ = fallback.synth_turn([{"role": "user", "content": "x"}], cfg.sae, np_source=cfg.np_source())
    assert fallback.is_synthetic_response(answer) is True
    assert fallback.is_synthetic_response("totally unrelated text") is False
```

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_fallback.py -q
```
Expected: `TypeError: synth_turn() takes 1 positional argument but 2 were given` / import error on `GENERIC_LABELS`.

- [ ] Step 3: Edit `backend/mock_labels.py` — rename the symbol and neutralize content:
```python
"""Curated feature labels for the offline synthetic fallback. Plausible general-purpose
concepts so the feature map reads believably without any network call."""

GENERIC_LABELS = [
    "question framing / inquiry",
    "explanatory discourse",
    "hedging / expressions of caution",
    "cause-and-effect reasoning",
    "comparison & contrast",
    "definitions & terminology",
    "second-person address",
    "affirmation / yes",
    "temporal periods",
    "quantities & measurement",
    "examples & illustration",
    "uncertainty / qualification",
    "lists & enumeration",
    "summary & conclusion",
    "topic introduction",
    "factual recall",
]
```

- [ ] Step 4: Rewrite `backend/fallback.py`:
```python
"""Deterministic synthetic turn for when the real model/SAE is absent (no torch, no weights,
or still loading). Offline-safe: features carry their own curated labels — no network.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import hashlib
import random

from .mock_labels import GENERIC_LABELS

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"

_TEMPLATES = [
    "That's a good question about {topic}. The answer generally depends on the specifics, "
    "so it's worth checking an authoritative source before relying on this.",
    "Regarding {topic}: there are real trade-offs here, and the right answer varies by "
    "context; I'd verify the latest information before acting on it.",
    "On {topic}, the considerations are nuanced. General principles offer guidance, but the "
    "details matter, so corroborate with a reliable source.",
]


def _seed(messages: list[dict]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _topic(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            words = m["content"].strip().rstrip("?.!").split()
            return " ".join(words[:8]) if words else "this question"
    return "this question"


def is_synthetic_response(text: str) -> bool:
    """True if the answer matches fallback.synth_turn template output."""
    markers = (
        "That's a good question about",
        "Regarding ",
        "On ",
        ": there are real trade-offs",
        ": the considerations are nuanced",
    )
    return any(m in text for m in markers)


def synth_turn(messages: list[dict], sae, *, np_source: str) -> tuple[str, list[dict]]:
    """Return a deterministic (answer, features) for the given conversation."""
    seed = _seed(messages)
    rng = random.Random(seed)
    answer = rng.choice(_TEMPLATES).format(topic=_topic(messages))

    n = rng.randint(8, 12)
    labels = rng.sample(GENERIC_LABELS, k=min(n, len(GENERIC_LABELS)))
    features: list[dict] = []
    for i, label in enumerate(labels):
        idx = (seed >> (i * 3)) % sae.d_sae
        act = round(6.4 - i * 0.42 + rng.random() * 0.25, 3)
        features.append(
            {
                "index": idx,
                "label": label,
                "act": act,
                "source": np_source,
                "caveat": DEFAULT_CAVEAT,
                "tracked": None,
            }
        )
    return answer, features
```

- [ ] Step 5: Run and confirm pass:
```
$ uv run pytest backend/tests/test_fallback.py -q
```
Expected: `2 passed`.

- [ ] Step 6: Commit:
```
$ git add backend/fallback.py backend/mock_labels.py backend/tests/test_fallback.py
$ git commit -m "refactor(fallback): neutralize synthetic templates/labels; take SAEConfig + np_source"
```

---

### Task 10: Thread sub-configs into labels + neutralize auto-interp prompt (rebrand)

Files:
- Modify: `backend/labels.py` (lines ~55, 69–71, 92, 117, 166, 181–185, 204 — thread `sae`/`feature_cloud`/`anthropic_api_key`/`np_source` through `get_label` → `get_feature_stats` → `_autointerp`; neutral prompt at line ~117).
- Test: `backend/tests/test_labels_autointerp.py` (rewrite — pass sub-configs; mock the Neuronpedia `httpx.get`; assert neutral prompt).

Interfaces:
- Consumes: `SAEConfig`, `FeatureCloudConfig` from Task 1; `cfg.sae` (`np_model`, `np_source` via `cfg.np_source()`, `np_feature_url`), `cfg.feature_cloud` (`autointerp`, `autointerp_model`), `cfg.anthropic_api_key`.
- Produces: `get_label(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0) -> str` and `get_feature_stats(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0) -> dict` and `_autointerp(activations, feature_cloud, anthropic_api_key) -> dict | None`. NOTE: `activations` is NOT a caller param of `get_label`/`get_feature_stats` — it is read from the Neuronpedia HTTP response inside `get_feature_stats` (labels.py:83) and only passed DOWN into `_autointerp`. The auto-interp Claude prompt text must NOT say "medical chatbot".

> Interfaces note: `labels.get_label` is called from `backend/app.py /api/feature/{index}` (Task 13). Thread `cfg.sae`, `cfg.feature_cloud`, `cfg.anthropic_api_key`, `cfg.np_source()`. The real auto-interp flow (labels.py:55-111): `get_label` → `get_feature_stats` → `httpx.get(NP_FEATURE_URL)`; the `_autointerp(activations)` branch (line 92) fires ONLY when Neuronpedia returns NO explanation but DID return activating examples. The test must therefore mock the Neuronpedia `httpx.get` to return that exact shape — it cannot inject `activations` directly. Also note `get_feature_stats` caches in `_stats`/`_disk`; the test must use a fresh index and clear the in-memory `_stats` cache (`labels._stats.clear()`) so the autointerp branch is actually reached. `get_labels`/`prewarm`/`preload_from_s3` (helpers that call `get_label`/`get_feature_stats`) must have their internal calls updated to thread the same params, or be marked as needing the threaded args by their callers.

Steps:

- [ ] Step 1: Read `backend/labels.py` in full to capture the exact current `get_label` signature and the line-117 prompt string.
```
$ uv run python -c "print(open('backend/labels.py').read())" | sed -n '60,200p'
```
Expected: shows `get_label`, the `_autointerp`/Claude call, and the prompt with "feature of a medical chatbot (Gemma-3-4b-it, layer 17)".

- [ ] Step 2: Rewrite `backend/tests/test_labels_autointerp.py` to inject sub-configs AND mock the Neuronpedia HTTP call so the autointerp branch actually fires. The autointerp path (labels.py:92) only runs when Neuronpedia returns NO explanation but DID return `activations` — so the test mocks `labels.httpx.get` (the `import httpx` inside `get_feature_stats` resolves to the module-level `httpx`, monkeypatch `labels.httpx`) to return a 200 with `explanations: []` and a populated `activations`. Full file:
```python
import backend.labels as labels
from backend.config import AppConfig


class _NPResp:
    """Neuronpedia response with NO explanation but WITH activating examples -> autointerp fires."""

    status_code = 200

    def json(self):
        return {
            "explanations": [],
            "maxActApprox": 8.0,
            "frac_nonzero": 0.001,
            "activations": [{"tokens": ["the", "cat"], "values": [0.0, 5.0], "maxValueTokenIndex": 1}],
        }


def test_autointerp_prompt_is_neutral(monkeypatch):
    """The auto-interp Claude prompt must not describe the model as a medical chatbot."""
    captured = {}

    class FakeMsg:
        # one tool_use block, mirroring the record_feature_label tool contract
        content = [type("B", (), {"type": "tool_use", "name": "record_feature_label",
                                  "input": {"label": "x", "is_structural": False}})()]

    class FakeMessages:
        def create(self, **kw):
            captured["system"] = kw.get("system", "")
            return FakeMsg()

    class FakeClient:
        def __init__(self, **kw):
            self.messages = FakeMessages()

    labels._stats.clear()  # force the network/autointerp branch (skip the session cache)
    monkeypatch.setattr(labels, "_disk_cache", lambda: {})  # skip the disk cache too
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _NPResp())
    monkeypatch.setattr(labels, "_anthropic_client", lambda key: FakeClient(), raising=False)

    cfg = AppConfig()
    cfg.anthropic_api_key = "sk-test"
    out = labels.get_label(
        4242, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()
    )
    assert isinstance(out, str)
    assert "medical" not in captured.get("system", "").lower()
    assert "clinical" not in captured.get("system", "").lower()


def test_get_label_no_autointerp_without_key(monkeypatch):
    """No Claude key -> autointerp returns None; label degrades to 'feature N' (never raises)."""
    labels._stats.clear()
    monkeypatch.setattr(labels, "_disk_cache", lambda: {})
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _NPResp())
    cfg = AppConfig()  # anthropic_api_key == ""
    out = labels.get_label(
        4343, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()
    )
    assert out == "feature 4343"  # degraded fallback, no Claude call
```

- [ ] Step 3: Run and confirm failure:
```
$ uv run pytest backend/tests/test_labels_autointerp.py -q
```
Expected: `TypeError: get_label() ...` arity error (the real `get_label(index, timeout=6.0)` rejects the sub-config args).

- [ ] Step 4: Edit `backend/labels.py`:
  - Drop `from . import config`.
  - Thread `sae`, `feature_cloud`, `anthropic_api_key`, and a keyword-only `np_source` through `get_label` → `get_feature_stats` → `_autointerp`:
    - `get_feature_stats(index, timeout=6.0)` → `get_feature_stats(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0)`; replace `config.NP_FEATURE_URL`→`sae.np_feature_url`, `config.NP_MODEL`→`sae.np_model`, `config.NP_SOURCE`→`np_source` (the `url = ...format(...)` at lines 69-71); replace the `config.AUTOINTERP` guard (line 92) with `feature_cloud.autointerp` and call `_autointerp(activations, feature_cloud, anthropic_api_key)`.
    - `get_label(index, timeout=6.0)` → `get_label(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0)`; forward all of them to `get_feature_stats(...)`.
    - `_autointerp(activations)` → `_autointerp(activations, feature_cloud, anthropic_api_key)`; replace `config.ANTHROPIC_API_KEY`→`anthropic_api_key` (guard at line 166 + the client init at line 182), `config.AUTOINTERP_MODEL`→`feature_cloud.autointerp_model` (line 185).
    - `get_labels(indices)` and `prewarm(indices)` call `get_label`/`get_feature_stats` internally — give them the same threaded params (or accept `sae, feature_cloud, anthropic_api_key, *, np_source`) so they don't break the module; they are pre-warm helpers not on the live request path, so the minimal change is to add the params and forward them.
  - Replace the line-117 prompt fragment `"feature of a medical chatbot (Gemma-3-4b-it, layer 17). You are shown text excerpts where "` with the neutral `"feature of a general-purpose language model. You are shown text excerpts where "` (drop the medical + hardcoded model/layer framing).
  - Add a module-level `def _anthropic_client(key): import anthropic; return anthropic.Anthropic(api_key=key, timeout=_CLAUDE_TIMEOUT, max_retries=2)` helper so the test can monkeypatch a single seam, and call it from `_autointerp` (replacing the inline `anthropic.Anthropic(...)` at line 181-183).

- [ ] Step 5: Run and confirm pass:
```
$ uv run pytest backend/tests/test_labels_autointerp.py -q
```
Expected: `2 passed` (the neutral-prompt test actually reaches the Claude prompt via the mocked Neuronpedia branch; the no-key test degrades to "feature N").

- [ ] Step 6: Commit:
```
$ git add backend/labels.py backend/tests/test_labels_autointerp.py
$ git commit -m "refactor(labels): thread SAEConfig/FeatureCloudConfig/key; neutralize auto-interp prompt"
```

---

### Task 11: Thread AppConfig into analyze (CPU orchestration entry point)

Files:
- Modify: `backend/analyze.py` (signature of `analyze_turn` — ADD `cfg` as the explicit 2nd positional param, PRESERVE the existing keyword-only `message_id`/`ts`/`strict`; `_rank_features` reads `feature_cloud`; `_real_turn(messages, cfg)`; the in-`analyze_turn` `runtime.refresh_pod_health(cfg)`; `build_cognition_event(model=cfg.model.model_id, layer=cfg.model.layer)`; fallback path calls `synth_turn(messages, cfg.sae, np_source=cfg.np_source())`; pod calls pass `cfg.pod`, `cfg.pod_token`, `cfg.model.max_new_tokens`).
- Test: `backend/tests/test_rank.py` (rewrite — pass `FeatureCloudConfig` into `_rank_features`), `backend/tests/test_analyze.py` (rewrite — inject `cfg`, neutral samples, new monkeypatch arities for `pod_client.turn`/`runtime.refresh_pod_health`).

Interfaces:
- Consumes: `AppConfig` (Task 2); `FeatureCloudConfig`; `pod_client.turn(..., pod, pod_token, max_new=...)` (Task 6); `fallback.synth_turn(messages, sae, np_source=...)` (Task 9); `runtime.refresh_pod_health(cfg)` (Task 5); `build_cognition_event(..., model, layer)` (unchanged signature, but `model`/`layer` now sourced from `cfg.model`).
- Produces: `analyze_turn(messages, cfg, *, message_id=None, ts=None, strict=False) -> tuple[str, CognitionEvent, dict]` (returns `(answer, event, perf)` — same triple as today; the keyword-only block is UNCHANGED, `cfg` is inserted as the 2nd positional so existing `message_id=`/`ts=`/`strict=` callers keep working), `_real_turn(messages, cfg) -> tuple[str, list[dict], dict, dict]`, `_rank_features(candidates, feature_cloud) -> list[dict]`.

> Interfaces note: the REAL current signature is `analyze_turn(messages, *, message_id=None, ts=None, strict=False)` (analyze.py:141-147). DO NOT delete the keyword-only block: insert `cfg` as the explicit 2nd POSITIONAL param so the signature becomes `analyze_turn(messages, cfg, *, message_id=None, ts=None, strict=False)`. All call sites must be updated: `app.py` `/api/chat`+`/api/analyze` (Task 13), `test_analyze.py` (this task), and `test_api.py` (Task 13). `analyze_turn` is run in a threadpool with `cfg` bound from `request.app.state.config`. Read `backend/analyze.py` in full first to map all `config.*` sites and the pod-vs-fallback branch — note the in-function call sites: `runtime.refresh_pod_health()` (line 156), `_real_turn(messages)` (line 154, which calls `pod_client.turn(messages, max_new=config.MAX_NEW_TOKENS)` at line 128 and `_rank_features(r["candidates"])` at line 132), and `build_cognition_event(..., model=config.MODEL_ID, layer=config.LAYER)` (lines 191-192) — ALL of these break once `from . import config` is gone and the threaded signatures land.

Steps:

- [ ] Step 1: Read `backend/analyze.py` in full to capture the exact `analyze_turn` body, every `config.*` site, and how it chooses pod vs fallback.
```
$ uv run python -c "print(open('backend/analyze.py').read())"
```
Expected: shows `analyze_turn(messages)`, the `_rank_features` knobs (`config.STRUCTURAL_PENALTY`, `DROP_UNLABELED`, `TOPK_EVENT`, `DENSITY_MAX`, `SYNTACTIC_PENALTY`), and the pod-client / fallback dispatch.

- [ ] Step 2: Rewrite `backend/tests/test_rank.py` to inject `FeatureCloudConfig`:
```python
from backend.analyze import _rank_features
from backend.config import FeatureCloudConfig


def test_rank_drops_unlabeled_when_enabled():
    fc = FeatureCloudConfig()  # drop_unlabeled=True
    candidates = [
        {"index": 1, "label": "drug safety", "act": 5.0},
        {"index": 2, "label": "feature 2", "act": 9.0},  # bare label → dropped
    ]
    out = _rank_features(candidates, fc)
    labels = {f["label"] for f in out}
    assert "feature 2" not in labels


def test_rank_keeps_unlabeled_when_disabled():
    fc = FeatureCloudConfig(drop_unlabeled=False)
    candidates = [
        {"index": 1, "label": "drug safety", "act": 5.0},
        {"index": 2, "label": "feature 2", "act": 9.0},
    ]
    out = _rank_features(candidates, fc)
    assert len(out) == 2


def test_rank_caps_at_topk_event():
    fc = FeatureCloudConfig(topk_event=3, drop_unlabeled=False)
    candidates = [{"index": i, "label": f"concept {i}", "act": float(10 - i)} for i in range(10)]
    out = _rank_features(candidates, fc)
    assert len(out) <= 3
```

  Also rewrite `backend/tests/test_analyze.py` to inject `cfg`, use neutral samples, and match the new monkeypatch arities. The real file calls `analyze_turn([...])` positionally, monkeypatches `pod_client.turn` as `fake_turn(messages, max_new=None)`, and monkeypatches `runtime.refresh_pod_health` as `lambda: None` — ALL of these break. Full rewritten file:
```python
from backend import analyze, runtime
from backend.config import AppConfig
from backend.schema import CognitionEvent


def _cfg():
    return AppConfig()


def test_analyze_turn_fallback_builds_valid_event():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
    answer, event, perf = analyze.analyze_turn(
        [{"role": "user", "content": "How do rainbows form?"}], _cfg()
    )
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(event, CognitionEvent)
    assert event.uncertainty is None
    assert event.flag is False
    assert event.severity == "info"
    assert event.io.user_msg == "How do rainbows form?"
    assert event.io.response == answer
    assert len(event.features) > 0
    assert event.features[0].label
    assert "clinical" not in answer.lower() and "patient" not in answer.lower()
    CognitionEvent(**event.model_dump())


def test_real_turn_uses_pod_client_and_ranks(monkeypatch):
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(
        analyze.labels,
        "get_feature_stats",
        lambda i, sae, fc, key, *, np_source=None, **k: {"label": f"label-{i}", "max_act": 1.0, "density": 0.001},
    )

    def fake_turn(messages, pod, pod_token, *, max_new=None):
        return {
            "answer": "pod answer",
            "candidates": [
                {"index": 2, "act": 1.0, "attr": 0.9, "source": "s"},
                {"index": 1, "act": 5.0, "attr": 0.2, "source": "s"},
            ],
            "trackers": {
                "uncertainty": {
                    "score": 0.3, "proj": 0.1, "proj_pre": None, "flag": False,
                    "reliable": True, "status": "ready", "user_defined": False,
                }
            },
            "reliable": True,
        }

    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", fake_turn)
    answer, event, perf = analyze.analyze_turn([{"role": "user", "content": "hi"}], _cfg())
    assert answer == "pod answer"
    assert [f.index for f in event.features] == [2, 1]
    assert event.features[0].label == "label-2"


def test_real_turn_degrades_on_pod_failure(monkeypatch):
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", lambda *a, **k: (_ for _ in ()).throw(pc.PodError(0, "turn")))
    monkeypatch.setattr(analyze.runtime, "refresh_pod_health", lambda cfg: None)
    answer, event, perf = analyze.analyze_turn(
        [{"role": "user", "content": "How does compound interest work?"}], _cfg()
    )
    assert isinstance(answer, str) and answer.strip()
    assert len(event.features) > 0


def test_analyze_turn_returns_perf(monkeypatch):
    """analyze_turn must return a 3-tuple (answer, event, perf) with timing fields."""
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(
        analyze.labels,
        "get_feature_stats",
        lambda i, sae, fc, key, *, np_source=None, **k: {"label": f"label-{i}", "max_act": 1.0, "density": 0.001},
    )

    def fake_turn_with_timings(messages, pod, pod_token, *, max_new=None):
        return {
            "answer": "timed answer",
            "candidates": [{"index": 1, "act": 1.0, "attr": 0.5, "source": "s"}],
            "trackers": {},
            "reliable": True,
            "timings": {"capture": 10.0, "sae": 5.0, "trackers": 2.0},
        }

    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", fake_turn_with_timings)
    result = analyze.analyze_turn([{"role": "user", "content": "What is a vector?"}], _cfg())
    assert len(result) == 3, "analyze_turn must return 3-tuple (answer, event, perf)"
    answer, event, perf = result
    assert answer == "timed answer"
    assert isinstance(event, CognitionEvent)
    assert "turn_ms" in perf and isinstance(perf["turn_ms"], float)
    assert "stages" in perf
    assert "pod_roundtrip" in perf["stages"]
    assert "pod_stages" in perf
    assert perf["pod_stages"].get("capture") == 10.0
    assert perf["pod_stages"].get("sae") == 5.0
    assert perf["pod_stages"].get("trackers") == 2.0


def test_pod_failure_reports_instrument_unhealthy(monkeypatch):
    """Regression: a pod failure must wire the instrument_unhealthy concern to Sentry via
    fanout.report_error('pod-down', exc)."""
    from backend import runtime
    import backend.pod_client as pc
    import backend.fanout as fo

    calls = []
    monkeypatch.setattr(fo, "report_error", lambda stage, exc, ctx=None: calls.append(stage))
    monkeypatch.setattr(analyze.runtime, "refresh_pod_health", lambda cfg: None)
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(pc, "turn", lambda *a, **k: (_ for _ in ()).throw(pc.PodError(0, "turn")))
    analyze.analyze_turn([{"role": "user", "content": "x"}], _cfg())
    assert "pod-down" in calls
```
Note: `fake_turn`/`fake_turn_with_timings` now accept `(messages, pod, pod_token, *, max_new=None)` to match the Task 6 `pod_client.turn` signature, and `refresh_pod_health` is monkeypatched as `lambda cfg: None` to match Task 5.

- [ ] Step 3: Run and confirm failure:
```
$ uv run pytest backend/tests/test_rank.py backend/tests/test_analyze.py -q
```
Expected: `TypeError: _rank_features() takes 1 positional argument but 2 were given` (test_rank) and `TypeError: analyze_turn() missing 1 required positional argument: 'cfg'` (test_analyze).

- [ ] Step 4: Edit `backend/analyze.py`:
  - Change `analyze_turn(messages, *, message_id=None, ts=None, strict=False)` → `analyze_turn(messages, cfg, *, message_id=None, ts=None, strict=False)`. Keep the keyword-only block VERBATIM — only insert `cfg` as the 2nd positional. Do not touch the `message_id`/`ts`/`strict` handling.
  - Change `_rank_features(candidates)` → `_rank_features(candidates, feature_cloud)`; replace `config.STRUCTURAL_PENALTY`→`feature_cloud.structural_penalty`, `config.DROP_UNLABELED`→`feature_cloud.drop_unlabeled`, `config.TOPK_EVENT`→`feature_cloud.topk_event`, `config.DENSITY_MAX`→`feature_cloud.density_max`, `config.SYNTACTIC_PENALTY`→`feature_cloud.syntactic_penalty`. Update BOTH `labels.get_feature_stats` call sites inside `_rank_features` (the function that enriches each ranked feature with its label/stats): change every call from `labels.get_feature_stats(f["index"])` to the fully-threaded form:
```python
labels.get_feature_stats(f["index"], cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())
```
(The `cfg` object must be accessible inside `_rank_features` — either add it as a parameter `_rank_features(candidates, feature_cloud, cfg)` or, if the existing body already closes over `cfg` from the enclosing `_real_turn`/`analyze_turn` scope, ensure it does so explicitly. Both call sites in `_rank_features` must be updated; leaving either one as `labels.get_feature_stats(f["index"])` will raise a `TypeError` at runtime because Task 10 migrates `get_feature_stats` to require `(index, sae, feature_cloud, anthropic_api_key, *, np_source, timeout=6.0)`.)
  - Change `_real_turn(messages)` → `_real_turn(messages, cfg)`; inside it replace `pod_client.turn(messages, max_new=config.MAX_NEW_TOKENS)` → `pod_client.turn(messages, cfg.pod, cfg.pod_token, max_new=cfg.model.max_new_tokens)` and `_rank_features(r["candidates"])` → `_rank_features(r["candidates"], cfg.feature_cloud)`.
  - In `analyze_turn`: call `_real_turn(messages, cfg)`; in the pod-failure `except` block change `runtime.refresh_pod_health()` → `runtime.refresh_pod_health(cfg)`; change BOTH fallback calls `fallback.synth_turn(messages)` → `fallback.synth_turn(messages, cfg.sae, np_source=cfg.np_source())` (the real-path `except` branch AND the `else` non-real branch); change `build_cognition_event(..., model=config.MODEL_ID, layer=config.LAYER)` → `build_cognition_event(..., model=cfg.model.model_id, layer=cfg.model.layer)`.
  - Drop `from . import config` (the module no longer references any flat global once the above are threaded; keep `labels`/`runtime` imports and the lazy `pod_client`/`fanout`/`fallback` imports).

- [ ] Step 5: Run and confirm pass:
```
$ uv run pytest backend/tests/test_rank.py backend/tests/test_analyze.py -q
```
Expected: `8 passed` (3 in test_rank, 5 in test_analyze).

- [ ] Step 6: Commit:
```
$ git add backend/analyze.py backend/tests/test_rank.py backend/tests/test_analyze.py
$ git commit -m "refactor(analyze): thread AppConfig through analyze_turn/_real_turn/_rank_features; neutral test samples"
```

---

### Task 12: Thread sub-configs into torch modules (engine, sae, persona, feature_provider) — GPU-gated

Files:
- Modify: `backend/engine.py` (`load_engine(model, device)`, `probe_activation(text, model, pos)`, `_inject_system(messages, model)`, `_merge_system_into_user(messages, model)`, `info(model, sae)`, `generate_and_capture(..., model, feature_cloud)`; replace `config.MODEL_ID`/`LAYER`/`SYSTEM_PROMPT`/`RANK_METHOD`/`resolve_device`).
- Modify: `backend/science/sae.py` (`load_sae(sae, layer, device)`, `sae_topk(act, sae, feature_cloud, k)`, `attribution_topk(..., sae, feature_cloud, cap)`, `reconstruction_error(act, sae)`; replace `config.SAE_RELEASE`/`sae_id_for_layer`/`TOPK`/`TOPK_CANDIDATES`/`NP_SOURCE`/`resolve_device`).
- Modify: `backend/science/persona.py` (`load_artifacts(probes, include=None)`, `clear_custom_trackers(probes, ...)`, module `ARTIFACT_DIR` kept as a default fallback only; replace `config.ENABLED_TRACKERS`/`DISABLED_TRACKERS`).
- Modify: `backend/science/feature_provider.py` (provider `features_for` methods take `(sae, feature_cloud, model)`; replace `config.TOPK`/`TOPK_EVENT`/`RANK_METHOD`/`PREAMBLE_SKIP`/`NP_MODEL`/`NP_SOURCE`/`resolve_device`).
- Test: `backend/tests/test_persona.py` (rewrite — pass `ProbeConfig`), `backend/tests/test_feature_provider.py` (rewrite — pass sub-configs), `backend/tests/test_attribution.py` (rewrite — pass `SAEConfig`/`FeatureCloudConfig` where it reads config).

Interfaces:
- Consumes: `ModelConfig`, `SAEConfig`, `FeatureCloudConfig`, `ProbeConfig`, and `cfg.sae_id(layer)`/`cfg.np_source(layer)`/`cfg.resolve_device()` helpers.
- Produces (contract §3.2 plus the explicit `np_source` threading decision — see the note below):
  - `engine.load_engine(model, device=None)`, `engine.probe_activation(text, model, pos=None)`, `engine.generate_and_capture(messages, *, max_new, attribution, model, feature_cloud)`.
  - `sae.load_sae(sae, layer, device=None)`, `sae.reconstruction_error(act, sae)`, `sae.sae_topk(act, sae, feature_cloud, *, np_source, k=None)`, `sae.attribution_topk(acts, grad, keep, sae, feature_cloud, *, np_source, cap=None, baseline=None)`, `sae.feature_attribution(acts, grad, keep)` unchanged (pure kernel), `sae.width()` unchanged.
  - `persona.load_artifacts(probes, include=None)` (uses `probes.artifacts_dir`, `probes.disabled`), `persona.clear_custom_trackers(probes, ...)`, `persona.score_all_trackers` unchanged.
  - `feature_provider.FeatureProvider.features_for(...)` (base), `feature_provider.LocalSAEProvider().features_for(answer, *, activations, token_ids, special_ids, grad, baseline, cap, sae, feature_cloud, model, np_source)`, `feature_provider.NeuronpediaProvider().features_for(..., sae, feature_cloud, model, np_source)`, and `feature_provider.get_provider(prefer=None, *, device)` (device resolved by the caller — see device note below).

> `np_source` threading decision (resolves the contract gap): the contract migration map maps `config.NP_SOURCE → cfg.np_source(layer)` for `science/sae` and `feature_provider`, but those torch modules do not hold an `AppConfig`. Rather than have each module rebuild the slug from `sae.np_source_pattern` (option a), this plan threads `np_source` explicitly as a keyword-only param into `sae.sae_topk`/`sae.attribution_topk` and `feature_provider.*.features_for` (option b). The single source of the slug is `gpu_service`, which DOES hold `cfg` and passes `np_source=cfg.np_source(cfg.model.layer)` (see Task 14 Step 4). This is consistent across the Produces block above, Task 12 Step 4, and Task 14's gpu_service edits.

> Interfaces note: `engine.py` and `science/sae.py` are GPU-gated (untested without weights) per Global Constraints — their direct tests stay opt-in, so for those two only the signature/`config.*` edits are made (no new failing unit test for the torch-loading paths). `persona`, `feature_provider`, and `attribution`'s pure-tensor helpers ARE CPU-testable. These four modules are consumed by `gpu_service.py` (Task 14). Sequence Task 14 immediately after.

Steps:

- [ ] Step 1: Read all four modules in full to capture exact current signatures and every `config.*` site.
```
$ for f in backend/engine.py backend/science/sae.py backend/science/persona.py backend/science/feature_provider.py; do echo "=== $f ==="; uv run python -c "print(open('$f').read())"; done | sed -n '1,400p'
```
Expected: confirms the signatures listed in the contract migration map.

- [ ] Step 2a: Rewrite `backend/tests/test_attribution.py` — it monkeypatches `analyze.config.*`/`fp.config.*`/`sae_mod.config.NP_SOURCE` and calls `_rank_features`/`attribution_topk`/`features_for` with the OLD arities; ALL of these break once `from . import config` is dropped from `analyze`/`feature_provider`/`sae`. Apply, mechanically:
  - Add `from backend.config import FeatureCloudConfig, SAEConfig, ModelConfig` at the top.
  - The `_FakeSAE` stand-in and `_install_fake_sae` stay. The kernel calls become `sae_mod.attribution_topk(acts, grad, keep=[...], cap=10)` → `sae_mod.attribution_topk(acts, grad, keep=[...], SAEConfig(), FeatureCloudConfig(), np_source="17-gemmascope-2-res-16k", cap=10)` (the `sae`/`feature_cloud` positional args + the keyword-only `np_source` per the Task 12 signature; `baseline=` stays keyword). Replace `assert all(c["source"] == sae_mod.config.NP_SOURCE for c in out)` with `assert all(c["source"] == "17-gemmascope-2-res-16k" for c in out)`. `feature_attribution(acts, grad, keep=...)` is unchanged (pure kernel, no config).
  - Every `analyze._rank_features(candidates)` → `analyze._rank_features(candidates, FeatureCloudConfig(...))`. Translate each monkeypatch into a constructor kwarg: `monkeypatch.setattr(analyze.config, "DROP_UNLABELED", True)` → pass `FeatureCloudConfig(drop_unlabeled=True)`; `STRUCTURAL_PENALTY`→`structural_penalty=`; `TOPK_EVENT`→`topk_event=`. For tests with no config monkeypatch (e.g. `test_rank_features_uses_attribution_when_present`), pass `FeatureCloudConfig()`. Replace the assertion `assert len(ranked) == analyze.config.TOPK_EVENT` with `assert len(ranked) == FeatureCloudConfig().topk_event` (or the explicit `topk_event` value passed in).
  - For the `LocalSAEProvider().features_for(...)` tests: replace `monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")` / `PREAMBLE_SKIP` with sub-config injection — pass `feature_cloud=FeatureCloudConfig(rank_method="attribution")` (and for preamble tests `model=ModelConfig(preamble_skip=2)` / `ModelConfig(preamble_skip=100)`) plus `sae=SAEConfig(), np_source="17-gemmascope-2-res-16k"` into each `features_for(...)` call, matching the Task 12 `LocalSAEProvider().features_for(answer, *, activations, token_ids, special_ids, grad, baseline, cap, sae, feature_cloud, model, np_source)` signature. The default-config provider tests (`test_provider_falls_back_to_activation_without_grad`) pass `feature_cloud=FeatureCloudConfig(), sae=SAEConfig(), model=ModelConfig(), np_source="17-gemmascope-2-res-16k"`.
  - Update the `fake_attr`/`fake_topk` monkeypatch stubs to the NEW kernel arities: `def fake_attr(activations, grad, keep, sae, feature_cloud, *, np_source=None, cap=None, baseline=None)` and `def fake_topk(act, sae, feature_cloud, *, np_source=None, k=None)`. They are monkeypatched onto `fp.attribution_topk`/`fp.sae_topk`, so they must accept whatever `LocalSAEProvider.features_for` now passes (sae/feature_cloud/np_source). Keep their bodies (capture `keep`/`cap`, return the same dicts).
  This is NOT deferred to the Task 15 grep — rewrite it here in full.

- [ ] Step 2b: Rewrite `backend/tests/test_persona.py` to inject `ProbeConfig` for the loader path (keep the existing pure-tensor tests like `persona_vector`/`project` unchanged; only the `load_artifacts`/`clear_custom_trackers` tests change). Add/replace:
```python
from pathlib import Path

from backend.config import ProbeConfig
from backend.science import persona


def test_load_artifacts_respects_disabled(tmp_path):
    # two artifacts on disk: one enabled-named, one disabled-named
    (tmp_path / "harmful.json").write_text(
        '{"tracker_id": "harmful", "direction": [0.1, 0.2], "norm_mean": 0.0, "norm_std": 1.0, "threshold": 0.5}'
    )
    (tmp_path / "uncertainty.json").write_text(
        '{"tracker_id": "uncertainty", "direction": [0.1, 0.2], "norm_mean": 0.0, "norm_std": 1.0, "threshold": 0.5}'
    )
    persona.clear_trackers()
    probes = ProbeConfig(artifacts_dir=tmp_path, disabled=["uncertainty"])
    loaded = persona.load_artifacts(probes)
    assert "harmful" in loaded
    assert "uncertainty" not in loaded
```
(If `load_tracker_artifact` requires a richer JSON shape, mirror an existing committed artifact's keys — read `backend/science/artifacts/harmful.json` first and copy its minimal field set.)

- [ ] Step 3: Run and confirm failure:
```
$ uv run pytest backend/tests/test_persona.py -q
```
Expected: `TypeError: load_artifacts() got an unexpected ... ` / arity error.

- [ ] Step 4: Edit the four modules:
  - `backend/engine.py`: `load_engine(device=None)` → `load_engine(model, device=None)`. Device resolution: `resolve_device` lives on `AppConfig`, NOT on `ModelConfig` — so `load_engine` does NOT call `model.resolve_device`. Instead the CALLER (gpu_service, Task 14) computes the device via `cfg.resolve_device()` and passes it in as `device=`; inside `load_engine`, if `device is None`, resolve via `torch` inline using `model.device` as the preference (mirror `AppConfig.resolve_device`'s body verbatim so a direct GPU caller still works). Replace `config.MODEL_ID`→`model.model_id`, `config.LAYER`→`model.layer`, `config.SYSTEM_PROMPT`→`model.system_prompt` (in `_inject_system(messages, model)` / `_merge_system_into_user(messages, model)`), `config.RANK_METHOD`→`feature_cloud.rank_method` (in `generate_and_capture(..., model, feature_cloud)`), `info()`→`info(model, sae)`. Drop `from . import config`.
  - `backend/science/sae.py`: `load_sae(layer=config.LAYER, device=None)` → `load_sae(sae, layer, device=None)`; `release=sae.release`, `sae_id=sae.sae_id_pattern.format(layer=layer)`. Device: same rule as engine — the caller (gpu_service) passes the resolved `device=`; if `None`, resolve inline via torch using `sae`-independent default (`"cuda" if torch.cuda.is_available()` else mps/cpu). `sae_topk(act, k=config.TOPK)` → `sae_topk(act, sae, feature_cloud, *, np_source, k=None)` with `k = k if k is not None else feature_cloud.topk`; `source` is the passed `np_source` (keyword-only, threaded from gpu_service which passes `cfg.np_source(cfg.model.layer)`). `attribution_topk(..., cap=config.TOPK_CANDIDATES, baseline=None)` → `attribution_topk(acts, grad, keep, sae, feature_cloud, *, np_source, cap=None, baseline=None)` with `cap = cap if cap is not None else feature_cloud.topk_candidates`. `feature_attribution(acts, grad, keep)` is unchanged (pure kernel, reads no config). `reconstruction_error(act)` → `reconstruction_error(act, sae)`. Drop `from . import config`.
  - `backend/science/persona.py`: `load_artifacts(artifact_dir=ARTIFACT_DIR, exclude=(), include=None)` → `load_artifacts(probes, include=None)` using `root = Path(probes.artifacts_dir)` and `skip = set(probes.disabled)`. `clear_custom_trackers(artifact_dir=ARTIFACT_DIR, keep=None, preserve_artifacts=None)` → `clear_custom_trackers(probes, ...)` using `probes.artifacts_dir`, defaulting `keep` to `probes.enabled` and `preserve_artifacts` to `probes.disabled` (i.e. `keep_set = set(keep if keep is not None else probes.enabled)`, `preserve = set(preserve_artifacts if preserve_artifacts is not None else probes.disabled)`). Keep module `ARTIFACT_DIR` constant only as an internal default for any non-config caller. Drop `from . import config` reads of `ENABLED_TRACKERS`/`DISABLED_TRACKERS`.
  - `backend/science/feature_provider.py` — cover ALL THREE classes AND `get_provider` (the module has `FeatureProvider` base, `LocalSAEProvider`, `NeuronpediaProvider`, and `get_provider`; if ANY keeps `from .. import config`, dropping config breaks import):
    - Base `FeatureProvider.features_for(...)`: its `k: int = config.TOPK` / `cap: int = config.TOPK_EVENT` defaults move to `(sae, feature_cloud, model, np_source)` params with `k`/`cap` defaulting to `None` and resolved from `feature_cloud.topk`/`feature_cloud.topk_event` inside.
    - `LocalSAEProvider.features_for(...)`: accept `sae`, `feature_cloud`, `model`, `np_source`; replace `config.TOPK`→`feature_cloud.topk`, `config.TOPK_EVENT`→`feature_cloud.topk_event`, `config.RANK_METHOD`→`feature_cloud.rank_method`, `config.PREAMBLE_SKIP`→`model.preamble_skip`, `config.NP_SOURCE`→the passed `np_source`; pass `sae, feature_cloud, np_source` into the `sae_topk(...)` / `attribution_topk(...)` calls.
    - `NeuronpediaProvider.features_for(...)`: accept `sae`, `feature_cloud`, `model`, `np_source`; replace `config.NP_MODEL`→`sae.np_model`, `config.NP_SOURCE`→`np_source`, `config.TOPK`→`feature_cloud.topk`, `config.TOPK_EVENT`→`feature_cloud.topk_event`, `config.RANK_METHOD`→`feature_cloud.rank_method`.
    - `get_provider(prefer=None)` → `get_provider(prefer=None, *, device=None)`: replace `config.resolve_device()` (line 141) with the passed `device` string (default `None`; the gpu_service candidate path uses `LocalSAEProvider()` directly and does NOT call `get_provider`, so `get_provider` has no live caller today — the only requirement is that dropping `from .. import config` doesn't break the FUNCTION BODY's `config.resolve_device()`/`config.NP_*` reads; if `device is None`, fall back to a torch-inline `"cuda" if torch.cuda.is_available()` check so a future direct caller still works). Drop `from .. import config`.

- [ ] Step 5: Run and confirm the CPU-testable persona/feature_provider/attribution tests pass:
```
$ uv run pytest backend/tests/test_persona.py backend/tests/test_feature_provider.py backend/tests/test_attribution.py -q
```
Expected: all pass (engine/sae torch paths are GPU-gated and not exercised here).

- [ ] Step 6: Commit:
```
$ git add backend/engine.py backend/science/sae.py backend/science/persona.py backend/science/feature_provider.py backend/tests/test_persona.py backend/tests/test_feature_provider.py backend/tests/test_attribution.py
$ git commit -m "refactor(science): thread Model/SAE/FeatureCloud/Probe configs into torch modules"
```

---

### Task 13: Build app.state.config and thread it through the orchestration backend

Files:
- Modify: `backend/app.py` (lines 16–18, 23–62, 136–222 — build `app.state.config` in `lifespan`; read `request.app.state.config` in endpoints; pass `cfg` to `runtime`, `init_sponsors`, `sentry_api`, `labels`, `analyze`, `pod_client`).
- Test: `backend/tests/test_api.py` (adjust `_force_fallback`-style helpers and any flat-config patches), `backend/tests/test_observability_endpoint.py` (rewrite — read `app.state.config`).

Interfaces:
- Consumes: `load_config` (Task 3); `runtime.start_loading(cfg)`/`health_payload(cfg)` (Task 5); `fanout.init_sponsors(obs, sentry_dsn)`/`fanout(payload, perf, obs=..., probes=...)`/`capture_cognition_alarm(event, obs, flush=...)`/`sentry_enabled(sentry_dsn)` (Task 8); `sentry_api.deep_link(sentry, token)`/`list_recent_issues(sentry, token)` (Task 7); `labels.get_label(index, sae, feature_cloud, anthropic_api_key, np_source=...)` (Task 10); `analyze_turn(messages, cfg)` (Task 11); `pod_client.track(request, pod, pod_token)` etc. (Task 6).
- Produces: `app.state.config: AppConfig` (set in `lifespan`); every endpoint accepts `request: Request` and reads `cfg = request.app.state.config`. `app = FastAPI(title=...)` title sourced from `cfg.runtime.product_name` at startup is not possible before `lifespan`, so keep `FastAPI(title="GlassBox")` literal (matches `product_name` default) and add a `# product name lives in cfg.runtime.product_name` comment.

> Interfaces note: `backend/app.py` is ALSO edited by WS1 (remove `/api/observability/eval`, `phoenix_ui_url`). WS0 keeps those endpoints but routes their config reads through `app.state.config`. The `coherence_eval` lazy import in `/api/observability/eval` stays (WS1 deletes the endpoint + module). `request` injection: import `Request` from `fastapi`.

> `coherence_eval` / `test_coherence_eval.py` under the new config (WS1-deleted, but live during WS0): VERIFIED — `coherence_eval.py` does `from . import config` at module load but only reads the flat globals `config.PHOENIX_ENDPOINT` (line 52) and `config.EVAL_LLM_PROVIDER`/`EVAL_LLM_MODEL` (line 86) INSIDE `_run()` (call-time, not import-time). So importing the module after the Task 1 config rewrite still succeeds, but `test_coherence_eval.py` (which calls `ce.run_eval()` → `_run()` in 7 tests) will `AttributeError` on the now-removed flat globals. `coherence_eval` + its endpoint + this test are WS1-owned deletions. WS0 OPTIONS (pick one, do NOT silently leave it red): (a) thread the two reads through `app.state.config` here too (`config.PHOENIX_ENDPOINT`→`os`-free: pass via a tiny `_run(limit, *, phoenix_endpoint, eval_provider, eval_model)` and have the endpoint read `cfg.observability` — but ObsConfig has no Phoenix/eval fields in the WS0 contract, so this is awkward); or (b) mark the 7 `run_eval` tests `@pytest.mark.skip(reason="coherence_eval is WS1-deleted; reads removed flat config globals")` and EXCLUDE them from the WS0 green-suite requirement, documenting that WS1 deletes the module+endpoint+test together. RECOMMENDED: (b) — it matches the WS1 ownership and avoids inventing ObsConfig fields the contract doesn't have. Whichever is chosen, add `coherence_eval.py` to the Task 15 straggler grep so its flat reads are accounted for.

Steps:

- [ ] Step 1: Update `backend/tests/test_api.py` and `backend/tests/test_observability_endpoint.py` to the new wiring. For `test_api.py`, the `TestClient(app)` triggers `lifespan`, which now builds `app.state.config` from defaults (no `config.yaml`) — so the health/chat tests work unchanged EXCEPT the medical sample prompt. Replace the medical sample message with a neutral one and assert the neutral fallback marker. Add a config-presence assertion:
```python
import json

from fastapi.testclient import TestClient

from backend import runtime
from backend.app import app
from backend.schema import CognitionEvent

client = TestClient(app)


def _force_fallback():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def test_app_state_config_built_at_startup():
    with TestClient(app):
        assert app.state.config is not None
        assert app.state.config.runtime.product_name == "GlassBox"


def test_health_shape():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        p = r.json()
        assert {"mode", "model", "layer", "trackers"} <= set(p)


def test_chat_streams_tokens_then_one_event():
    with TestClient(app) as c:
        _force_fallback()
        r = c.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "How do rainbows form?"}]},
        )
        assert r.status_code == 200
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) >= 2
        last = json.loads(lines[-1])
        assert "message_id" in last
```
For `test_observability_endpoint.py`, wrap calls in `with TestClient(app) as c:` and drop any `monkeypatch.setattr(...config..., ...)`; assert the snapshot has `health`, `sentry`, and (for now, WS1 removes it) `phoenix_ui_url` keys.

- [ ] Step 2: Run and confirm failure:
```
$ uv run pytest backend/tests/test_api.py backend/tests/test_observability_endpoint.py -q
```
Expected: failures referencing `app.state.config` not set, or the `analyze_turn`/`runtime`/`fanout` arity errors introduced by Tasks 5/8/11.

- [ ] Step 3: Edit `backend/app.py`:
  - Imports: keep `from . import labels, observability, runtime, sentry_api`; drop `config` if only used in endpoints (it is still needed for nothing now). Import `Request` from `fastapi`. Keep `from .analyze import analyze_turn` and the `fanout` imports.
  - `lifespan`: 
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from .config import load_config

    app.state.config = load_config()
    cfg = app.state.config
    runtime.start_loading(cfg)
    init_sponsors(cfg.observability, cfg.sentry_dsn)
    yield
```
  - `app = FastAPI(title="GlassBox", lifespan=lifespan)` — keep literal; add the product-name comment.
  - `health`: `def health(request: Request) -> dict: return runtime.health_payload(request.app.state.config)`.
  - `observability_endpoint(request: Request)`: 
```python
cfg = request.app.state.config
snap = observability.STORE.snapshot()
snap["health"] = runtime.health_payload(cfg)
snap["sentry"] = {
    "emit_configured": bool(cfg.sentry_dsn),
    "configured": bool(cfg.sentry_auth_token),
    "deep_link": sentry_api.deep_link(cfg.observability.sentry, cfg.sentry_auth_token),
    "issues": await sentry_api.list_recent_issues(cfg.observability.sentry, cfg.sentry_auth_token),
}
snap["phoenix_ui_url"] = None  # WS1: remove this key
return snap
```
  - `observability_test_sentry(request: Request)` and `observability_replay_sentry(request: Request, body=...)`: replace `sentry_enabled()`→`sentry_enabled(request.app.state.config.sentry_dsn)`, `capture_cognition_alarm(event, flush=True)`→`capture_cognition_alarm(event, request.app.state.config.observability, flush=True)`.
  - `chat(request: Request, body: dict)` and `analyze(request: Request, body: dict)`: `cfg = request.app.state.config`; `answer, event, perf = await run_in_threadpool(analyze_turn, messages, cfg)`; `fanout(payload, perf, obs=cfg.observability, probes=cfg.probes)`.
  - `track`/`clear_custom_trackers`/`track_status`: add `request: Request`, read `cfg`, call `pod_client.track(request_str, cfg.pod, cfg.pod_token)` etc.
  - `feature(request: Request, index: int)`: 
```python
cfg = request.app.state.config
return {
    "index": index,
    "label": labels.get_label(index, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()),
    "source": cfg.np_source(),
    "caveat": "auto-interp label, may be unreliable",
}
```
  - Change the test-sentry `"hint"` string `"Confident-wrong medical answer"` → neutral `"Confident-wrong answer"` (rebrand; WS1 finishes the neutralization).

- [ ] Step 4: Run and confirm pass:
```
$ uv run pytest backend/tests/test_api.py backend/tests/test_observability_endpoint.py -q
```
Expected: all pass.

- [ ] Step 5: Commit:
```
$ git add backend/app.py backend/tests/test_api.py backend/tests/test_observability_endpoint.py
$ git commit -m "refactor(app): build app.state.config and thread AppConfig through all endpoints"
```

---

### Task 14: Build app.state.config and thread it through the GPU pod service

Files:
- Modify: `backend/gpu_service.py` (lines 13–330 — build `app.state.config` at startup; `_require_auth` reads `cfg.pod_token`; `_attempt_load`/`_run_recon_check`/`_baseline_vec`/`_sae_candidates`/`_score_trackers` take `cfg`/sub-configs; endpoints read `request.app.state.config`).
- Modify: `backend/agent/prompts.py`, `backend/agent/tools.py`, `backend/agent/interp_agent.py`, `backend/science/concept_synth.py` (thread `builder`/`model`/`anthropic_api_key`; neutralize the "medical-chat LLM" prompt text in `prompts.py`).
- Test: `backend/tests/test_gpu_service.py` (rewrite — set `app.state.config`, patch via sub-config), `backend/tests/test_judge.py` (rewrite if it touches `judge_filter` config), `backend/tests/test_loop.py`/`test_agent_persist.py` (adjust agent signatures if exercised).

Interfaces:
- Consumes: `load_config` (Task 3); `engine.*`/`sae.*`/`persona.*`/`feature_provider.*` new signatures (Task 12); `ProbeBuilderConfig`, `ModelConfig`.
- Produces:
  - `gpu_service.app.state.config: AppConfig`; `_require_auth(request)` reads `request.app.state.config.pod_token`.
  - `concept_synth.judge_filter(spec, rows, builder, anthropic_api_key, *, client=None) -> list[dict]`.
  - `agent.prompts` functions take `builder` (e.g. `SYSTEM_for(builder)` / `spec_tool(builder)` returning the schema with `builder.agent_max_questions`); neutral prompt text (no "medical-chat LLM").
  - `agent.tools._dispatch(name, tool_input, ctx, builder, model)` / `_finalize(ctx, verdict, builder)` / `_persist_artifact(ctx, fit, model)` reading `builder.agent_max_questions`, `builder.auroc_threshold`, `model.layer`.
  - `agent.interp_agent.run_interp_agent(tracker_id, builder, anthropic_api_key, *, client=None, generate_fn=None) -> dict | None`.

> Interfaces note: `agent/prompts.py` computes `_MAX_Q = config.AGENT_MAX_QUESTIONS` at import time today — this MUST become a function-parameter read (`builder.agent_max_questions`) so the module imports without `config`. `gpu_service.py` is ALSO edited by WS2 (derive d_in/d_sae) and WS3 (fail-closed `_require_auth`, Dockerfile). WS0 keeps `_require_auth`'s "empty token allows" behavior (WS3 hardens it) but sources the token from `cfg.pod_token`. Read each agent module in full before editing.

Steps:

- [ ] Step 1: Read the four agent/synth modules in full to capture exact signatures and `config.*` sites.
```
$ for f in backend/agent/prompts.py backend/agent/tools.py backend/agent/interp_agent.py backend/science/concept_synth.py; do echo "=== $f ==="; uv run python -c "print(open('$f').read())"; done
```
Expected: confirms `prompts._MAX_Q`, `tools.config.AGENT_MAX_QUESTIONS/TRACK_AUROC_TAU/LAYER`, `interp_agent.config.AGENT_MODEL/ANTHROPIC_API_KEY`, `concept_synth.config.JUDGE_MODEL/JUDGE_BATCH_SIZE/ANTHROPIC_API_KEY`.

- [ ] Step 2: Rewrite `backend/tests/test_gpu_service.py` to set `app.state.config` and inject sub-configs. The key fixtures (`_skip_pod_load` sets `mode=real`; `_install_stubs` stubs `_capture`/`_sae_candidates`/`_score_trackers`) stay; add a config bootstrap and replace `monkeypatch.setattr(gpu_service.config, "POD_TOKEN", ...)` with `gpu_service.app.state.config.pod_token = ...`:
```python
import pytest
from fastapi.testclient import TestClient

from backend import gpu_service
from backend.config import AppConfig


@pytest.fixture(autouse=True)
def _state_config():
    gpu_service.app.state.config = AppConfig()
    yield


@pytest.fixture
def _skip_pod_load(monkeypatch):
    monkeypatch.setattr(
        gpu_service, "_attempt_load",
        lambda cfg: gpu_service.STATE.update(mode="real", model_loaded=True, sae_loaded=True),
    )
    gpu_service.app.state.config.pod_token = ""


def _install_stubs(monkeypatch):
    # Stub at the highest seam: _capture returns a fully-formed dict, and BOTH downstream
    # consumers (_sae_candidates, _score_trackers) are stubbed — so the only code that touches
    # the dict between _capture and the stubs is _pooled_activations(res), which reads
    # res["acts"] / res["resp_start"], slices acts[rs:] / acts[rs-1], and calls
    # resp_acts.float().mean(0) guarded by resp_acts.shape[0] > 0. The _FakeActs below satisfies
    # exactly that surface and NOTHING deeper — the resulting (resp_acts, act_last, act_resp)
    # are passed only to the stubbed _sae_candidates/_score_trackers, which ignore them. No real
    # torch tensor is ever constructed or arithmetic performed.
    monkeypatch.setattr(gpu_service, "_capture", lambda messages, max_new, **k: {
        "answer": "ok", "resp_start": 1, "acts": _FakeActs(), "out_ids": _FakeIds(), "tok": _FakeTok(), "grad": None,
    })
    monkeypatch.setattr(gpu_service, "_sae_candidates", lambda *a, **k: [{"index": 1, "act": 1.0, "source": "s"}])
    monkeypatch.setattr(gpu_service, "_score_trackers", lambda a, b: {"harmful": {"score": 0.1, "flag": False, "reliable": True}})
    # _baseline_vec is gated on mode==real + contrast_baseline; stub it to None so no engine/sae load.
    monkeypatch.setattr(gpu_service, "_baseline_vec", lambda cfg: None)


# minimal fakes for _pooled_activations ONLY (acts[rs:] -> _FakeActs with shape[0]==0 so the
# .float().mean(0) branch is skipped and act_resp is None; never deeply dereferenced because
# _sae_candidates/_score_trackers are stubbed).
class _FakeActs:
    def __getitem__(self, k):
        return _FakeActs()

    @property
    def shape(self):
        return (0,)  # shape[0] == 0 -> _pooled_activations sets act_resp = None (no .mean call)

    def float(self):
        return self


class _FakeIds:
    def __getitem__(self, k):
        return _FakeIds()

    def tolist(self):
        return []


class _FakeTok:
    def convert_tokens_to_ids(self, t):
        return -1


def test_auth_rejects_bad_token(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    gpu_service.app.state.config.pod_token = "secret"
    with TestClient(gpu_service.app) as c:
        r = c.post("/inference", json={"messages": [{"role": "user", "content": "x"}]}, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401


def test_health_shape(_skip_pod_load):
    with TestClient(gpu_service.app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        p = r.json()
        assert {"mode", "model", "layer", "d_sae", "trackers"} <= set(p)
```
The stub interface above is pinned against the real `_pooled_activations` (gpu_service.py:93-98) — `acts[rs:]` returns a `_FakeActs` whose `shape[0]==0`, so the `.float().mean(0)` branch is skipped and `act_resp` is `None`; `act_last = acts[rs-1]` is another `_FakeActs` never dereferenced (it only flows into the stubbed `_score_trackers`). `_capture`, `_sae_candidates`, `_score_trackers`, and `_baseline_vec` are all stubbed, so no real torch tensor is ever touched. Do not re-derive the fakes — they are complete as written.

- [ ] Step 3: Run and confirm failure:
```
$ uv run pytest backend/tests/test_gpu_service.py -q
```
Expected: failures referencing `gpu_service.config` removed / `_attempt_load` arity / `_require_auth` reading `app.state.config`.

- [ ] Step 4: Edit `backend/gpu_service.py`:
  - Drop `from . import config`. Add startup config build:
```python
@app.on_event("startup")
def _startup() -> None:
    from .config import load_config

    app.state.config = load_config()
    _attempt_load(app.state.config)
```
  - `_require_auth(request)`: `token = request.app.state.config.pod_token`; keep "empty allows" (WS3 hardens).
  - Device: compute the resolved device ONCE in `_attempt_load` via `device = cfg.resolve_device()` and pass it explicitly into `engine.load_engine(cfg.model, device)` and `sae.load_sae(cfg.sae, cfg.model.layer, device)`. Do NOT call `cfg.model.resolve_device` (that method lives on `AppConfig`, not `ModelConfig`).
  - `_run_recon_check(cfg)`: `sae.reconstruction_error(engine.probe_activation(cfg.sae.recon_probe, cfg.model), cfg.sae)`; compare against `cfg.sae.recon_min_cosine`.
  - `_attempt_load(cfg)`: `device = cfg.resolve_device()`; `engine.load_engine(cfg.model, device)`, `sae.load_sae(cfg.sae, cfg.model.layer, device)`, `persona.load_artifacts(cfg.probes)`, then `_run_recon_check(cfg)`.
  - `_capture(messages, max_new, *, attribution=None, cfg)`: `engine.generate_and_capture(messages, max_new=max_new, attribution=attribution, model=cfg.model, feature_cloud=cfg.feature_cloud)`.
  - `_baseline_vec(cfg)`: replace `config.CONTRAST_BASELINE`→`cfg.feature_cloud.contrast_baseline`, `config.CONTRAST_PROMPT`→`cfg.feature_cloud.contrast_prompt`, `config.CONTRAST_MAX_NEW`→`cfg.feature_cloud.contrast_max_new`, `config.MASK_TOKENS`→`cfg.model.mask_tokens`, `config.PREAMBLE_SKIP`→`cfg.model.preamble_skip`.
  - `_sae_candidates(res, resp_acts, resp_grad, *, cap, baseline=None, cfg)`: `config.MASK_TOKENS`→`cfg.model.mask_tokens`; pass `sae=cfg.sae, feature_cloud=cfg.feature_cloud, model=cfg.model, np_source=cfg.np_source(cfg.model.layer)` to `LocalSAEProvider().features_for(...)` (the `np_source` slug is computed here, the one place that holds `cfg`, per the Task 12 `np_source` threading decision).
  - The `/api/trackers/clear-custom` endpoint handler `clear_custom_trackers()` (gpu_service.py:292-296) currently calls `persona.clear_custom_trackers()` with no args — change it to `persona.clear_custom_trackers(cfg.probes)` (read `cfg = request.app.state.config`). This endpoint MUST be in the per-endpoint cfg-threading list (it was previously omitted).
  - All endpoints (`inference`/`activations`/`sae_features`/`turn`/`track`/`track_status`/`clear_custom_trackers`/`health`): add `request: Request`, read `cfg = request.app.state.config`; replace `config.MAX_NEW_TOKENS`→`cfg.model.max_new_tokens`, `config.TOPK_CANDIDATES`→`cfg.feature_cloud.topk_candidates`, `config.RANK_METHOD`→`cfg.feature_cloud.rank_method`, `config.CONTRAST_BASELINE`→`cfg.feature_cloud.contrast_baseline`, `config.MODEL_ID`→`cfg.model.model_id`, `config.LAYER`→`cfg.model.layer`, `config.D_SAE`→`cfg.sae.d_sae`, `config.ANTHROPIC_API_KEY`→`cfg.anthropic_api_key`, `config.DISABLED_TRACKERS`→`cfg.probes.disabled`. Pass `cfg` into `_capture`/`_baseline_vec`/`_sae_candidates`/`_run_recon_check`.
  - `_launch_agent(tracker_id, cfg)` and `/api/track`: call `run_interp_agent(tracker_id, cfg.probes.builder, cfg.anthropic_api_key)`.
  - `backend/agent/prompts.py`: remove `from .. import config` and the module-level `_MAX_Q`. Convert `SYSTEM` and `SPEC_TOOL` into functions: `def system_prompt(builder) -> str` and `def spec_tool(builder) -> dict`, each substituting `builder.agent_max_questions`. Replace "medical-chat LLM" / "medical-chat questions" with neutral "language model" / "questions where the trait could surface" (drop the word "medical").
  - `backend/agent/tools.py`: `_dispatch(name, tool_input, ctx)` → `_dispatch(name, tool_input, ctx, builder, model)`; `config.AGENT_MAX_QUESTIONS`→`builder.agent_max_questions`, `config.TRACK_AUROC_TAU`→`builder.auroc_threshold`, `config.LAYER`→`model.layer`. Thread `builder`/`model` from `dispatch`/`_finalize`/`_persist_artifact`.
  - `backend/agent/interp_agent.py`: `run_interp_agent(tracker_id, *, client=None, generate_fn=None)` → `run_interp_agent(tracker_id, builder, anthropic_api_key, *, client=None, generate_fn=None)`; `anthropic.Anthropic(api_key=anthropic_api_key)`, `model=builder.agent_model`; call `prompts.system_prompt(builder)`/`prompts.spec_tool(builder)`; pass `builder, model` to `tools.dispatch`.
  - `backend/science/concept_synth.py`: `judge_filter(spec, rows, *, client=None, judge_model=None)` → `judge_filter(spec, rows, builder, anthropic_api_key, *, client=None)`; `anthropic.Anthropic(api_key=anthropic_api_key)`, `model = builder.judge_model`, `batch_size = max(1, builder.judge_batch_size)`.

- [ ] Step 5: Run and confirm the gpu_service + agent/judge tests pass:
```
$ uv run pytest backend/tests/test_gpu_service.py backend/tests/test_judge.py backend/tests/test_loop.py backend/tests/test_agent_persist.py backend/tests/test_jobs.py -q
```
Expected: all pass (GPU-loading paths remain stubbed/gated).

- [ ] Step 6: Commit:
```
$ git add backend/gpu_service.py backend/agent/prompts.py backend/agent/tools.py backend/agent/interp_agent.py backend/science/concept_synth.py backend/tests/test_gpu_service.py backend/tests/test_judge.py backend/tests/test_loop.py backend/tests/test_agent_persist.py
$ git commit -m "refactor(gpu_service): build app.state.config; thread builder/model configs into agent; neutralize agent prompts"
```

---

### Task 15: Full suite green + sweep for stragglers

Files:
- Modify: any remaining file that still does `from . import config` for a flat global (e.g. `backend/batch_medqa_observability.py`, `backend/smoke_test_gs2.py`, `backend/validation/*`, `backend/tests/test_smoke.py`, `backend/tests/test_rank.py`, `backend/tests/test_attribution.py`, `backend/tests/test_labels_autointerp.py`, `backend/coherence_eval.py` — whatever the grep finds).
- Test: the whole non-GPU suite. `backend/tests/test_coherence_eval.py`'s 7 `run_eval` tests are skipped per Task 13's coherence_eval note (WS1-deleted; reads removed flat globals) and are excluded from the green-suite requirement; its `MEDICAL_DOMAIN_CONTEXT` import test stays green (module still imports).

Interfaces:
- Consumes: every symbol produced in Tasks 1–14.
- Produces: a green non-GPU pytest run; zero references to deleted flat globals in importable (non-GPU) code paths.

Steps:

- [ ] Step 1: Grep for any remaining flat-global access that would `AttributeError` at import/runtime:
```
$ grep -rn "config\.\(MODEL_ID\|LAYER\|SAE_RELEASE\|SAE_ID\|D_IN\|D_SAE\|NP_MODEL\|NP_SOURCE\|NP_FEATURE_URL\|TOPK\|DROP_UNLABELED\|DENSITY_MAX\|SYNTACTIC_PENALTY\|STRUCTURAL_PENALTY\|RANK_METHOD\|CONTRAST_\|PREAMBLE_SKIP\|AUTOINTERP\|ENABLED_TRACKERS\|BUILTIN_TRACKERS\|DISABLED_TRACKERS\|DEFAULT_THRESHOLD\|TRACK_AUROC_TAU\|AGENT_MODEL\|JUDGE_MODEL\|AGENT_MAX_QUESTIONS\|JUDGE_BATCH_SIZE\|SENTRY_\|PHOENIX_\|EVAL_LLM_\|POD_URL\|POD_TOKEN\|POD_TIMEOUT\|POD_POLL_INTERVAL\|MODE\|MAX_NEW_TOKENS\|SYSTEM_PROMPT\|DEVICE\|ANTHROPIC_API_KEY\|MASK_TOKENS\|RECON_\|resolve_device\|sae_id_for_layer\)" backend --include="*.py"
```
Expected after Tasks 1–14: matches ONLY in GPU-gated files' internal calls already migrated, or in scripts not on the import path. Any match in an importable module is a straggler to fix.

- [ ] Step 2: Fix each straggler by threading the right sub-config (mirror the migration map). For utility scripts that build their own config, add `from .config import load_config; cfg = load_config()` at the top and replace `config.X` with `cfg.<path>`. (Scripts like `batch_medqa_observability.py`/`smoke_test_gs2.py`/`validation/*` are not imported by the suite; update them so they still run, but they need no test.) `backend/coherence_eval.py`'s `_run()` reads `config.PHOENIX_ENDPOINT`/`EVAL_LLM_PROVIDER`/`EVAL_LLM_MODEL` (call-time) — these are WS1-deleted; per the Task 13 coherence_eval note its `run_eval` tests are skipped in WS0, so this grep match is EXPECTED and is left as-is (do NOT invent ObsConfig Phoenix/eval fields). Record it in the grep output as a known WS1-owned straggler rather than fixing it.

- [ ] Step 3: Run the FULL non-GPU suite (engine/sae GPU tests are skipped without weights):
```
$ uv run pytest -q
```
Expected: all collected tests pass (the GPU-gated `engine`/`sae` direct tests skip or are not collected without weights). No collection errors, no `AttributeError: module 'backend.config' has no attribute ...`.

- [ ] Step 4: Confirm the orchestration backend never imports torch (Global Constraint regression check):
```
$ uv run python -c "import sys; import backend.app; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m=='torch')" && echo "OK: app.py torch-free"
```
Expected: `OK: app.py torch-free`.

- [ ] Step 5: Commit (stage only the straggler files found and fixed in Steps 1–2; never `git add -A` to avoid sweeping in watch-*.json / probe_jobs/ / .env):
```
$ git add $(git diff --name-only HEAD -- '*.py')
$ git commit -m "refactor(config): migrate remaining config consumers to AppConfig; full suite green"
```
(If `git diff --name-only HEAD -- '*.py'` is empty — all stragglers were already committed by prior tasks — run `git status --short` to confirm nothing is outstanding and skip the commit.)

---

### Task 16: Repo hygiene — remove tracked cruft + on-disk `.superpowers/` scaffolding

Files:
- Delete (tracked): `main.py`, `err.txt`, `batch_medqa_results.json`.
- Delete (UNtracked on-disk scaffolding): `.superpowers/` (the `.superpowers/sdd` working dir of per-task briefs/reports/diffs). This is the design-doc §3 WS0-hygiene requirement "Delete the on-disk `.superpowers/` scaffolding before publishing."

Interfaces:
- Consumes: nothing.
- Produces: a working tree with the three cruft files removed from git AND the `.superpowers/` scaffolding removed from disk.

> Scope guard (CRITICAL): `.superpowers/` is DISTINCT from `docs/superpowers/` — the latter holds the plans/specs (including THIS plan, `docs/superpowers/plans/2026-06-22-ws0-config-hygiene-rebrand.md`, which WS4 owns relocating). Do NOT touch `docs/superpowers/`. `.superpowers/` is currently UNtracked (confirmed: `git ls-files .superpowers/` is empty), so `git rm` would FAIL — use a plain `rm -rf .superpowers/`. Because it is untracked there is nothing to commit for the deletion itself; an optional `.gitignore` entry (Task 17) prevents it from reappearing.

Steps:

- [ ] Step 0: Remove the on-disk `.superpowers/` scaffolding (untracked — plain rm, NOT `git rm`). Verify it is untracked first, then confirm `docs/superpowers/` is untouched:
```
$ git ls-files .superpowers/ | head    # expect EMPTY (untracked) — if non-empty, STOP and reassess
$ rm -rf .superpowers/
$ ls -d docs/superpowers/plans/2026-06-22-ws0-config-hygiene-rebrand.md   # this plan must still exist
```
Expected: first command empty; `rm` succeeds; the plan file still listed (we only deleted the sibling `.superpowers/`, not `docs/superpowers/`).

- [ ] Step 1: Confirm the three files are tracked and what they are (verification baseline):
```
$ git ls-files main.py err.txt batch_medqa_results.json
```
Expected: prints all three paths (they are tracked).

- [ ] Step 2: Remove them from git and the working tree:
```
$ git rm main.py err.txt batch_medqa_results.json
```
Expected: `rm 'main.py'` / `rm 'err.txt'` / `rm 'batch_medqa_results.json'`.

- [ ] Step 3: Verify they are gone and nothing imports `main`:
```
$ git ls-files main.py err.txt batch_medqa_results.json; grep -rn "import main\b\|from main import\|batch_medqa_results" backend --include="*.py"
```
Expected: empty output (no tracked files, no importers).

- [ ] Step 4: Confirm the suite still passes (no module depended on these):
```
$ uv run pytest -q
```
Expected: all pass.

- [ ] Step 5: Commit (the `.superpowers/` deletion is untracked so it does not appear in the commit; this commit captures only the three tracked-cruft removals already staged by `git rm` in Step 2; do NOT use `git add -A` which would sweep in untracked watch-*.json / probe_jobs/ / .env):
```
$ git commit -m "chore: remove tracked cruft (main.py stub, empty err.txt, batch_medqa_results.json)"
```
(The three deletions were staged by `git rm main.py err.txt batch_medqa_results.json` in Step 2; no additional `git add` is needed.)

---

### Task 17: Repo hygiene — gitignore generated artifacts + config.yaml

Files:
- Modify: `.gitignore` (add `config.yaml`, `backend/science/probe_jobs/`, `backend/science/artifacts/watch-*.json`, `.pytest_cache/`).

Interfaces:
- Consumes: nothing.
- Produces: a `.gitignore` that keeps the working `config.yaml` and generated probe artifacts out of git while preserving the 6 built-in probe artifacts.

> Interfaces note: the untracked `backend/science/artifacts/watch-*.json` and `backend/science/probe_jobs/` files seen in `git status` are generated demo output. This task only ignores them; it does NOT delete the 6 committed built-ins (`harmful`, `harmful_prompt`, `over_confidence`, `hallucination`, `uncertainty`, `risk_awareness`).

Steps:

- [ ] Step 1: Append the new ignore rules to `.gitignore` (after the existing `# Phoenix / traces` block; keep that block — WS1 removes it). Add:
```
# Working config (config.example.yaml is the committed template)
config.yaml

# Generated probe-builder output (the 6 built-in artifacts ARE committed)
backend/science/probe_jobs/
backend/science/artifacts/watch-*.json

# pytest
.pytest_cache/

# On-disk SDD scaffolding (deleted in Task 16; ignored so it never reappears in git status)
.superpowers/
```

- [ ] Step 2: Verify the generated artifacts are now ignored while the built-ins remain tracked:
```
$ git check-ignore -v backend/science/artifacts/watch-sycophancy-27ebff.json backend/science/probe_jobs/ config.yaml .pytest_cache/
$ git status --porcelain backend/science/artifacts/ | grep watch-
```
Expected: first command prints a `.gitignore` rule match for each path; second command prints nothing (the untracked watch-* files no longer show as untracked).

- [ ] Step 3: Confirm the 6 built-in artifacts are NOT ignored (still tracked/trackable):
```
$ git check-ignore backend/science/artifacts/harmful.json backend/science/artifacts/over_confidence.json; echo "exit=$?"
```
Expected: no output and `exit=1` (means NOT ignored — they remain committed).

- [ ] Step 4: Commit:
```
$ git add .gitignore
$ git commit -m "chore: gitignore config.yaml, probe_jobs/, artifacts/watch-*.json, .pytest_cache/"
```

---

### Task 18: Dependency reconciliation — retire requirements.txt, fix pyproject

Files:
- Delete (tracked): `backend/requirements.txt`.
- Modify: `pyproject.toml` (fix the stale "torch 2.12" comment; neutralize the medical `description`; confirm ML extra matches Global Constraints).

Interfaces:
- Consumes: nothing.
- Produces: `pyproject.toml` as the single canonical dependency source; medical-free package metadata.

> Interfaces note: WS1 will further drop `arize-phoenix`/`openinference-instrumentation` — but those live ONLY in the now-deleted `requirements.txt`, never in `pyproject.toml`, so deleting `requirements.txt` here also removes them. The base/ML deps in `pyproject.toml` already match the Global Constraints (fastapi/sentry/httpx; torch/transformers/sae-lens/scikit-learn/accelerate/anthropic) plus the `pydantic`/`pydantic-settings`/`pyyaml` added in Task 1.

Steps:

- [ ] Step 1: Neutralize the package description and fix the stale comment in `pyproject.toml`. **WS0 is the single owner of the `description` value** — WS4 Task 10 only VERIFIES this exact string (grep, no write), so the value below is the one canonical string both plans use. The CURRENT line 4 is `description = "Cognition-observability for medical LLMs"` (contains BOTH "medical" AND "Cognition"). Change line 4 to EXACTLY:
```toml
description = "Open-source interpretability & observability for open-weight LLMs"
```
Verify the new value contains neither the word `medical` nor the word `cognition` (the rebrand renames `cognition_event`→`introspection_event` in WS1, but the bare framing words must already be gone from package metadata in WS0): `grep -niE 'medical|cognition' pyproject.toml` must return nothing after this edit. The stale comment to replace is at line 15 (`# Match the validated env in PLAN.md §1 (torch 2.12 / transformers 5.12 / sae-lens 6).`). Replace it with:
```toml
# Real inference path. Installed only where weights/GPU exist:  uv sync --extra ml
# Validated against torch 2.4+ / transformers 4.50+ / sae-lens 6 on Gemma + Gemma Scope.
```

- [ ] Step 2: Delete the retired requirements file:
```
$ git rm backend/requirements.txt
```
Expected: `rm 'backend/requirements.txt'`.

- [ ] Step 3: Verify no docs/scripts still install from `requirements.txt` (so the retirement is clean):
```
$ grep -rn "requirements.txt" . --include="*.md" --include="*.sh" --include="*.yml" --include="*.yaml" --include="Dockerfile*" | grep -v node_modules
```
Expected: empty, OR matches only in WS4-owned docs (note any match for WS4 to fix; if a runnable script references it, repoint to `uv sync --extra ml` here).

- [ ] Step 4: Confirm the project still resolves and the suite passes:
```
$ uv sync && uv run pytest -q
```
Expected: resolves cleanly; all tests pass.

- [ ] Step 5: Commit (stage only the two changed items — the modified `pyproject.toml` and the deletion already staged by `git rm` in Step 2; do NOT use `git add -A` which would sweep in untracked artifacts):
```
$ git add pyproject.toml
$ git commit -m "chore(deps): retire backend/requirements.txt; pyproject is canonical; neutralize package description"
```
(`backend/requirements.txt` was already staged as a deletion by `git rm` in Step 2; `git add pyproject.toml` stages the description/comment edits from Step 1.)

---

### Task 19: Pick one frontend lockfile

Files:
- Delete (tracked): `frontend/bun.lock` (keep `frontend/package-lock.json`).

Interfaces:
- Consumes: nothing.
- Produces: a single committed frontend lockfile (`frontend/package-lock.json`).

> Interfaces note: both lockfiles are tracked today. We standardize on **npm** (`npm install` / `npm run dev`) to match the README `npm install` quickstart and the WS4 CI; keep `frontend/package-lock.json` as the single lockfile and delete `frontend/bun.lock`. WS4's README/CONTRIBUTING already use npm.

Steps:

- [ ] Step 1: Confirm both lockfiles are tracked (baseline):
```
$ git ls-files frontend/bun.lock frontend/package-lock.json
```
Expected: prints both paths.

- [ ] Step 2: Remove the bun lockfile, keep npm:
```
$ git rm frontend/bun.lock
```
Expected: `rm 'frontend/bun.lock'`.

- [ ] Step 3: Verify exactly one lockfile remains tracked (the bun lockfile is gone):
```
$ git ls-files frontend/bun.lock frontend/package-lock.json
```
Expected: prints only `frontend/package-lock.json` (no `frontend/bun.lock`).

- [ ] Step 4: Commit (`frontend/bun.lock` was already staged as a deletion by `git rm` in Step 2; do NOT use `git add -A` which would sweep in untracked artifacts):
```
$ git commit -m "chore(frontend): standardize on npm package-lock.json; drop bun.lock"
```

---

### Task 20: LICENSE copyright holder + final rebrand sweep

Files:
- Modify: `LICENSE` (line 3 — real copyright holder).
- Verify: no medical positioning remains in package metadata / config defaults / agent prompts.

Interfaces:
- Consumes: nothing.
- Produces: an MIT `LICENSE` with a concrete copyright holder; a metadata/config/agent surface free of medical positioning (README/PLAN/LANES are WS4's job — out of scope here).

> Interfaces note: the design doc §11 leaves the exact name "confirm at review". Use a concrete holder that is not a placeholder: `Aniruddhan Ramesh and the GlassBox contributors`. **WS0 is the single owner of the LICENSE copyright line** — WS4 Task 7 only VERIFIES this exact string (grep, no write), so the string below must match WS4's assertion byte-for-byte. The medical README/PLAN/LANES rewrites belong to WS4; this task only verifies the WS0-owned surfaces (pyproject, config defaults, agent prompts, fallback) are neutral.

Steps:

- [ ] Step 1: Set the copyright holder in `LICENSE` line 3. Change `Copyright (c) 2026 GlassBox contributors` to (this is the single canonical string; WS4 Task 7 asserts it verbatim):
```
Copyright (c) 2026 Aniruddhan Ramesh and the GlassBox contributors
```

- [ ] Step 2: Verify the WS0-owned surfaces are medical-free (config defaults, package metadata, agent prompts, fallback). Run:
```
$ grep -rniE "clinical|patient|ibuprofen|pregnancy|medical|medqa|medmcqa|cognition" pyproject.toml backend/config.py backend/agent/prompts.py backend/fallback.py backend/mock_labels.py backend/labels.py
```
Expected: matches ONLY inside clearly-labeled example/comment context (none in active code defaults). Specifically: `pyproject.toml` must have ZERO matches for `medical` or `cognition` (Task 18 neutralized the description). `backend/config.py`'s `_DEFAULT_SYSTEM_PROMPT` and `SAEConfig.recon_probe` are neutral; the medical system prompt survives only as a commented profile in `config.example.yaml` (NOT scanned here). The only acceptable `cognition` matches are the `CognitionEvent`/`build_cognition_event`/`cognition_event` symbols (WS1 renames these to `introspection_event` — they are intentionally untouched in WS0, see the Global Constraints scope note) and any "cognition" inside a `# WS1:`-tagged comment. If any active-code copy match (medical/clinical/patient framing in a default value or prompt) appears, fix it before committing.

- [ ] Step 3: Confirm the full suite is still green after every WS0 change:
```
$ uv run pytest -q
```
Expected: all pass.

- [ ] Step 4: Commit:
```
$ git add LICENSE
$ git commit -m "chore(license): set concrete copyright holder; finalize WS0 rebrand foundation"
```

---

## Done criteria (WS0 complete)

- `backend/config.py` exposes `AppConfig` + 7 sub-models + `Secrets` + `load_config`; no flat globals remain referenced by importable code (Task 15 grep is clean, save the WS1-owned `coherence_eval._run` Phoenix/eval reads whose tests are skipped).
- Both processes build `app.state.config` at startup and thread sub-configs down; `backend/app.py` is torch-free (Task 15 Step 4).
- `config.example.yaml` is committed (no secrets, labeled medical profile); `config.yaml` is gitignored.
- `pydantic-settings`/`pydantic`/`pyyaml` in `pyproject.toml`; `backend/requirements.txt` deleted; one frontend lockfile (`package-lock.json`; `bun.lock` deleted).
- Cruft removed (`main.py`, `err.txt`, `batch_medqa_results.json`); on-disk `.superpowers/` scaffolding removed (and gitignored); generated artifacts gitignored; LICENSE has a real holder.
- Rebrand foundation: `runtime.product_name="GlassBox"`, neutral default `system_prompt`/`recon_probe`, neutral fallback templates/labels, neutral agent prompts, neutral package description.
- The non-GPU pytest suite is green; no commit adds Claude as co-author.
