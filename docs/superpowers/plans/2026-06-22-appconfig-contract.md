# AppConfig Contract — Authoritative Reference

**Date:** 2026-06-22
**Status:** Authoritative. All five workstream plans (WS0–WS4) build on this.
**Source of truth for:** `backend/config.py` → `AppConfig` refactor (design doc §2).

This document pins the exact `AppConfig` shape, the loader behavior, the threading pattern,
and the migration map from today's flat `config.X` globals. It is derived from reading
`backend/config.py` in full and grepping every `config.` consumer across `backend/`. Any
field name or signature here is load-bearing — downstream planners treat it as fixed.

---

## 0. Design rules (locked)

- **One parent object.** `AppConfig` with six sub-models: `model`, `sae`, `probes`,
  `observability`, `pod`, `runtime`. Built once at startup, stored on `app.state`, threaded
  down. Replaces ~50 flat module-level globals.
- **Structure from `config.yaml`; secrets from env only.** The five secrets
  (`ANTHROPIC_API_KEY`, `SENTRY_DSN`, `SENTRY_AUTH_TOKEN`, `POD_TOKEN`, `HF_TOKEN`) NEVER appear in
  `config.yaml` / `config.example.yaml`. They load only from env / `.env`. Everything else loads
  from YAML with env override.
- **Boots with no `config.yaml`.** Absent YAML → Gemma + Gemma-Scope defaults, so the app still
  runs. `config.example.yaml` is the committed template; the working `config.yaml` is gitignored.
- **Defaults reproduce today's Gemma behavior EXCEPT the system prompt**, which becomes a
  neutral, non-medical assistant prompt (the medical prompt moves to a labeled
  `config.example.yaml` "medical" profile).
- **Pure-CPU modules** (`runtime`, `fanout`, `sentry_api`, `analyze`, `labels`) never import
  torch and receive only the sub-config they need. **Torch modules** (`engine`, `science/sae`,
  `science/persona`, `science/feature_provider`) receive `model` / `sae` / `probes`.

---

## 1. Pydantic models

Uses `pydantic` v2 + `pydantic-settings` (add `pydantic-settings>=2.0` to `pyproject.toml`;
`pydantic>=2.9` is already a dep). Secrets are a separate `Secrets` settings object so they are
structurally incapable of being populated from YAML.

```python
"""backend/config.py — unified AppConfig (replaces the flat globals)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# Sub-models (structure; loaded from config.yaml, overridable by env)         #
# --------------------------------------------------------------------------- #

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
    # Residual layer the SAE + probes hook. Single int in v1.
    layer: int = 17
    # Candidate residual layers for multi-layer SAE work (today: SAE_LAYERS).
    sae_layers: list[int] = Field(default_factory=lambda: [9, 17, 22, 29])
    device: str = "cuda"  # "cuda" | "mps" | "cpu" | "auto"; resolved via resolve_device()
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    # Tokens masked out of the SAE feature cloud (BOS + Gemma turn markers).
    mask_tokens: list[str] = Field(
        default_factory=lambda: ["<bos>", "<start_of_turn>", "<end_of_turn>"]
    )
    # Skip the first N response tokens during attribution (Gemma's formulaic preamble).
    preamble_skip: int = 12
    # Response length cap for the chat/analyze path.
    max_new_tokens: int = 512


class SAEConfig(BaseModel):
    """The SAE under inspection + its Neuronpedia label source (the swap seam)."""

    release: str = "gemma-scope-2-4b-it-res"
    # Per-layer SAE id pattern. Rendered with `layer` to produce the SAELens sae_id.
    # Today: f"layer_{LAYER}_width_16k_l0_medium". `{layer}` is substituted at load.
    sae_id_pattern: str = "layer_{layer}_width_16k_l0_medium"
    # Hidden / latent dims. Used as FALLBACKS — derived from the loaded SAE at runtime (WS2).
    d_in: int = 2560
    d_sae: int = 16384
    # Neuronpedia label source. np_model MUST match the loaded SAE variant
    # (it-res → "gemma-3-4b-it"). np_source width must match sae_id_pattern's width.
    np_model: str = "gemma-3-4b-it"
    np_source_pattern: str = "{layer}-gemmascope-2-res-16k"
    np_feature_url: str = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
    # SAE wiring sanity check (runtime._run_recon_check), surfaced on /health.
    recon_min_cosine: float = 0.85
    # Neutral default probe (was the ibuprofen/pregnancy medical string).
    recon_probe: str = "What is the boiling point of water at sea level?"


class FeatureCloudConfig(BaseModel):
    """SAE feature-cloud ranking knobs (Family A). Pure-CPU + torch read these."""

    topk: int = 15                  # top features per token
    topk_event: int = 30            # union cap across tokens in the final event
    topk_candidates: int = 50       # raw candidate pool before final cut
    drop_unlabeled: bool = True     # hide bare "feature N" from the cloud
    density_max: float = 0.01       # drop features firing on > this corpus fraction
    syntactic_penalty: float = 0.12 # down-rank syntactic-labeled features (activation path)
    structural_penalty: float = 0.15  # down-rank structural-tagged features (attribution path)
    rank_method: Literal["attribution", "activation"] = "attribution"
    contrast_baseline: bool = True  # subtract per-feature attribution on a neutral prompt
    contrast_prompt: str = "Can you explain how rainbows form?"
    contrast_max_new: int = 64
    autointerp: bool = True         # Claude-label features Neuronpedia hasn't explained
    autointerp_model: str = "claude-haiku-4-5"


class ProbeBuilderConfig(BaseModel):
    """Interpretability-agent / probe-builder knobs (the probe builder writes here)."""

    agent_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    agent_max_questions: int = 12   # cap contrastive questions for judge_filter
    judge_batch_size: int = 8
    auroc_threshold: float = 0.75   # deploy gate (was TRACK_AUROC_TAU)


class ProbeConfig(BaseModel):
    """Probe set + builder. ARTIFACT_DIR + enabled/disabled trackers + builder sub-config."""

    # Live probe set — only these artifacts load at GPU startup (persona.load_artifacts).
    enabled: list[str] = Field(
        default_factory=lambda: ["harmful", "harmful_prompt", "over_confidence"]
    )
    # On-disk artifacts that are NOT loaded (available for re-training).
    disabled: list[str] = Field(
        default_factory=lambda: ["uncertainty", "hallucination", "risk_awareness"]
    )
    artifacts_dir: Path = Path(__file__).parent / "science" / "artifacts"
    default_threshold: float = 0.5
    builder: ProbeBuilderConfig = Field(default_factory=ProbeBuilderConfig)


class SentryConfig(BaseModel):
    """Sentry read/emit surfaces. The DSN + auth token are SECRETS (see Secrets), not here."""

    environment: str = "production"          # was SENTRY_ENVIRONMENT ("hackathon")
    release: str | None = None               # None → Sentry auto-detects git SHA
    # PHI gate: when False, raw user_msg/response are NOT attached to Sentry events.
    send_io: bool = False                    # was SENTRY_SEND_IO ("0")
    org_slug: str = ""                       # was SENTRY_ORG_SLUG
    project_slug: str = ""                   # was SENTRY_PROJECT_SLUG
    api_base: str = "https://sentry.io"      # was SENTRY_API_BASE
    org_url: str = "https://sentry.io"       # deep-link host; was SENTRY_ORG_URL


class ObsConfig(BaseModel):
    """Observability. Phoenix/Arize is REMOVED in WS1 — only Sentry remains."""

    sentry: SentryConfig = Field(default_factory=SentryConfig)


class PodConfig(BaseModel):
    """GPU pod connection (orchestration → remote torch service). token is a SECRET."""

    url: str = "http://localhost:8001"   # rstrip("/") applied in loader
    timeout: float = 120.0
    poll_interval: float = 5.0


class RuntimeConfig(BaseModel):
    """Process-level runtime + branding."""

    mode: Literal["posthoc", "live"] = "posthoc"   # was GLASSBOX_MODE
    product_name: str = "GlassBox"                  # rebrand seam (WS0)


# --------------------------------------------------------------------------- #
# Secrets (env / .env ONLY — never config.yaml)                               #
# --------------------------------------------------------------------------- #

class Secrets(BaseSettings):
    """Secret values. Sourced ONLY from environment / .env, never YAML.

    Kept as a dedicated settings object so secrets are structurally incapable of being
    read from config.yaml. Injected into AppConfig sub-models by the loader.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    sentry_auth_token: str = Field(default="", alias="SENTRY_AUTH_TOKEN")
    pod_token: str = Field(default="", alias="POD_TOKEN")
    hf_token: str = Field(default="", alias="HF_TOKEN")


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #

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

    # --- Derived helpers (replace config.sae_id_for_layer / config.resolve_device) ---

    def sae_id(self, layer: int | None = None) -> str:
        """SAELens sae_id for `layer` (defaults to model.layer)."""
        return self.sae.sae_id_pattern.format(layer=layer if layer is not None else self.model.layer)

    def np_source(self, layer: int | None = None) -> str:
        """Neuronpedia source slug for `layer`."""
        return self.sae.np_source_pattern.format(layer=layer if layer is not None else self.model.layer)

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

> **Note on `FeatureCloudConfig`:** the design doc §2 sketch folds the cloud-ranking knobs into
> `model`/`sae`. They are a distinct cohesive group (15 fields, read by `analyze`,
> `feature_provider`, `science/sae`, `gpu_service`), so this contract gives them their own
> `feature_cloud` sub-model hung off `AppConfig`. Downstream planners: treat `feature_cloud` as
> a first-class sub-model. If a planner prefers folding it into `sae`, that is the ONLY allowed
> deviation and must be flagged — every other path is fixed.

---

## 2. Loader

```python
_CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build the AppConfig once.

    Order of precedence (lowest → highest):
      1. Gemma + Gemma-Scope defaults baked into the pydantic models.
      2. Structure from config.yaml (path arg, else repo-root config.yaml). Absent → defaults
         only, so the app still boots.
      3. Per-field env override (12-factor): GLASSBOX__MODEL__LAYER=20 etc. (nested delimiter
         "__"), applied on top of YAML.
      4. Secrets from env / .env via the Secrets object — env-ONLY, never YAML.

    Returns a fully-populated AppConfig. Never reads secrets from YAML.
    """
    cfg_path = Path(path) if path is not None else _CFG_PATH

    # (1)+(2) structure
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        raw.pop("anthropic_api_key", None)   # defense-in-depth: never honor secrets from YAML
        raw.pop("sentry_dsn", None)
        raw.pop("sentry_auth_token", None)
        raw.pop("pod_token", None)
        raw.pop("hf_token", None)
    cfg = AppConfig.model_validate(raw)

    # (3) env override for structural fields — implemented via a thin BaseSettings mirror or
    #     explicit os.environ scan with the GLASSBOX__SECTION__FIELD convention. Left to the
    #     loader impl; the CONTRACT is: env beats YAML for any structural field.

    # (4) secrets (env / .env only)
    secrets = Secrets()
    cfg.anthropic_api_key = secrets.anthropic_api_key
    cfg.sentry_dsn = secrets.sentry_dsn
    cfg.sentry_auth_token = secrets.sentry_auth_token
    cfg.pod_token = secrets.pod_token
    cfg.hf_token = secrets.hf_token

    # Normalize pod url
    cfg.pod.url = cfg.pod.url.rstrip("/")
    return cfg
```

**Behavioral guarantees (planners may rely on these):**
- No `config.yaml` → returns Gemma defaults with secrets from env. App boots.
- A secret key present in `config.yaml` is ignored (popped) — secrets only ever come from env.
- Env overrides YAML for structural fields.
- `cfg.pod.url` is always trailing-slash-normalized.

---

## 3. Threading pattern

### 3.1 Build once, store on `app.state`

Both FastAPI processes build the config at startup and store it on `app.state.config`.

**`backend/app.py` (CPU orchestration):**
```python
from .config import load_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = load_config()
    runtime.start_loading(app.state.config)        # was start_loading()
    init_sponsors(app.state.config.observability)   # was init_sponsors()
    yield

app = FastAPI(title="GlassBox", lifespan=lifespan)

# Endpoints read it off the request:
@app.get("/api/observability")
async def observability_endpoint(request: Request):
    cfg = request.app.state.config
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload(cfg)
    snap["sentry"] = {
        "emit_configured": bool(cfg.sentry_dsn),
        "configured": bool(cfg.sentry_auth_token),
        ...
    }
    return snap
```

**`backend/gpu_service.py` (GPU pod):**
```python
from .config import load_config

@app.on_event("startup")          # or a lifespan ctx
def _startup():
    app.state.config = load_config()

def _require_auth(request: Request) -> None:
    cfg = request.app.state.config
    token = cfg.pod_token
    ...   # WS3 hardens this to fail closed when token unset on a non-loopback bind
```

### 3.2 Sub-config injection signatures (the exact contract)

Modules receive the **narrowest** sub-config they need. Pure-CPU modules must still not import
torch. The injection signatures below are authoritative.

**Torch modules (GPU pod side):**

| Module / entry point | Current signature | New signature |
|---|---|---|
| `engine.load_engine` | `load_engine(device=None)` | `load_engine(model: ModelConfig, device=None)` |
| `engine.probe_activation` | `probe_activation(text, pos=None)` | `probe_activation(text, model: ModelConfig, pos=None)` |
| `science/sae.load_sae` | `load_sae(layer=config.LAYER, device=None)` | `load_sae(sae: SAEConfig, layer: int, device=None)` |
| `science/sae.reconstruction_error` | `reconstruction_error(act)` | `reconstruction_error(act, sae: SAEConfig)` |
| `science/persona.load_artifacts` | `load_artifacts(artifact_dir=ARTIFACT_DIR, exclude=(), include=None)` | `load_artifacts(probes: ProbeConfig, include=None)` (uses `probes.artifacts_dir`, `probes.disabled`) |
| `science/feature_provider` | reads `config.TOPK*`, `RANK_METHOD`, `PREAMBLE_SKIP`, `NP_*` | functions take `(sae: SAEConfig, feature_cloud: FeatureCloudConfig, model: ModelConfig)` as needed |
| `analyze.analyze_turn` | reads `config.*` cloud knobs + `MODEL_ID`/`LAYER`/`MAX_NEW_TOKENS` | `analyze_turn(..., cfg: AppConfig)` (needs model + sae + feature_cloud together) |
| `labels` (auto-interp) | reads `NP_*`, `AUTOINTERP*`, `ANTHROPIC_API_KEY` | `(sae: SAEConfig, feature_cloud: FeatureCloudConfig, anthropic_api_key: str)` |

**Agent / probe-builder modules (run on the pod, need Claude key):**

| Module / entry point | Current signature | New signature |
|---|---|---|
| `agent/interp_agent.run_interp_agent` | `run_interp_agent(tracker_id, *, client=None, generate_fn=None)` | `run_interp_agent(tracker_id, builder: ProbeBuilderConfig, anthropic_api_key: str, *, client=None, generate_fn=None)` — required/positional surface is fixed. WS3 MAY add the two **keyword-only, optional** params `domain: str = prompts.DEFAULT_DOMAIN` and `model: ModelConfig | None = None` after the `*` (authorized addition — see §6; the pod injects `domain` for the configurable-domain seam and `model` so `_persist_artifact` can read `model.layer`). |
| `agent/prompts` | reads `AGENT_MAX_QUESTIONS` | functions take `builder: ProbeBuilderConfig` |
| `agent/tools` | reads `AGENT_MAX_QUESTIONS`, `TRACK_AUROC_TAU`, `LAYER` | take `builder: ProbeBuilderConfig` + `model: ModelConfig` as needed |
| `science/concept_synth.judge_filter` | `judge_filter(spec, rows, *, client=None, judge_model=None)` | `judge_filter(spec, rows, builder: ProbeBuilderConfig, anthropic_api_key: str, *, client=None)` (keep `client=` injection point) |

**Pure-CPU modules (orchestration side — must NOT import torch):**

| Module / entry point | Current signature | New signature |
|---|---|---|
| `runtime.start_loading` | `start_loading()` | `start_loading(cfg: AppConfig)` (needs `pod.url`, `pod.poll_interval`) |
| `runtime.health_payload` | `health_payload()` | `health_payload(cfg: AppConfig)` (reads `model.model_id`, `model.layer`, `sae.d_sae`, `pod.url`) |
| `runtime.refresh_pod_health` | `refresh_pod_health()` | `refresh_pod_health(cfg: AppConfig)` |
| `fanout.fanout` / `capture_cognition_alarm` | read `SENTRY_SEND_IO`, `DISABLED_TRACKERS` | take `obs: ObsConfig` + `probes: ProbeConfig` (or `cfg`) |
| `fanout.init_sponsors` | `init_sponsors()` | `init_sponsors(obs: ObsConfig, sentry_dsn: str)` — Phoenix branch removed (WS1) |
| `sentry_api` | reads `SENTRY_*` | functions take `sentry: SentryConfig` + `sentry_auth_token: str` |
| `pod_client` | reads `POD_URL`, `POD_TOKEN`, `POD_TIMEOUT`, `MAX_NEW_TOKENS` | functions take `pod: PodConfig` + `pod_token: str`, and `model.max_new_tokens` for generate calls |
| `fallback` | reads `D_SAE`, `NP_SOURCE` | take `sae: SAEConfig` (+ `cfg.np_source()`); genericize canned medical labels to neutral (WS0/§7) |

> Because `runtime`, `fanout`, `pod_client`, `sentry_api` are imported by `app.py` and must stay
> torch-free, they receive plain sub-models / scalars — never anything that triggers a torch
> import. `resolve_device` is the one helper that imports torch lazily inside its body, so a CPU
> module can hold an `AppConfig` without importing torch as long as it never calls
> `resolve_device`.

---

## 4. Migration map — every current `config.X` → new path

Verified against a full grep of `backend/` (every `config.` access). All ~50 globals accounted
for. "Consumers" lists the files that read each today.

### model
| Current global | New path | Consumers |
|---|---|---|
| `config.MODEL_ID` | `cfg.model.model_id` | engine, gpu_service, runtime, analyze |
| `config.LAYER` | `cfg.model.layer` | engine, sae, runtime, analyze, gpu_service, agent/tools, science/sae |
| `config.SAE_LAYERS` | `cfg.model.sae_layers` | (config-internal; multi-layer work) |
| `config.SYSTEM_PROMPT` | `cfg.model.system_prompt` | engine | **default changes → neutral** |
| `config.MASK_TOKENS` | `cfg.model.mask_tokens` | gpu_service |
| `config.PREAMBLE_SKIP` | `cfg.model.preamble_skip` | gpu_service, feature_provider |
| `config.MAX_NEW_TOKENS` | `cfg.model.max_new_tokens` | gpu_service, analyze, pod_client |
| `config.DEVICE` | `cfg.model.device` | (via resolve_device) |
| `config.resolve_device()` | `cfg.resolve_device()` | engine, sae, feature_provider |

### sae
| Current global | New path | Consumers |
|---|---|---|
| `config.SAE_RELEASE` | `cfg.sae.release` | science/sae |
| `config.SAE_ID` / `config.sae_id_for_layer()` | `cfg.sae_id(layer)` (pattern `cfg.sae.sae_id_pattern`) | science/sae |
| `config.D_IN` | `cfg.sae.d_in` (fallback; derived at runtime WS2) | (load-time) |
| `config.D_SAE` | `cfg.sae.d_sae` (fallback; derived at runtime WS2) | gpu_service, runtime, fallback |
| `config.NP_MODEL` | `cfg.sae.np_model` | labels, feature_provider |
| `config.NP_SOURCE` / per-layer | `cfg.np_source(layer)` (pattern `cfg.sae.np_source_pattern`) | app, labels, fallback, science/sae, feature_provider |
| `config.NP_FEATURE_URL` | `cfg.sae.np_feature_url` | labels |
| `config.RECON_MIN_COSINE` | `cfg.sae.recon_min_cosine` | gpu_service |
| `config.RECON_PROBE` | `cfg.sae.recon_probe` | gpu_service | **default changes → neutral** |

### feature_cloud
| Current global | New path | Consumers |
|---|---|---|
| `config.TOPK` | `cfg.feature_cloud.topk` | science/sae, feature_provider |
| `config.TOPK_EVENT` | `cfg.feature_cloud.topk_event` | analyze, feature_provider |
| `config.TOPK_CANDIDATES` | `cfg.feature_cloud.topk_candidates` | gpu_service, science/sae |
| `config.DROP_UNLABELED` | `cfg.feature_cloud.drop_unlabeled` | analyze |
| `config.DENSITY_MAX` | `cfg.feature_cloud.density_max` | analyze |
| `config.SYNTACTIC_PENALTY` | `cfg.feature_cloud.syntactic_penalty` | analyze |
| `config.STRUCTURAL_PENALTY` | `cfg.feature_cloud.structural_penalty` | analyze |
| `config.RANK_METHOD` | `cfg.feature_cloud.rank_method` | engine, gpu_service, feature_provider |
| `config.CONTRAST_BASELINE` | `cfg.feature_cloud.contrast_baseline` | gpu_service |
| `config.CONTRAST_PROMPT` | `cfg.feature_cloud.contrast_prompt` | gpu_service |
| `config.CONTRAST_MAX_NEW` | `cfg.feature_cloud.contrast_max_new` | gpu_service |
| `config.AUTOINTERP` | `cfg.feature_cloud.autointerp` | labels |
| `config.AUTOINTERP_MODEL` | `cfg.feature_cloud.autointerp_model` | labels |

### probes
| Current global | New path | Consumers |
|---|---|---|
| `config.ENABLED_TRACKERS` | `cfg.probes.enabled` (list) | science/persona |
| `config.BUILTIN_TRACKERS` | `sorted(cfg.probes.enabled)` (alias) | scripts/docs |
| `config.DISABLED_TRACKERS` | `cfg.probes.disabled` (list) | gpu_service, fanout, science/persona |
| `config.DEFAULT_THRESHOLD` | `cfg.probes.default_threshold` | (probe scoring) |
| `config.TRACK_AUROC_TAU` | `cfg.probes.builder.auroc_threshold` | agent/tools |
| `config.AGENT_MODEL` | `cfg.probes.builder.agent_model` | agent/interp_agent |
| `config.JUDGE_MODEL` | `cfg.probes.builder.judge_model` | science/concept_synth |
| `config.AGENT_MAX_QUESTIONS` | `cfg.probes.builder.agent_max_questions` | agent/prompts, agent/tools |
| `config.JUDGE_BATCH_SIZE` | `cfg.probes.builder.judge_batch_size` | science/concept_synth |
| `persona.ARTIFACT_DIR` | `cfg.probes.artifacts_dir` | science/persona |

### observability (Sentry only; Phoenix dropped in WS1)
| Current global | New path | Consumers |
|---|---|---|
| `config.SENTRY_ENVIRONMENT` | `cfg.observability.sentry.environment` | fanout |
| `config.SENTRY_RELEASE` | `cfg.observability.sentry.release` | fanout |
| `config.SENTRY_SEND_IO` | `cfg.observability.sentry.send_io` | fanout |
| `config.SENTRY_ORG_SLUG` | `cfg.observability.sentry.org_slug` | app, sentry_api |
| `config.SENTRY_PROJECT_SLUG` | `cfg.observability.sentry.project_slug` | sentry_api |
| `config.SENTRY_API_BASE` | `cfg.observability.sentry.api_base` | sentry_api |
| `config.SENTRY_ORG_URL` | `cfg.observability.sentry.org_url` | sentry_api |
| `config.PHOENIX_ENDPOINT` | **REMOVED** (WS1) | fanout |
| `config.PHOENIX_UI_URL` | **REMOVED** (WS1) | app |
| `config.EVAL_LLM_PROVIDER` | **REMOVED** (WS1 — coherence_eval deleted) | coherence_eval |
| `config.EVAL_LLM_MODEL` | **REMOVED** (WS1 — coherence_eval deleted) | coherence_eval |

### pod
| Current global | New path | Consumers |
|---|---|---|
| `config.POD_URL` | `cfg.pod.url` | runtime, pod_client |
| `config.POD_TIMEOUT` | `cfg.pod.timeout` | pod_client |
| `config.POD_POLL_INTERVAL` | `cfg.pod.poll_interval` | runtime |

### runtime
| Current global | New path | Consumers |
|---|---|---|
| `config.MODE` (`GLASSBOX_MODE`) | `cfg.runtime.mode` | (mode switch) |
| (new) product name | `cfg.runtime.product_name` | app title, README, UI copy (WS0 rebrand) |

### secrets (env-only — OUTSIDE config.yaml)
| Current global | New path | Consumers |
|---|---|---|
| `config.ANTHROPIC_API_KEY` | `cfg.anthropic_api_key` | gpu_service, labels, science/concept_synth, agent/interp_agent |
| `config.SENTRY_DSN` | `cfg.sentry_dsn` | app, fanout |
| `config.SENTRY_AUTH_TOKEN` | `cfg.sentry_auth_token` | app, sentry_api |
| `config.POD_TOKEN` | `cfg.pod_token` | gpu_service, pod_client |
| (new) `HF_TOKEN` | `cfg.hf_token` | model/SAE download (WS2/WS3) |

---

## 5. `config.example.yaml` (committed template, NO secrets)

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
# To use, copy model.system_prompt into the active model block:
#   system_prompt: |
#     You are a clinical decision-support assistant for licensed healthcare professionals...
#     (full medical prompt — the former config.py default — ships here, clearly labeled)
```

---

## 6. Notes for downstream planners

- **WS0 owns this refactor.** It must land the models, loader, `app.state` wiring, the signature
  changes in §3.2, `config.example.yaml`, and gitignore `config.yaml`. The 30-file test suite is
  the safety net; add tests for: defaults-when-no-YAML, env-overrides-YAML, secrets-never-from-YAML.
- **WS1** deletes `PHOENIX_*`, `EVAL_LLM_*` (the four rows marked REMOVED), removes the Phoenix
  branch of `init_sponsors`, and renames `cognition_event` → `introspection_event` (orthogonal to
  this contract — the Sentry sub-config field names already use neutral terms).
- **WS2** consumes the `sae` seam: derive `d_in`/`d_sae` from the loaded SAE, falling back to
  `cfg.sae.d_in`/`cfg.sae.d_sae`.
- **WS3** hardens `_require_auth` to fail closed using `cfg.pod_token`, and removes committed
  default tokens. `HF_TOKEN` (new secret) feeds the model/SAE download. **WS3 also adds two
  keyword-only optional params to `run_interp_agent` — `domain` (the configurable target-assistant
  domain seam) and `model: ModelConfig | None` (so `_persist_artifact` reads `model.layer` from the
  per-job `ctx`).** This is an authorized, backward-compatible addition: the contract's fixed
  required/positional surface (`tracker_id, builder, anthropic_api_key, *, client, generate_fn`) is
  unchanged; only optional keyword-only params are appended after the `*`. See the amended §3.2 row.
- **WS0 rebrand** also genericizes the canned medical strings in `fallback.py` (lines 18, 44),
  `labels.py` (line 117 auto-interp prompt), and `agent/prompts.py` (lines 10, 12, 39
  "medical-chat LLM") to neutral defaults / a configurable domain string derived from
  `model.system_prompt`.
- **One deviation flagged:** the design-doc sketch lists five sub-models (`model`, `sae`,
  `probes`, `observability`, `pod`, `runtime`); this contract adds `feature_cloud` as a seventh
  cohesive sub-model rather than scattering 13 cloud-ranking knobs into `model`/`sae`. Planners
  must use `cfg.feature_cloud.*` for those fields.
```
