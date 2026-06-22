"""backend/config.py — unified AppConfig (replaces the flat globals).

WS0 Task 1: sub-models + Secrets added.
Flat globals retained below for backward compatibility during migration; Tasks 5–14 remove them.
"""

from __future__ import annotations

import os
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


# --------------------------------------------------------------------------- #
# LEGACY flat globals — retained for backward compatibility during WS0 migration.
# Tasks 5–14 replace every consumer with the new sub-model paths; globals are
# deleted once all callers have been migrated.
# --------------------------------------------------------------------------- #

# Load a gitignored repo-root .env (if present) so local secrets — Sentry token/DSN, pod creds —
# stay out of source. Real environment variables still take precedence (load_dotenv won't override).
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv optional; absence just means no .env convenience
    pass

MODEL_ID = os.getenv("MODEL_ID", "unsloth/gemma-3-4b-it")
LAYER = int(os.getenv("LAYER", "17"))
SAE_LAYERS = [9, 17, 22, 29]
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = os.getenv("SAE_ID", f"layer_{LAYER}_width_16k_l0_medium")
D_IN = 2560
D_SAE = 16384

NP_MODEL = os.getenv("NP_MODEL", "gemma-3-4b-it")
NP_SOURCE = os.getenv("NP_SOURCE", f"{LAYER}-gemmascope-2-res-16k")
NP_FEATURE_URL = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"

# --- Family A (SAE cloud) ---
TOPK = 15
TOPK_EVENT = 30
TOPK_CANDIDATES = int(os.getenv("TOPK_CANDIDATES", "50"))
DROP_UNLABELED = os.getenv("DROP_UNLABELED", "1") == "1"
DENSITY_MAX = float(os.getenv("DENSITY_MAX", "0.01"))
SYNTACTIC_PENALTY = float(os.getenv("SYNTACTIC_PENALTY", "0.12"))
RANK_METHOD = os.getenv("RANK_METHOD", "attribution")
CONTRAST_BASELINE = os.getenv("CONTRAST_BASELINE", "1") == "1"
CONTRAST_PROMPT = os.getenv("CONTRAST_PROMPT", "Can you explain how rainbows form?")
CONTRAST_MAX_NEW = int(os.getenv("CONTRAST_MAX_NEW", "64"))
PREAMBLE_SKIP = int(os.getenv("PREAMBLE_SKIP", "12"))
AUTOINTERP = os.getenv("AUTOINTERP", "1") == "1"
AUTOINTERP_MODEL = os.getenv("AUTOINTERP_MODEL", "claude-haiku-4-5")
STRUCTURAL_PENALTY = float(os.getenv("STRUCTURAL_PENALTY", "0.15"))

ENABLED_TRACKERS = frozenset({"harmful", "over_confidence", "harmful_prompt"})
BUILTIN_TRACKERS = sorted(ENABLED_TRACKERS)

DEFAULT_THRESHOLD = 0.5

DISABLED_TRACKERS = frozenset({"uncertainty", "hallucination", "risk_awareness"})

POD_URL = os.getenv("POD_URL", "http://localhost:8001").rstrip("/")
POD_TOKEN = os.getenv("POD_TOKEN", "")
POD_TIMEOUT = float(os.getenv("POD_TIMEOUT", "120"))
POD_POLL_INTERVAL = float(os.getenv("POD_POLL_INTERVAL", "5"))

TRACK_AUROC_TAU = float(os.getenv("TRACK_AUROC_TAU", "0.75"))
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-8")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-4-8")
AGENT_MAX_QUESTIONS = int(os.getenv("AGENT_MAX_QUESTIONS", "12"))
JUDGE_BATCH_SIZE = int(os.getenv("JUDGE_BATCH_SIZE", "8"))

MODE = os.getenv("GLASSBOX_MODE", "posthoc")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

_DEFAULT_MEDICAL_SYSTEM_PROMPT = """You are a clinical decision-support assistant for licensed healthcare professionals. You provide accurate, evidence-based medical information grounded in current clinical guidelines and the peer-reviewed literature.

Operating principles:
- Accuracy first. Base answers on established evidence and current guidelines (e.g. ACOG, FDA, CDC, NICE, WHO, and specialty-society consensus). Never invent studies, statistics, doses, or citations. If the evidence is uncertain or mixed, say so explicitly rather than guessing.
- Calibrated confidence. Distinguish well-established facts from areas of genuine clinical uncertainty or ongoing debate. Do not overstate certainty; state the strength of the evidence where it changes the decision.
- Safety first. Proactively surface contraindications, clinically significant drug-drug interactions, boxed warnings, and risks in special populations - pregnancy and lactation, pediatric and geriatric patients, and renal or hepatic impairment. When an intervention is contraindicated, lead with that.
- Dosing. Give typical adult dosing only with the relevant caveats (indication, route, adjustments, maximum). Flag when individualized dosing or therapeutic monitoring is required.
- Scope and escalation. You support, you do not replace, the clinician's judgment and a complete patient assessment. Recommend confirming against primary sources and, for an individual patient, consulting the treating physician or pharmacist. Direct anyone describing an emergency (e.g. chest pain, anaphylaxis, stroke symptoms, overdose) to emergency services immediately.

Be clear and well-structured. Define abbreviations on first use. When a question is ambiguous, state the key assumptions you are making. Prioritize the information that changes clinical decisions."""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _DEFAULT_MEDICAL_SYSTEM_PROMPT)
DEVICE = os.getenv("DEVICE", "cuda")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "hackathon")
SENTRY_RELEASE = os.getenv("SENTRY_RELEASE") or None
SENTRY_SEND_IO = os.getenv("SENTRY_SEND_IO", "0").lower() not in ("0", "false", "no", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_UI_URL = os.getenv("PHOENIX_UI_URL", PHOENIX_ENDPOINT)
SENTRY_ORG_SLUG = os.getenv("SENTRY_ORG_SLUG", "")
SENTRY_PROJECT_SLUG = os.getenv("SENTRY_PROJECT_SLUG", "")
SENTRY_AUTH_TOKEN = os.getenv("SENTRY_AUTH_TOKEN", "")
SENTRY_API_BASE = os.getenv("SENTRY_API_BASE", "https://sentry.io").rstrip("/")
SENTRY_ORG_URL = os.getenv("SENTRY_ORG_URL", "https://sentry.io")
EVAL_LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "anthropic")
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "claude-haiku-4-5-20251001")

MASK_TOKENS = ["<bos>", "<start_of_turn>", "<end_of_turn>"]

RECON_MIN_COSINE = float(os.getenv("RECON_MIN_COSINE", "0.85"))
RECON_PROBE = os.getenv("RECON_PROBE", "Is ibuprofen safe during the third trimester of pregnancy?")


def sae_id_for_layer(layer: int = LAYER) -> str:
    """Gemma Scope width-16k SAE id for a residual `layer` (defaults to LAYER=17).
    An explicit SAE_ID env var, when set, overrides the per-layer id."""
    return os.getenv("SAE_ID") or f"layer_{layer}_width_16k_l0_medium"


def resolve_device(pref: str | None = None) -> str:
    """Resolve the runtime device."""
    import torch

    p = (pref or DEVICE or "auto").lower()
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
