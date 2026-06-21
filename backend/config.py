"""Constants + env loading. The layer/SAE choices are LOCKED (see README model decision)."""

from __future__ import annotations

import os

MODEL_ID = os.getenv("MODEL_ID", "unsloth/gemma-3-4b-it")
LAYER = int(os.getenv("LAYER", "17"))
SAE_LAYERS = [9, 17, 22, 29]
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = os.getenv("SAE_ID", f"layer_{LAYER}_width_16k_l0_medium")
D_IN = 2560
D_SAE = 16384

NP_MODEL = "gemma-3-4b"
NP_SOURCE = f"{LAYER}-gemmascope-2-res-16k"
NP_FEATURE_URL = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"

TOPK = 15
TOPK_EVENT = 30

BUILTIN_TRACKERS = ["uncertainty", "harmful", "hallucination"]
DEFAULT_THRESHOLD = 0.5

MODE = os.getenv("GLASSBOX_MODE", "posthoc")
DEVICE = os.getenv("DEVICE", "cuda")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Hackathon default so Sentry works out of the box. Override via the SENTRY_DSN env var
# (or a .env) — and move it OUT of source before any public/shared deploy: a committed
# DSN lets anyone who finds it write events into this project. It is NOT read access.
SENTRY_DSN = os.getenv(
    "SENTRY_DSN",
    "https://296000e0a9e12dd30d23a38c373df8eb@o4511600334536704.ingest.us.sentry.io/4511600481206272",
)
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "hackathon")
SENTRY_RELEASE = os.getenv("SENTRY_RELEASE") or None  # None → Sentry auto-detects git SHA
# PHI gate: when false, the raw user_msg/response are NOT attached to Sentry events.
SENTRY_SEND_IO = os.getenv("SENTRY_SEND_IO", "1").lower() not in ("0", "false", "no", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

MASK_TOKENS = ["<bos>", "<start_of_turn>", "<end_of_turn>"]


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
