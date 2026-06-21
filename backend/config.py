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

# --- Family A (SAE cloud) ---
TOPK = 15  # top features per token
TOPK_EVENT = 30  # union cap across tokens in the final event

# --- Family B (probes) ---
BUILTIN_TRACKERS = ["uncertainty", "harmful", "hallucination"]
# Thresholds are calibrated offline (validation/) and loaded from science/vectors/thresholds.json.
DEFAULT_THRESHOLD = 0.5

# --- Interpretability Agent ---
TRACK_AUROC_TAU = float(os.getenv("TRACK_AUROC_TAU", "0.75"))  # deploy gate
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-8")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-4-8")

# --- Runtime ---
MODE = os.getenv("GLASSBOX_MODE", "posthoc")  # posthoc | live
DEVICE = os.getenv("DEVICE", "cuda")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

MASK_TOKENS = ["<bos>", "<start_of_turn>", "<end_of_turn>"]


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
