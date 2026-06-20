"""Constants + env loading. The layer/SAE choices are LOCKED (see README model decision)."""
from __future__ import annotations

import os

# --- Model + SAE (LOCKED) ---
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2-2b-it")  # or unsloth/gemma-2-2b-it (ungated)
LAYER = 12                                                # residual-stream layer for BOTH families
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = f"layer_{LAYER}/width_16k/canonical"             # d_in=2304 -> d_sae=16384, L0~100
D_IN = 2304
D_SAE = 16384

# --- Neuronpedia label lookup (keyless GET) ---
NP_MODEL = "gemma-2-2b"
NP_SOURCE = f"{LAYER}-gemmascope-res-16k"
NP_FEATURE_URL = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"

# --- Family A (SAE cloud) ---
TOPK = 15            # top features per token
TOPK_EVENT = 30      # union cap across tokens in the final event

# --- Family B (probes) ---
BUILTIN_TRACKERS = ["uncertainty", "harmful", "hallucination"]
# Thresholds are calibrated offline (validation/) and loaded from science/vectors/thresholds.json.
DEFAULT_THRESHOLD = 0.5

# --- Runtime ---
MODE = os.getenv("GLASSBOX_MODE", "posthoc")  # posthoc | live
DEVICE = os.getenv("DEVICE", "cuda")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

# Special tokens to mask in the SAE cloud (high-norm noise on chat-template control tokens).
MASK_TOKENS = ["<bos>", "<start_of_turn>", "<end_of_turn>"]
