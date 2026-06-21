"""Constants + env loading. The layer/SAE choices are LOCKED (see README model decision)."""

from __future__ import annotations

import os

# Load a gitignored repo-root .env (if present) so local secrets — Sentry token/DSN, pod creds —
# stay out of source. Real environment variables still take precedence (load_dotenv won't override).
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv optional; absence just means no .env convenience
    pass

MODEL_ID = os.getenv("MODEL_ID", "unsloth/gemma-3-4b-it")
LAYER = int(os.getenv("LAYER", "17"))
SAE_LAYERS = [9, 17, 22, 29]
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = os.getenv("SAE_ID", f"layer_{LAYER}_width_16k_l0_medium")
D_IN = 2560
D_SAE = 16384

# Neuronpedia model id. MUST match the *loaded* SAE's variant: the IT SAE
# (gemma-scope-2-4b-it-res) maps to 'gemma-3-4b-it'. The non-it slug points at the PT
# base-model SAE — a DIFFERENT latent space, so its labels/maxActApprox/frac_nonzero would be
# attached to the wrong feature. Verified against the SAELens registry's neuronpedia field.
NP_MODEL = os.getenv("NP_MODEL", "gemma-3-4b-it")
# Neuronpedia source slug. Width must match SAE_ID's width (16k/65k/262k). Override via env
# when changing SAE width, e.g. NP_SOURCE=22-gemmascope-2-res-65k for layer_22_width_65k.
NP_SOURCE = os.getenv("NP_SOURCE", f"{LAYER}-gemmascope-2-res-16k")
NP_FEATURE_URL = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"

# --- Family A (SAE cloud) ---
TOPK = 15  # top features per token
TOPK_EVENT = 30  # union cap across tokens in the final event
# Raw candidate pool re-ranked by NP-relative activation (act / maxActApprox) before the
# final TOPK_EVENT cut — surfaces specific (e.g. medical) features over high-norm grammatical
# ones. See backend/analyze.py:_rank_features and PLAN.md §7.
TOPK_CANDIDATES = int(os.getenv("TOPK_CANDIDATES", "50"))
# Hide features Neuronpedia hasn't labelled (bare "feature N") from the cloud — keeps it
# interpretable. Set DROP_UNLABELED=0 to show them (exploratory).
DROP_UNLABELED = os.getenv("DROP_UNLABELED", "1") == "1"
# Drop features that fire on more than this fraction of the corpus (Neuronpedia frac_nonzero).
# Generic/grammatical features fire densely (~1%+); specific medical features are rare (~0.1%).
# Lower = stricter / more on-topic. Set 1.0 to disable.
DENSITY_MAX = float(os.getenv("DENSITY_MAX", "0.01"))
# Down-rank features whose LABEL is syntactic/surface ("X followed by Y", "contraction",
# "adverb"...) by this factor, so abstract/conceptual features outrank them. Activation stats
# can't tell syntactic from semantic — only the label can. Set 1.0 to disable.
# (Only applied on the activation-ranking fallback path; attribution makes it unnecessary.)
SYNTACTIC_PENALTY = float(os.getenv("SYNTACTIC_PENALTY", "0.12"))

# How candidates are selected and ranked:
#   "attribution" (default) — rank by causal effect on the response: act × (∇_resid L · decoder),
#       summed over response tokens. Surfaces features that SHAPE the answer and down-weights
#       high-frequency grammatical ones structurally. Needs one backward pass (engine).
#   "activation" — legacy: raw-activation top-K pool re-ranked by act/maxActApprox + density +
#       syntactic heuristics. Used as the automatic fallback if the backward pass fails/OOMs.
RANK_METHOD = os.getenv("RANK_METHOD", "attribution")

# Contrastive attribution: subtract per-feature attribution measured on a fixed neutral prompt
# (computed once, cached) so "always-on" discourse features (greetings, "Okay, let's...") that
# fire on every reply cancel out, leaving topic-specific (e.g. medical) features on top. The
# baseline prompt should be a generic informational question that triggers the same chatty
# framing but no domain content. Set CONTRAST_BASELINE=0 to disable.
CONTRAST_BASELINE = os.getenv("CONTRAST_BASELINE", "1") == "1"
CONTRAST_PROMPT = os.getenv("CONTRAST_PROMPT", "Can you explain how rainbows form?")
CONTRAST_MAX_NEW = int(os.getenv("CONTRAST_MAX_NEW", "64"))

# Attribution ignores the first PREAMBLE_SKIP response tokens. gemma-3-4b-it opens every answer
# with a formulaic, high-confidence preamble ("Okay, let's talk about...") whose discourse
# features (greetings, "Okay,") otherwise dominate the attribution sum because they fire on the
# very tokens the model is most certain about. Skipping the preamble lets the content (e.g.
# medical) tokens carry the attribution. Applied to both the turn and the contrast baseline. 0=off.
PREAMBLE_SKIP = int(os.getenv("PREAMBLE_SKIP", "12"))

# Auto-interp: when Neuronpedia has no explanation for a feature, label it ourselves by sending
# its top activating examples (already in the per-feature GET) to Claude. Closes the coverage gap
# that leaves ~half the attribution cloud showing "feature N". Cached to disk; degrades silently
# to "feature N". The model also tags each feature structural-vs-concept, which drives the re-rank.
AUTOINTERP = os.getenv("AUTOINTERP", "1") == "1"
AUTOINTERP_MODEL = os.getenv("AUTOINTERP_MODEL", "claude-haiku-4-5")  # cheap, one-time per feature
# Down-rank (don't drop) features whose label/auto-interp tag is structural — punctuation,
# formatting, whitespace, token fragments, discourse glue — so genuine medical features surface
# beneath them on the attribution path. NOT a density gate (density misorders: a sparse medical
# feature can be rarer than a dense structural one). 1.0 = disable.
STRUCTURAL_PENALTY = float(os.getenv("STRUCTURAL_PENALTY", "0.15"))

# Live probe set — ONLY these artifacts load at GPU service startup (persona.load_artifacts).
# Trained artifacts for deprecated probes stay on disk but are excluded via DISABLED_TRACKERS.
ENABLED_TRACKERS = frozenset({"harmful", "over_confidence", "harmful_prompt"})
BUILTIN_TRACKERS = sorted(ENABLED_TRACKERS)  # alias for scripts/docs

DEFAULT_THRESHOLD = 0.5

# On-disk artifacts that are NOT loaded. Keeps hallucination/uncertainty/risk_awareness
# available for re-training without surfacing them in /health or /turn scores.
DISABLED_TRACKERS = frozenset({"uncertainty", "hallucination", "risk_awareness"})

# --- GPU pod (orchestration → remote torch service) ---
# Defaults wire to the local SSH tunnel (scripts/tunnel_pod.sh → localhost:8001) so the backend
# reaches the pod no matter how it's launched — a bare `uvicorn backend.app:app` without the env
# vars set would otherwise silently fall back to synthetic/offline. Override via env for other
# setups; POD_TOKEN is the shared dev secret (move it to a .env before any public/shared deploy).
POD_URL = os.getenv("POD_URL", "http://localhost:8001").rstrip("/")
POD_TOKEN = os.getenv("POD_TOKEN", "glassbox-dev-secret")
POD_TIMEOUT = float(os.getenv("POD_TIMEOUT", "120"))
POD_POLL_INTERVAL = float(os.getenv("POD_POLL_INTERVAL", "5"))

# --- Interpretability Agent ---
TRACK_AUROC_TAU = float(os.getenv("TRACK_AUROC_TAU", "0.75"))  # deploy gate
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-8")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-4-8")

# --- Runtime ---
MODE = os.getenv("GLASSBOX_MODE", "posthoc")  # posthoc | live
# Response length cap for the chat/analyze path. The old 48-token default truncated answers
# mid-sentence. Each extra token costs one more per-token SAE pass + a longer post-hoc forward,
# so this trades latency for completeness. Override with MAX_NEW_TOKENS.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# --- System prompt ---
# Injected as a leading system turn before every generation (engine._encode), so it also frames
# the contrastive baseline — the system-induced "always-on" features then cancel in the
# attribution subtraction. Gemma templates that reject a system role fall back to merging it into
# the first user turn. Set SYSTEM_PROMPT="" to disable (raw chat, the prior behaviour).
_DEFAULT_SYSTEM_PROMPT = """You are a clinical decision-support assistant for licensed healthcare professionals. You provide accurate, evidence-based medical information grounded in current clinical guidelines and the peer-reviewed literature.

Operating principles:
- Accuracy first. Base answers on established evidence and current guidelines (e.g. ACOG, FDA, CDC, NICE, WHO, and specialty-society consensus). Never invent studies, statistics, doses, or citations. If the evidence is uncertain or mixed, say so explicitly rather than guessing.
- Calibrated confidence. Distinguish well-established facts from areas of genuine clinical uncertainty or ongoing debate. Do not overstate certainty; state the strength of the evidence where it changes the decision.
- Safety first. Proactively surface contraindications, clinically significant drug-drug interactions, boxed warnings, and risks in special populations - pregnancy and lactation, pediatric and geriatric patients, and renal or hepatic impairment. When an intervention is contraindicated, lead with that.
- Dosing. Give typical adult dosing only with the relevant caveats (indication, route, adjustments, maximum). Flag when individualized dosing or therapeutic monitoring is required.
- Scope and escalation. You support, you do not replace, the clinician's judgment and a complete patient assessment. Recommend confirming against primary sources and, for an individual patient, consulting the treating physician or pharmacist. Direct anyone describing an emergency (e.g. chest pain, anaphylaxis, stroke symptoms, overdose) to emergency services immediately.

Be clear and well-structured. Define abbreviations on first use. When a question is ambiguous, state the key assumptions you are making. Prioritize the information that changes clinical decisions."""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)
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
SENTRY_SEND_IO = os.getenv("SENTRY_SEND_IO", "0").lower() not in ("0", "false", "no", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

# --- Observability surfaces (read paths + eval) ---
PHOENIX_UI_URL      = os.getenv("PHOENIX_UI_URL", PHOENIX_ENDPOINT)        # iframe src
SENTRY_ORG_SLUG     = os.getenv("SENTRY_ORG_SLUG", "")
SENTRY_PROJECT_SLUG = os.getenv("SENTRY_PROJECT_SLUG", "")
SENTRY_AUTH_TOKEN   = os.getenv("SENTRY_AUTH_TOKEN", "")                    # internal-integration, event:read+project:read
SENTRY_API_BASE     = os.getenv("SENTRY_API_BASE", "https://sentry.io").rstrip("/")
SENTRY_ORG_URL      = os.getenv("SENTRY_ORG_URL", "https://sentry.io")     # deep-link host
EVAL_LLM_PROVIDER   = os.getenv("EVAL_LLM_PROVIDER", "anthropic")
EVAL_LLM_MODEL      = os.getenv("EVAL_LLM_MODEL", "claude-haiku-4-5-20251001")

MASK_TOKENS = ["<bos>", "<start_of_turn>", "<end_of_turn>"]

# SAE wiring sanity check run once at load (runtime._run_recon_check). A correctly-wired Gemma
# Scope SAE reconstructs its own training-layer resid_post with high cosine; a low value means
# the SAE/layer/dtype are mismatched and the whole feature cloud is noise. Surfaced on /health.
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
