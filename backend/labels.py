"""Neuronpedia feature stats + label cache, with Claude auto-interp for the gaps.

OWNER: Lane A. CPU only — must NOT import torch (Global Constraint). One GET per feature returns
the auto-interp label, the activation stats used for ranking (`maxActApprox`, `frac_nonzero`), AND
the top activating examples. When Neuronpedia has no explanation for a feature (~half of the
attribution cloud's top features), we label it ourselves: send its activating examples to Claude
Haiku and get back {label, is_structural}. Results persist to an on-disk cache so labelling is a
one-time cost; everything degrades to "feature N" and never raises into a chat turn.
"""

from __future__ import annotations

import json
import os
import threading

from . import config

_stats: dict[int, dict] = {}  # session cache (includes unresolved "feature N" fallbacks)
_disk: dict | None = None  # persistent cache of resolved labels (labelled features only)
_lock = threading.Lock()
_UA = {"User-Agent": "glassbox-hackathon/0.1"}

_CACHE_PATH = os.getenv(
    "AUTOINTERP_CACHE", os.path.join(os.path.dirname(__file__), "autointerp_cache.json")
)
_N_EXAMPLES = int(os.getenv("AUTOINTERP_N_EXAMPLES", "15"))
_WINDOW_RADIUS = int(os.getenv("AUTOINTERP_WINDOW_RADIUS", "10"))  # tokens each side of the peak
_CLAUDE_TIMEOUT = float(os.getenv("AUTOINTERP_CLAUDE_TIMEOUT", "20"))


def _disk_cache() -> dict:
    """Lazily load the persistent label cache (resolved labels survive restarts)."""
    global _disk
    if _disk is None:
        try:
            with open(_CACHE_PATH) as f:
                _disk = json.load(f)
        except Exception:
            _disk = {}
    assert _disk is not None
    return _disk


def _save_disk() -> None:
    try:
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_disk, f)
        os.replace(tmp, _CACHE_PATH)  # atomic; cache is an optimisation, never break a turn over it
    except Exception:
        pass


def get_feature_stats(index: int, timeout: float = 6.0) -> dict:
    """Return {label, max_act, density, is_structural} for an SAE feature index (cached).

    Resolution order: in-memory cache -> disk cache -> Neuronpedia explanation -> Claude auto-interp
    on the activating examples. Degrades to {'feature {index}', None, None, False} on any error."""
    if index in _stats:
        return _stats[index]
    disk = _disk_cache()
    if str(index) in disk:
        _stats[index] = disk[str(index)]
        return _stats[index]

    import httpx

    url = config.NP_FEATURE_URL.format(
        model=config.NP_MODEL, source=config.NP_SOURCE, index=index
    )
    label = max_act = density = None
    activations = None
    try:
        r = httpx.get(url, headers=_UA, timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            exps = d.get("explanations") or []
            if exps:
                label = exps[0].get("description")
            max_act = d.get("maxActApprox")
            density = d.get("frac_nonzero")
            activations = d.get("activations")
    except Exception:
        pass

    is_structural = False
    resolved = label is not None
    # No Neuronpedia explanation -> label it ourselves from the activating examples.
    if label is None and config.AUTOINTERP and activations:
        got = _autointerp(activations)
        if got:
            label, is_structural, resolved = got["label"], got["is_structural"], True

    stats = {
        "label": (label or f"feature {index}").strip(),
        "max_act": float(max_act) if max_act else None,
        "density": float(density) if density is not None else None,
        "is_structural": bool(is_structural),
    }
    _stats[index] = stats
    if resolved:  # persist only real labels — a transient failure ("feature N") can retry next run
        with _lock:
            disk[str(index)] = stats
            _save_disk()
    return stats


# ---- Claude auto-interp --------------------------------------------------------
_SYSTEM = (
    "You are a mechanistic-interpretability researcher labelling a single sparse-autoencoder "
    "feature of a medical chatbot (Gemma-3-4b-it, layer 17). You are shown text excerpts where "
    "the feature fired most strongly; the single peak-activating token in each excerpt is wrapped "
    "in <<double angle brackets>>. Find the pattern COMMON ACROSS ALL excerpts and describe the "
    "CONTEXT/ROLE the feature responds to — NOT merely the literal bracketed token. A feature "
    "firing on 'and' or '.' across unrelated sentences is detecting sentence/clause STRUCTURE, not "
    "those words. Then record a label (a noun phrase, <=10 words, no hedging, no trailing period) "
    "and set is_structural=true when the feature is grammatical/structural/formatting (punctuation, "
    "whitespace, paragraph or line breaks, list/markdown scaffolding, discourse glue, token "
    "fragments/morphology) rather than a concept; otherwise false."
)

_TOOL = {
    "name": "record_feature_label",
    "description": "Record the interpretation of the SAE feature.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Noun phrase, <=10 words, no trailing period."},
            "is_structural": {
                "type": "boolean",
                "description": "True if grammatical/structural/formatting rather than a concept.",
            },
        },
        "required": ["label", "is_structural"],
        "additionalProperties": False,
    },
}


def _highlighted_windows(activations, n: int, radius: int) -> list[str]:
    """Trim each example to a +-`radius`-token window around its peak-activating token and wrap
    the peak in <<...>>. Cuts the ~512-token examples to ~20 tokens each (~9k -> ~300 total)."""
    out: list[str] = []
    for a in (activations or [])[:n]:
        toks = a.get("tokens") or []
        vals = a.get("values") or []
        if not toks or not vals:
            continue
        mvi = a.get("maxValueTokenIndex")
        if mvi is None:
            mvi = max(range(len(vals)), key=lambda i: vals[i])
        lo, hi = max(0, mvi - radius), min(len(toks), mvi + radius + 1)
        out.append("".join(f"<<{toks[i]}>>" if i == mvi else toks[i] for i in range(lo, hi)))
    return out


def _autointerp(activations) -> dict | None:
    """Label a feature from its activating examples via Claude Haiku. None on any failure."""
    if not config.ANTHROPIC_API_KEY:
        return None
    windows = _highlighted_windows(activations, _N_EXAMPLES, _WINDOW_RADIUS)
    if not windows:
        return None
    try:
        import anthropic
    except Exception:
        return None
    examples = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(windows))
    user = (
        f"Top activating excerpts (peak token in <<>>):\n{examples}\n\n"
        "Call record_feature_label with the shared concept across ALL excerpts."
    )
    try:
        client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY, timeout=_CLAUDE_TIMEOUT, max_retries=2
        )
        resp = client.messages.create(
            model=config.AUTOINTERP_MODEL,
            max_tokens=200,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "record_feature_label"},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "record_feature_label":
                lbl = (block.input.get("label") or "").strip().rstrip(".")
                if lbl:
                    return {"label": lbl, "is_structural": bool(block.input.get("is_structural"))}
    except Exception:
        return None
    return None


def get_label(index: int, timeout: float = 6.0) -> str:
    """Auto-interp/Neuronpedia description for an SAE feature index (cached)."""
    return get_feature_stats(index, timeout)["label"]


def get_labels(indices) -> dict[int, str]:
    """Fetch+cache a batch of labels. Use offline to pre-warm before a demo."""
    return {i: get_label(i) for i in indices}


def prewarm(indices) -> None:
    """Resolve + cache labels for the given feature indices offline, so the live demo path never
    makes a synchronous Claude call. Run over the expected top-candidate indices before a demo."""
    for i in indices:
        get_feature_stats(i)


def preload_from_s3() -> None:
    """Optional: bulk-load an S3 dump into the cache at startup (no live calls during demo)."""
    ...
