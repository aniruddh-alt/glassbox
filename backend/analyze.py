"""Shared turn analysis: generate (or synthesize) a response and assemble its CognitionEvent.
Branches on runtime.STATE; used by both /api/chat (streaming) and /api/analyze.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint) — the real path
calls pod_client.turn(...) over HTTP to the GPU service.
"""

from __future__ import annotations

import time
import uuid

from . import config, labels, runtime
from .events import build_cognition_event
from .schema import CognitionEvent

# Transitional cfg — Task 11 threads cfg into analyze_turn.
_cfg = config.load_config()

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"

# Markers that flag a feature LABEL as syntactic / surface-level (grammar, token patterns,
# discourse glue) rather than a concept. Used to down-rank them so abstract features surface.
_SYNTACTIC_MARKERS = (
    "followed by", "preceded by", "modifying", "modifier", "modifies",
    "preposition", "conjunction", "auxiliar", "pronoun", "determiner", "adverb",
    "contraction", "possessive", "suffix", "prefix", " tense", "ordinal", "cardinal number",
    "punctuation", "capitaliz", "syntactic", "grammar", "forms of", "the word ", "word pairs",
    "phrase", "it's ", "this/that", "this isn't", "for purposes", "general purpose",
    "sequence progression", "subsequent word", "subsequent action", "it depends",
    "okay", "greeting", "acknowledgment", "transition to task",
    # formatting / structural markers (also catch auto-interp labels for unlabelled features)
    "newline", "line break", "paragraph", "whitespace", "formatting", "markdown",
    "bullet", "list", "code snippet", "section break", "heading", "discourse",
    "token fragment", "word fragment", "subword", "morpholog", "plural", "start of",
    "end of sentence", "sentence boundary", "function word",
)


def _is_syntactic(label: str) -> bool:
    """True if the label describes grammar/surface structure rather than a concept."""
    low = label.lower()
    return any(m in low for m in _SYNTACTIC_MARKERS)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _rank_features(candidates: list[dict]) -> list[dict]:
    """Attach labels to candidate features and truncate to TOPK_EVENT.

    Two paths. When candidates carry `attr` (attribution ranking, the default) they are scored by
    causal effect on the response (act × grad·decoder); we label them (auto-interp fills the gaps),
    demote structural features by STRUCTURAL_PENALTY, then rank.

    Legacy fallback (raw-activation candidates, no `attr`): re-rank by activation RELATIVE to each
    feature's Neuronpedia ceiling (maxActApprox), since raw magnitude is dominated by high-norm
    grammatical features that fire far below ceiling. Down-rank syntactic labels and drop
    over-dense ones. Features whose ceiling is unknown fall back below the scored ones. See PLAN.md §7.
    """
    def _feature(f: dict, label: str) -> dict:
        return {
            "index": f["index"],
            "act": f["act"],
            "source": f["source"],
            "label": label,
            "caveat": DEFAULT_CAVEAT,
            "tracked": None,
        }

    def _unlabeled(feat: dict) -> bool:
        return feat["label"] == f"feature {feat['index']}"

    # Attribution path: candidates carry `attr` = causal effect on the response (act × grad·decoder).
    # Attach labels (auto-interp fills Neuronpedia's gaps) and DEMOTE structural features — punctuation,
    # formatting, discourse glue, token fragments — by STRUCTURAL_PENALTY so genuine concept/medical
    # features surface beneath them. Demote, not drop: a structural feature that truly dominates the
    # answer still survives. Structural = the auto-interp tag OR a structural/syntactic label.
    if any("attr" in c for c in candidates):
        scored: list[tuple[float, dict]] = []
        for f in candidates:
            s = labels.get_feature_stats(f["index"])
            structural = s.get("is_structural") or _is_syntactic(s["label"])
            penalty = config.STRUCTURAL_PENALTY if structural else 1.0
            scored.append((f.get("attr", 0.0) * penalty, _feature(f, s["label"])))
        scored.sort(key=lambda t: -t[0])
        if config.DROP_UNLABELED:
            semantic = [t for t in scored if not _unlabeled(t[1]) and not _is_syntactic(t[1]["label"])]
            structural = [t for t in scored if not _unlabeled(t[1]) and _is_syntactic(t[1]["label"])]
            if semantic:
                scored = semantic + structural + [t for t in scored if _unlabeled(t[1])]
        return [feat for _, feat in scored[: config.TOPK_EVENT]]

    scored: list[tuple[float, dict]] = []
    for f in candidates:
        s = labels.get_feature_stats(f["index"])
        # drop features Neuronpedia hasn't labelled ("feature N") — keeps the cloud legible
        if config.DROP_UNLABELED and s["label"] == f"feature {f['index']}":
            continue
        # drop generic/grammatical features: they fire on a large fraction of the corpus,
        # unlike specific (e.g. clinical) features which are rare
        d = s["density"]
        if config.DENSITY_MAX < 1.0 and d is not None and d > config.DENSITY_MAX:
            continue
        mx = s["max_act"]
        rel = (f["act"] / mx) if mx else 0.0
        # down-rank syntactic/surface features so abstract concepts outrank them
        if config.SYNTACTIC_PENALTY < 1.0 and _is_syntactic(s["label"]):
            rel *= config.SYNTACTIC_PENALTY
        scored.append((rel, _feature(f, s["label"])))

    if not scored:  # everything was unlabeled — show raw rather than an empty cloud
        return [
            _feature(f, labels.get_feature_stats(f["index"])["label"])
            for f in candidates[: config.TOPK_EVENT]
        ]
    scored.sort(key=lambda t: (-t[0], -t[1]["act"]))
    return [feat for _, feat in scored[: config.TOPK_EVENT]]


def _real_turn(messages: list[dict]) -> tuple[str, list[dict], dict, dict]:
    """Returns (answer, features, trackers, perf_stages) where perf_stages carries
    pod_roundtrip_ms, ranking_ms, and pod_stages from the pod's additive timings."""
    from . import pod_client

    # Transitional: load cfg here until Task 13 threads AppConfig down to analyze_turn.
    _cfg = config.load_config()
    _t_pod = time.perf_counter()
    r = pod_client.turn(messages, _cfg.pod, _cfg.pod_token, max_new=_cfg.model.max_new_tokens)
    pod_roundtrip_ms = (time.perf_counter() - _t_pod) * 1000.0

    _t_rank = time.perf_counter()
    feats = _rank_features(r["candidates"])
    ranking_ms = (time.perf_counter() - _t_rank) * 1000.0

    trackers = r.get("trackers") or {}
    pod_stages = r.get("timings") or {}
    stages = {"pod_roundtrip": pod_roundtrip_ms, "ranking": ranking_ms}
    return r["answer"], feats, trackers, {"stages": stages, "pod_stages": pod_stages}


def analyze_turn(
    messages: list[dict],
    *,
    message_id: str | None = None,
    ts: float | None = None,
    strict: bool = False,
) -> tuple[str, CognitionEvent, dict]:
    message_id = message_id or uuid.uuid4().hex
    ts = time.time() if ts is None else ts
    _t_turn = time.perf_counter()

    if runtime.STATE["mode"] == "real":
        try:
            answer, feats, trackers, timing_data = _real_turn(messages)
        except Exception as e:
            runtime.refresh_pod_health(_cfg)
            from . import fanout
            fanout.report_error("pod-down", e)  # instrument_unhealthy concern → Sentry (sanitized)
            if strict:
                raise RuntimeError(f"pod turn failed (strict mode, no fallback): {e}") from e
            print(f"[analyze] pod turn failed ({e}); synthetic fallback for this turn")
            from . import fallback

            answer, feats = fallback.synth_turn(messages)
            trackers = {}
            timing_data = {"stages": {}, "pod_stages": {}}
    else:
        if strict:
            raise RuntimeError(
                f"backend mode={runtime.STATE['mode']} (strict mode requires real pod path)"
            )
        from . import fallback

        answer, feats = fallback.synth_turn(messages)
        trackers = {}
        timing_data = {"stages": {}, "pod_stages": {}}

    if strict:
        from .fallback import is_synthetic_response

        if is_synthetic_response(answer):
            raise RuntimeError("synthetic response detected (strict mode)")

    event = build_cognition_event(
        message_id=message_id,
        ts=ts,
        user_msg=_last_user(messages),
        response=answer,
        trackers=trackers,
        features=feats,
        model=config.MODEL_ID,
        layer=config.LAYER,
    )
    turn_ms = (time.perf_counter() - _t_turn) * 1000.0
    perf: dict = {
        "turn_ms": turn_ms,
        "stages": timing_data["stages"],
        "pod_stages": timing_data["pod_stages"],
    }
    return answer, event, perf
