"""Shared turn analysis: generate (or synthesize) a response and assemble its CognitionEvent.
Branches on runtime.STATE; used by both /api/chat (streaming) and /api/analyze.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint) — the real path
imports engine/science lazily inside _real_turn.
"""

from __future__ import annotations

import time
import uuid

from . import config, labels, runtime
from .events import build_cognition_event
from .schema import CognitionEvent

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _label_features(features: list[dict]) -> list[dict]:
    """Attach Neuronpedia labels to bare {index,act,source} features (real path).
    Features that already carry a label (fallback path) are left untouched."""
    for f in features:
        if not f.get("label"):
            f["label"] = labels.get_label(f["index"])
        f.setdefault("caveat", DEFAULT_CAVEAT)
        f.setdefault("tracked", None)
    return features


def _real_turn(messages: list[dict]) -> tuple[str, list[dict], dict]:
    from . import engine
    from .science.feature_provider import get_provider
    from .science.persona import score_all_trackers

    res = engine.generate_and_capture(messages)
    tok = res["tok"]
    special = {tok.convert_tokens_to_ids(t) for t in config.MASK_TOKENS}
    acts, rs = res["acts"], res["resp_start"]
    resp_acts = acts[rs:]
    resp_ids = res["out_ids"][rs:].tolist()

    feats = get_provider().features_for(
        res["answer"], activations=resp_acts, token_ids=resp_ids, special_ids=special
    )
    _label_features(feats)

    act_last = acts[rs - 1] if rs > 0 else None
    act_resp = resp_acts.float().mean(0) if resp_acts.shape[0] > 0 else None
    trackers = score_all_trackers(act_last, act_resp)  # {} until probes exist
    return res["answer"], feats, trackers


def analyze_turn(
    messages: list[dict], *, message_id: str | None = None, ts: float | None = None
) -> tuple[str, CognitionEvent]:
    message_id = message_id or uuid.uuid4().hex
    ts = time.time() if ts is None else ts

    if runtime.STATE["mode"] == "real":
        answer, feats, trackers = _real_turn(messages)
    else:
        from . import fallback

        answer, feats = fallback.synth_turn(messages)
        trackers = {}

    event = build_cognition_event(
        message_id=message_id,
        ts=ts,
        user_msg=_last_user(messages),
        response=answer,
        trackers=trackers,
        features=_label_features(feats),
        model=config.MODEL_ID,
        layer=config.LAYER,
    )
    return answer, event
