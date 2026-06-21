"""Assemble the ONE cognition_event per message. Pure CPU. No torch, no tensors out.

OWNER: Lane A.
"""

from __future__ import annotations

from .config import DEFAULT_THRESHOLD
from .schema import IO, CognitionEvent


def build_cognition_event(
    *,
    message_id: str,
    ts: float,
    user_msg: str,
    response: str,
    trackers: dict,
    features: list,
    model: str = "gemma-2-2b-it",
    layer: int = 12,
) -> CognitionEvent:
    """trackers: dict[str -> {score,proj,flag,...}] from science.persona.score_all_trackers.
    features: list[Feature-like dicts] from science.sae.sae_topk (labels already attached).
    """
    unc = trackers.get("uncertainty", {})
    flag = bool(unc.get("flag", unc.get("score", 0.0) >= DEFAULT_THRESHOLD))
    return CognitionEvent(
        message_id=message_id,
        ts=ts,
        model=model,
        layer=layer,
        io=IO(user_msg=user_msg, response=response),
        uncertainty=unc.get("score", 0.0),
        uncertainty_proj=unc.get("proj", 0.0),
        uncertainty_proj_pre=unc.get("proj_pre"),
        flag=flag,
        severity="warning" if flag else "info",
        trackers=trackers,
        features=features,
        adjudication=None,  # filled async by fanout → claude judge if flag
    )
