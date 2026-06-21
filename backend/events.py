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
    model: str,
    layer: int,
) -> CognitionEvent:
    """trackers: dict[str -> {score,proj,flag,...}] from persona.score_all_trackers ({} until
    probes exist). features: list[Feature-like dicts] with labels already attached.

    Family B is WIP: with no "uncertainty" tracker, the meter fields stay null. The event-level
    flag is raised by any tracker whose own monitor flag is true.
    """
    unc = trackers.get("uncertainty")
    if unc is None:
        uncertainty = uncertainty_proj = uncertainty_proj_pre = None
    else:
        uncertainty = unc.get("score")
        uncertainty_proj = unc.get("proj")
        uncertainty_proj_pre = unc.get("proj_pre")
    flag = any(
        bool(t.get("flag", (t.get("score") or 0.0) >= DEFAULT_THRESHOLD))
        for t in trackers.values()
    )
    return CognitionEvent(
        message_id=message_id,
        ts=ts,
        model=model,
        layer=layer,
        io=IO(user_msg=user_msg, response=response),
        uncertainty=uncertainty,
        uncertainty_proj=uncertainty_proj,
        uncertainty_proj_pre=uncertainty_proj_pre,
        flag=flag,
        severity="warning" if flag else "info",
        trackers=trackers,
        features=features,
        adjudication=None,
    )
