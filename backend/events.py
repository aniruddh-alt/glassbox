"""Assemble the ONE cognition_event per message. Pure CPU. No torch, no tensors out.

OWNER: Lane A.
"""

from __future__ import annotations

from .schema import IO, CognitionEvent

# Default probe-flag threshold — matches ProbeConfig.default_threshold.
_DEFAULT_THRESHOLD: float = 0.5


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

    Family B: the primary meter fields (uncertainty_*) carry the over_confidence probe score
    when that probe is active — legacy field names kept for schema/observability compatibility.
    The event-level flag is raised by any tracker whose own monitor flag is true.
    """
    meter = trackers.get("over_confidence") or trackers.get("uncertainty")
    if meter is None:
        uncertainty = uncertainty_proj = uncertainty_proj_pre = None
    else:
        uncertainty = meter.get("score")
        uncertainty_proj = meter.get("proj")
        uncertainty_proj_pre = meter.get("proj_pre")
    flag = any(
        bool(t.get("flag", (t.get("score") or 0.0) >= _DEFAULT_THRESHOLD))
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
