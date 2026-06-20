"""SHARED SOURCE OF TRUTH for the cognition_event contract.

Mirrored byte-for-byte by frontend/src/types.ts and fixtures/cognition_event.sample.json.
No lane changes a field here without 3-way agreement (see LANES.md, contract #1).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Severity = Literal["info", "warning"]


class IO(BaseModel):
    user_msg: str
    response: str


class Tracker(BaseModel):
    """One concept being monitored. Built-ins and user-defined share this shape."""
    score: float = Field(..., description="calibrated probability in [0,1] — the meter")
    proj: float = Field(..., description="diff-of-means projection (baseline / eng view)")
    flag: bool = Field(..., description="score >= calibrated threshold")
    reliable: bool = Field(True, description="Family B (probe) = reliable; never set for SAE labels")
    proj_pre: float | None = Field(None, description="last-prompt-token projection (pre-gen early warning)")
    user_defined: bool = False
    status: Literal["computing", "ready"] = "ready"


class Feature(BaseModel):
    """One SAE feature firing — Family A, EXPLORATORY. Labels are auto-interp, may be wrong."""
    index: int
    label: str
    act: float
    source: str = "12-gemmascope-res-16k"
    caveat: str = "auto-interp label, may be unreliable"
    tracked: str | None = None  # tracker_id if this feature maps to a tracked concept


class Adjudication(BaseModel):
    """Anthropic-prize Claude honesty-judge. Present only when flag==True; filled async."""
    verdict: Literal["likely_correct", "likely_hallucinated", "uncertain"]
    rationale: str
    by: Literal["claude"] = "claude"


class CognitionEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    type: Literal["event"] = "event"  # discriminator vs streamed {"type":"token"} lines
    message_id: str
    ts: float  # unix seconds (pass in; do not call time.time() in pure builders if reproducibility matters)
    model: str = "gemma-2-2b-it"
    layer: int = 12
    io: IO

    # --- Family B: the RELIABLE signal (primary meter mirrors trackers["uncertainty"]) ---
    uncertainty: float = Field(..., description="calibrated LogReg prob in [0,1] — the meter")
    uncertainty_proj: float
    uncertainty_proj_pre: float | None = None
    flag: bool = Field(..., description="uncertainty >= calibrated max-F1 threshold (confident-wrong zone)")
    severity: Severity = "info"

    trackers: dict[str, Tracker] = Field(default_factory=dict)

    # --- Family A: EXPLORATORY, labels unreliable ---
    features: list[Feature] = Field(default_factory=list)

    # --- Anthropic prize, present only when flagged (filled async) ---
    adjudication: Adjudication | None = None


class TokenLine(BaseModel):
    """A streamed per-token line (live mode). Final line of /api/chat is the CognitionEvent."""
    type: Literal["token"] = "token"
    text: str
    top_features: list[Feature] = Field(default_factory=list)
    uncertainty: float | None = None
