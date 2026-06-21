"""User-defined concepts on demand. User names a concept → contrastive pairs (Oumi/Claude)
→ persona vector at layer 12 → cached tracker. SAE features are a FIXED dictionary; persona
vectors are what make arbitrary user concepts trackable.

OWNER: Lane B.
"""
from __future__ import annotations

import uuid

_jobs: dict[str, dict] = {}


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    return "".join(keep).strip("-")[:24] or "concept"


def create_job(request: str) -> str:
    """Register a new tracking job in 'pending'. Returns its tracker_id."""
    tracker_id = f"{_slug(request)}-{uuid.uuid4().hex[:6]}"
    _jobs[tracker_id] = {
        "tracker_id": tracker_id,
        "request": request,
        "status": "pending",
        "progress": {"step": "queued", "pct": 0},
        "trait_name": None,
        "auroc": None,
        "baseline_auroc": None,
        "n_kept": None,
        "verdict": None,
        "error": None,
    }
    return tracker_id


def update_job(tracker_id: str, **fields) -> None:
    job = _jobs.get(tracker_id)
    if job is not None:
        job.update(fields)


def get_job(tracker_id: str) -> dict | None:
    return _jobs.get(tracker_id)
