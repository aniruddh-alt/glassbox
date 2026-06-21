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


def fit_and_validate(rows: list[dict]) -> dict:
    """Stratified train/held-out split → diff-of-means direction + calibrated probe.
    Measures held-out AUROC and a plain-LogReg baseline. direction/calibrator/threshold
    are fit on ALL rows for deployment. Returns insufficient_data if a class is too small."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from . import persona

    fail = {
        "status": "insufficient_data", "auroc": None, "baseline_auroc": None,
        "n_kept": len(rows), "calibrated": False,
        "direction": None, "calibrator": None, "threshold": None,
    }
    y = np.array([int(r["label"]) for r in rows])
    if len(rows) < 6 or (y == 1).sum() < 3 or (y == 0).sum() < 3:
        return fail

    X = torch.stack([r["act_resp"].float() for r in rows])  # [n, d_in]
    idx = np.arange(len(rows))
    try:
        tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=0)
    except ValueError:
        return fail
    if y[te].sum() == 0 or y[te].sum() == len(te):  # held-out has only one class
        return fail

    # Validation probe trained on the train split only.
    clf, _ = persona.train_probe(X[tr], y[tr])
    probs = clf.predict_proba(X[te].numpy())[:, 1]
    auroc = float(roc_auc_score(y[te], probs))

    base = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr].numpy(), y[tr])
    baseline_auroc = float(roc_auc_score(y[te], base.predict_proba(X[te].numpy())[:, 1]))

    # Deployable artifacts fit on ALL rows.
    direction = persona.persona_vector(X[y == 1], X[y == 0])
    clf_all, threshold = persona.train_probe(X, y)
    calibrated = clf_all.__class__.__name__ == "CalibratedClassifierCV"
    return {
        "status": "ok", "auroc": auroc, "baseline_auroc": baseline_auroc,
        "n_kept": len(rows), "calibrated": calibrated,
        "direction": direction, "calibrator": clf_all, "threshold": threshold,
    }
