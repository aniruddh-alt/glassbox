"""Family B — persona-vector + calibrated probe detectors (the RELIABLE signal).

OWNER: Lane B. Per concept: a unit diff-of-means direction at layer 12 over RESPONSE
tokens, PLUS a calibrated logistic-regression probe. Detection = projection / predict_proba.
Steering is OUT of scope — detection only. Validate with held-out AUROC vs the LogReg
baseline (see validation/). Method follows Persona Vectors (arXiv 2507.21509).
"""
from __future__ import annotations

# Loaded at startup: {tracker_id -> {"dir": Tensor, "calibrator": sklearn, "threshold": float, "meta": {...}}}
_trackers: dict[str, dict] = {}


def persona_vector(act_pos, act_neg):
    """Unit diff-of-means direction = normalize(mean(act_pos) - mean(act_neg)) at layer 12."""
    # TODO(Lane B):
    # d = act_pos.mean(0) - act_neg.mean(0)
    # return d / d.norm()
    raise NotImplementedError("Lane B: diff-of-means")


def train_probe(X, y):
    """Calibrated logistic regression on residual activations. Returns (calibrator, threshold)."""
    # TODO(Lane B):
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.calibration import CalibratedClassifierCV
    # clf = CalibratedClassifierCV(LogisticRegression(class_weight="balanced"), method="sigmoid").fit(X, y)
    # threshold = pick_threshold(clf, X_val, y_val)   # max-F1 on PR curve
    # return clf, threshold
    raise NotImplementedError("Lane B: calibrated probe")


def score_all_trackers(act_last, act_resp) -> dict[str, dict]:
    """act_last: last-prompt-token activation (pre-gen early warning).
    act_resp: response-token-average activation.
    Returns {tracker_id -> {score, proj, proj_pre, flag, reliable, status, user_defined}}.
    """
    out: dict[str, dict] = {}
    # TODO(Lane B): for each tracker: proj = act_resp @ dir; score = calibrator.predict_proba(act_resp);
    #   proj_pre = act_last @ dir; flag = score >= threshold.
    return out
