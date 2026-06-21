"""Family B — persona-vector + calibrated probe detectors (the RELIABLE signal).

OWNER: Lane B. Per concept: a unit diff-of-means direction at layer 12 over RESPONSE
tokens, PLUS a calibrated logistic-regression probe. Detection = projection / predict_proba.
Steering is OUT of scope — detection only. Validate with held-out AUROC vs the LogReg
baseline (see validation/). Method follows Persona Vectors (arXiv 2507.21509).
"""

from __future__ import annotations

_trackers: dict[str, dict] = {}


def persona_vector(act_pos, act_neg):
    """Unit diff-of-means direction = normalize(mean(act_pos) - mean(act_neg)) at layer 12.
    act_pos/act_neg: [n_examples, d_in] tensors of mean-pooled response activations."""
    d = act_pos.float().mean(0) - act_neg.float().mean(0)
    return d / (d.norm() + 1e-8)


def project(acts, direction):
    """Scalar projection of one or many activations onto a (unit) direction.
    acts: [d_in] or [n, d_in]; returns float or [n] tensor."""
    return acts.float() @ direction.float()


def train_probe(X, y):
    """Calibrated logistic regression on residual activations. Returns (clf, threshold).
    X: [n, d_in] numpy/tensor, y: [n] {0,1}. Threshold = max-F1 point on the PR curve."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve

    Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    yn = np.asarray(y)
    base = LogisticRegression(class_weight="balanced", max_iter=1000)
    # cv=3 needs >=3 per class; fall back to no calibration for tiny smoke-test sets.
    try:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(Xn, yn)
    except ValueError:
        clf = base.fit(Xn, yn)
    p = clf.predict_proba(Xn)[:, 1]
    prec, rec, thr = precision_recall_curve(yn, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    threshold = float(thr[max(0, f1[:-1].argmax())]) if len(thr) else 0.5
    return clf, threshold


def score_all_trackers(act_last, act_resp) -> dict[str, dict]:
    """act_last: last-prompt-token activation (pre-gen early warning).
    act_resp: response-token-average activation.
    Returns {tracker_id -> {score, proj, proj_pre, flag, reliable, status, user_defined}}.
    """
    import numpy as np

    out: dict[str, dict] = {}
    for tid, t in _trackers.items():
        proj = float(project(act_resp, t["dir"]))
        proj_pre = float(project(act_last, t["dir"])) if act_last is not None else None
        clf = t.get("calibrator")
        score = (
            float(clf.predict_proba(np.asarray(act_resp.detach().cpu())[None])[:, 1][0])
            if clf
            else proj
        )
        out[tid] = {
            "score": score,
            "proj": proj,
            "proj_pre": proj_pre,
            "flag": score >= t.get("threshold", 0.5),
            "reliable": True,
            "status": "ready",
            "user_defined": t.get("meta", {}).get("user_defined", False),
        }
    return out
