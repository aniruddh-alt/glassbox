"""Family B — persona-vector + calibrated probe detectors (the RELIABLE signal).

OWNER: Lane B. Per concept: a unit diff-of-means direction at the configured layer over RESPONSE
tokens, PLUS a calibrated logistic-regression probe. Detection = projection / predict_proba.
Steering is OUT of scope — detection only. Validate with held-out AUROC vs the LogReg
baseline (see validation/). Method follows Persona Vectors (arXiv 2507.21509).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_trackers: dict[str, dict] = {}
ARTIFACT_DIR = Path(__file__).with_name("artifacts")
_ALERT_DIRECTIONS = {"high", "low"}


def persona_vector(act_pos, act_neg):
    """Unit diff-of-means direction = normalize(mean(act_pos) - mean(act_neg)).
    act_pos/act_neg: [n_examples, d_in] tensors of mean-pooled response activations."""
    d = act_pos.float().mean(0) - act_neg.float().mean(0)
    return d / (d.norm() + 1e-8)


def project(acts, direction, *, norm_mean=None, norm_std=None):
    """Scalar projection of one or many activations onto a (unit) direction.
    acts: [d_in] or [n, d_in]; returns float or [n] tensor.
    Optional norm_mean/norm_std apply per-dimension z-scoring before projection (normed diff-of-means).

    The direction and norm tensors come from JSON artifacts and so land on CPU, while live
    activations sit on the model's device (CUDA on the pod). Reconcile both onto the activations'
    device — otherwise the matmul raises a device mismatch and every tracker is silently skipped."""
    a = acts.float()
    dev = a.device
    if norm_mean is not None and norm_std is not None:
        mu = _as_tensor(norm_mean).to(dev)
        sd = _as_tensor(norm_std).to(dev)
        a = (a - mu) / (sd + 1e-8)
    return a @ _unit_direction(direction).float().to(dev)


def _as_tensor(x):
    import torch

    if hasattr(x, "detach"):
        return x.detach().float()
    return torch.as_tensor(x, dtype=torch.float32)


def _as_numpy_row(x):
    import numpy as np

    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()[None]
    return np.asarray(x, dtype="float32")[None]


def _unit_direction(direction):
    d = _as_tensor(direction).flatten()
    norm = d.norm()
    if d.numel() == 0 or float(norm) <= 1e-8:
        raise ValueError("tracker direction must be a non-zero vector")
    return d / norm


def register_tracker(
    tracker_id: str,
    *,
    direction,
    threshold: float = 0.5,
    calibrator: Any | None = None,
    meta: dict[str, Any] | None = None,
    alert_direction: str = "high",
    projection_center: float = 0.0,
    projection_scale: float = 1.0,
    norm_mean=None,
    norm_std=None,
    direction_method: str | None = None,
) -> dict:
    """Register a ready persona-vector tracker.

    `score` is always a bounded probability-like value in [0, 1]. If no calibrated classifier is
    available, the raw vector projection stays in `proj` and `score` is a sigmoid of that
    projection. `alert_direction="low"` is for protective concepts such as risk awareness, where
    low activation should raise the monitor flag.
    """
    tid = tracker_id.strip()
    if not tid:
        raise ValueError("tracker_id is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if alert_direction not in _ALERT_DIRECTIONS:
        raise ValueError(f"alert_direction must be one of {sorted(_ALERT_DIRECTIONS)}")
    if projection_scale <= 0:
        raise ValueError("projection_scale must be > 0")

    tracker = {
        "dir": _unit_direction(direction),
        "threshold": float(threshold),
        "calibrator": calibrator,
        "meta": meta or {},
        "alert_direction": alert_direction,
        "projection_center": float(projection_center),
        "projection_scale": float(projection_scale),
        "norm_mean": _as_tensor(norm_mean) if norm_mean is not None else None,
        "norm_std": _as_tensor(norm_std) if norm_std is not None else None,
        "direction_method": direction_method,
    }
    _trackers[tid] = tracker
    return tracker


def clear_trackers() -> None:
    """Clear registered trackers. Primarily for tests and hot-reload paths."""
    _trackers.clear()


def load_tracker_artifact(path: str | Path) -> str | None:
    """Load one JSON artifact if it contains a ready direction vector.

    Template-only artifacts are intentionally ignored: they describe training data but cannot score
    live activations until the offline pipeline writes a `direction` field.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    direction = data.get("direction") or data.get("dir")
    if direction is None:
        return None

    tid = data.get("id") or data.get("tracker_id") or data.get("concept") or p.stem
    meta = {
        "concept": data.get("concept", tid),
        "description": data.get("description"),
        "user_defined": bool(data.get("user_defined", False)),
        "artifact": str(p),
    }
    register_tracker(
        str(tid),
        direction=direction,
        threshold=float(data.get("threshold", 0.5)),
        meta=meta,
        alert_direction=data.get("alert_direction", "high"),
        projection_center=float(data.get("projection_center", 0.0)),
        projection_scale=float(data.get("projection_scale", 1.0)),
        norm_mean=data.get("norm_mean"),
        norm_std=data.get("norm_std"),
        direction_method=data.get("direction_method"),
    )
    return str(tid)


def load_artifacts(
    artifact_dir: str | Path = ARTIFACT_DIR,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Load all ready tracker artifacts from a directory.

    `exclude` skips artifacts by id (filename stem) — used to keep a trained probe on disk
    while leaving it out of the live tracker set.
    """
    root = Path(artifact_dir)
    if not root.exists():
        return []
    skip = set(exclude)
    loaded: list[str] = []
    for path in sorted(root.glob("*.json")):
        if path.stem in skip:
            continue
        tid = load_tracker_artifact(path)
        if tid:
            loaded.append(tid)
    return loaded


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
    out: dict[str, dict] = {}
    if act_resp is None:
        return out

    for tid, t in _trackers.items():
        try:
            nm, ns = t.get("norm_mean"), t.get("norm_std")
            proj = float(project(act_resp, t["dir"], norm_mean=nm, norm_std=ns))
            proj_pre = (
                float(project(act_last, t["dir"], norm_mean=nm, norm_std=ns))
                if act_last is not None
                else None
            )
        except Exception as e:  # noqa: BLE001 - one malformed artifact must not fail the turn
            print(f"[persona] skipping tracker {tid}: {e}")
            continue
        clf = t.get("calibrator")
        if clf:
            score = float(clf.predict_proba(_as_numpy_row(act_resp))[:, 1][0])
        else:
            z = (proj - t.get("projection_center", 0.0)) / t.get("projection_scale", 1.0)
            z = max(-60.0, min(60.0, z))
            score = 1.0 / (1.0 + math.exp(-z))
        threshold = t.get("threshold", 0.5)
        alert_direction = t.get("alert_direction", "high")
        flag = score <= threshold if alert_direction == "low" else score >= threshold
        out[tid] = {
            "score": score,
            "proj": proj,
            "proj_pre": proj_pre,
            "flag": flag,
            "reliable": True,
            "status": "ready",
            "user_defined": t.get("meta", {}).get("user_defined", False),
            "alert_direction": alert_direction,
        }
    return out
