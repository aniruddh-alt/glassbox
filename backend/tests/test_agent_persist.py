"""The agent persists a deployed probe to a JSON artifact so it reloads as a built-in tracker.
The key property: a loaded artifact must score GRADED (pos>0.5>neg, both strictly inside (0,1)) on
real-magnitude activations, not saturate to 0/1 — i.e. the projection calibration must survive."""
import json

import numpy as np
import torch

from backend.agent import tools
from backend.config import ModelConfig
from backend.science import persona


def _rows(n=12):
    """Separable rows at real residual-stream magnitude (~1e3), where sigmoid(raw_proj) would pin
    to 0/1 without calibration. Even labels -> class 0, odd -> class 1."""
    rows = []
    for i in range(n):
        lab = i % 2
        v = torch.zeros(8)
        v[0] = (3000.0 if lab else -3000.0) + (i % 3)
        v[1] = 500.0 if lab else -500.0
        rows.append({"act_resp": v, "label": lab})
    return rows


def _direction(rows):
    X = torch.stack([r["act_resp"] for r in rows])
    y = np.array([r["label"] for r in rows])
    return persona.persona_vector(X[y == 1], X[y == 0])


def test_proj_calibration_positive_scale():
    rows = _rows()
    center, scale = tools._proj_calibration(rows, _direction(rows))
    assert scale > 0
    assert np.isfinite(center) and np.isfinite(scale)


def test_persist_artifact_round_trip_scores_graded(tmp_path, monkeypatch):
    rows = _rows()
    direction = _direction(rows)
    monkeypatch.setattr(persona, "ARTIFACT_DIR", tmp_path)

    fit = {"direction": direction, "threshold": 0.5, "auroc": 1.0}
    ctx = {"tracker_id": "over-conf-test", "request": "flag over-confidence",
           "spec": {"trait_name": "over_confidence"}, "rows": rows}
    tools._persist_artifact(ctx, fit, ModelConfig())

    art = tmp_path / "over-conf-test.json"
    assert art.exists()
    data = json.loads(art.read_text())
    assert data["projection_scale"] > 0
    assert data["user_defined"] is True
    assert len(data["direction"]) == 8

    persona.clear_trackers()
    persona.load_tracker_artifact(art)
    pos = persona.score_all_trackers(None, torch.tensor([3000.0, 500.0, 0, 0, 0, 0, 0, 0]))["over-conf-test"]["score"]
    neg = persona.score_all_trackers(None, torch.tensor([-3000.0, -500.0, 0, 0, 0, 0, 0, 0]))["over-conf-test"]["score"]
    persona.clear_trackers()
    assert 0.0 < neg < 0.5 < pos < 1.0
