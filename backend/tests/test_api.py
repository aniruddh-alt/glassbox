import json

from fastapi.testclient import TestClient

from backend import runtime
from backend.app import app
from backend.schema import CognitionEvent

client = TestClient(app)


def _force_fallback():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    p = r.json()
    assert {"mode", "model", "layer", "trackers"} <= set(p)


def test_chat_streams_tokens_then_one_event():
    _force_fallback()
    r = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Is ibuprofen safe in the third trimester?"}]},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    parsed = [json.loads(ln) for ln in lines]
    assert len(parsed) >= 2
    assert all(p["type"] == "token" for p in parsed[:-1])
    assert parsed[-1]["type"] == "event"
    ev = parsed[-1]
    CognitionEvent(**ev)  # schema-valid
    assert ev["uncertainty"] is None
    assert ev["flag"] is False
    assert len(ev["features"]) > 0
    assert ev["features"][0]["label"]


def test_analyze_returns_event():
    _force_fallback()
    r = client.post("/api/analyze", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    CognitionEvent(**r.json())


def test_feature_label(monkeypatch):
    from backend import labels

    monkeypatch.setattr(labels, "get_label", lambda i, **k: f"feat-{i}")
    r = client.get("/api/feature/123")
    assert r.json()["label"] == "feat-123"


def test_track_generates_harmfulness_probe_artifact(monkeypatch, tmp_path):
    from backend.science import concept_synth

    monkeypatch.setattr(concept_synth, "DEFAULT_ARTIFACT_DIR", tmp_path, raising=False)
    r = client.post(
        "/api/track",
        json={
            "concept": "harmfulness",
            "description": "Detect unsafe medical advice and dangerous omissions.",
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["tracker_id"] == "harmful"
    assert body["status"] == "computing"
    assert body["artifact"].endswith("harmful.json")
    assert (tmp_path / "harmful.json").exists()


def test_track_status_reports_computing_until_artifact_has_direction(monkeypatch, tmp_path):
    from backend.science import concept_synth

    monkeypatch.setattr(concept_synth, "DEFAULT_ARTIFACT_DIR", tmp_path, raising=False)
    client.post("/api/track", json={"concept": "harmfulness", "description": "Detect harm."})

    r = client.get("/api/track/harmful")

    assert r.status_code == 200
    assert r.json()["status"] == "computing"
