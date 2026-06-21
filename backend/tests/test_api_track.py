from fastapi.testclient import TestClient

from backend import app as app_module
from backend.science import concept_synth as cs


def test_post_track_returns_pending_and_creates_job(monkeypatch):
    # Don't actually launch the agent during the API test.
    launched = {}
    monkeypatch.setattr(app_module, "_launch_agent", lambda tid: launched.setdefault("tid", tid))
    client = TestClient(app_module.app)

    r = client.post("/api/track", json={"request": "watch for sycophancy"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    tid = body["tracker_id"]
    assert cs.get_job(tid)["request"] == "watch for sycophancy"
    assert launched["tid"] == tid


def test_get_track_status_roundtrip(monkeypatch):
    monkeypatch.setattr(app_module, "_launch_agent", lambda tid: None)
    client = TestClient(app_module.app)
    tid = client.post("/api/track", json={"request": "watch hedging"}).json()["tracker_id"]
    r = client.get(f"/api/track/{tid}")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"


def test_get_track_unknown_is_404():
    client = TestClient(app_module.app)
    r = client.get("/api/track/does-not-exist")
    assert r.status_code == 404
