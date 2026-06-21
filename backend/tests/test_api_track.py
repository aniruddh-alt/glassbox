from fastapi.testclient import TestClient

from backend import app as app_module
from backend import pod_client


def test_post_track_proxies_request_to_pod(monkeypatch):
    captured = {}

    def fake_track(request):
        captured["request"] = request
        return {"tracker_id": "watch-x-abc123", "status": "pending"}

    monkeypatch.setattr(pod_client, "track", fake_track)
    client = TestClient(app_module.app)

    r = client.post("/api/track", json={"request": "watch for sycophancy"})
    assert r.status_code == 200
    assert r.json() == {"tracker_id": "watch-x-abc123", "status": "pending"}
    assert captured["request"] == "watch for sycophancy"


def test_get_track_status_proxies_to_pod(monkeypatch):
    monkeypatch.setattr(
        pod_client,
        "track_status",
        lambda tid: {"tracker_id": tid, "status": "ready", "auroc": 0.91, "verdict": "good"},
    )
    client = TestClient(app_module.app)

    r = client.get("/api/track/watch-x-abc123")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["auroc"] == 0.91


def test_track_degrades_when_pod_unavailable(monkeypatch):
    def boom(*_a, **_k):
        raise pod_client.PodError(0, "track")

    monkeypatch.setattr(pod_client, "track", boom)
    client = TestClient(app_module.app)

    r = client.post("/api/track", json={"request": "x"})
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


def test_track_status_degrades_when_pod_unavailable(monkeypatch):
    def boom(*_a, **_k):
        raise pod_client.PodError(0, "track_status")

    monkeypatch.setattr(pod_client, "track_status", boom)
    client = TestClient(app_module.app)

    r = client.get("/api/track/whatever")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"
