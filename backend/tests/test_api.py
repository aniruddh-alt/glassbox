import json

from fastapi.testclient import TestClient

from backend import runtime
from backend.app import app
from backend.schema import CognitionEvent

client = TestClient(app)


def _force_fallback():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def test_app_state_config_built_at_startup():
    with TestClient(app) as c:
        assert app.state.config is not None
        assert app.state.config.runtime.product_name == "GlassBox"


def test_health_shape():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        p = r.json()
        assert {"mode", "model", "layer", "trackers"} <= set(p)


def test_chat_streams_tokens_then_one_event():
    with TestClient(app) as c:
        _force_fallback()
        r = c.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "How do rainbows form?"}]},
        )
        assert r.status_code == 200
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) >= 2
        last = json.loads(lines[-1])
        assert "message_id" in last


def test_analyze_returns_event():
    with TestClient(app) as c:
        _force_fallback()
        r = c.post("/api/analyze", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        CognitionEvent(**r.json())


def test_feature_label(monkeypatch):
    from backend import labels

    monkeypatch.setattr(labels, "get_label", lambda i, *a, **k: f"feat-{i}")
    with TestClient(app) as c:
        r = c.get("/api/feature/123")
        assert r.json()["label"] == "feat-123"


# NOTE: tests for the old synth_concept-based /api/track were removed during the merge with
# main — main's /api/track is job-based (create_job → pending → background agent). That flow
# is currently untested (a pre-existing gap on main).
