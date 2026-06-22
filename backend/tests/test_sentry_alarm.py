import backend.fanout as fo


def test_capture_cognition_alarm_requires_flag(monkeypatch):
    monkeypatch.setattr(fo, "_sentry_on", True)
    captured = {}

    fake = type("S", (), {})()
    fake.capture_message = lambda msg, level=None: captured.setdefault("msg", msg)
    fake.flush = lambda timeout=2: None

    class Scope:
        def __enter__(s):
            return s

        def __exit__(s, *a):
            return False

        fingerprint = None

        def set_tag(s, *a):
            pass

        def set_context(s, k, v):
            captured["ctx"] = v

    fake.new_scope = lambda: Scope()
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", fake)

    assert fo.capture_cognition_alarm({"flag": False}) is False
    assert "msg" not in captured
    assert fo.capture_cognition_alarm(
        {
            "flag": True,
            "severity": "warning",
            "model": "test",
            "message_id": "m1",
            "uncertainty": 0.9,
            "trackers": {"over_confidence": {"score": 0.9, "flag": True}},
            "features": [{"label": "dosing"}],
        },
        None,
        flush=True,
    ) is True
    assert captured["msg"].startswith("Confident-wrong medical answer")
    assert "over_confidence" in captured["msg"]


def test_replay_sentry_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend import observability

    store = observability.ObservabilityStore()
    monkeypatch.setattr(observability, "STORE", store)
    store.record(
        {
            "message_id": "m-flag",
            "ts": 1.0,
            "flag": True,
            "severity": "warning",
            "trackers": {"harmful_prompt": {"score": 0.91, "flag": True}},
            "features": [{"index": 1, "label": "crisis"}],
        },
        None,
    )
    monkeypatch.setattr("backend.app.sentry_enabled", lambda: True)
    captured = {}

    def _capture(ev, obs=None, *, flush=False):
        captured["reason"] = __import__("backend.fanout", fromlist=["_flag_reason"])._flag_reason(ev)
        return True

    monkeypatch.setattr("backend.app.capture_cognition_alarm", _capture)
    r = TestClient(app).post("/api/observability/replay-sentry")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert captured["reason"] == "harmful_prompt"


def test_test_sentry_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import app

    monkeypatch.setattr("backend.app.sentry_enabled", lambda: False)
    r = TestClient(app).post("/api/observability/test-sentry")
    assert r.status_code == 503

    monkeypatch.setattr("backend.app.sentry_enabled", lambda: True)
    monkeypatch.setattr("backend.app.capture_cognition_alarm", lambda ev, obs=None, *, flush=False: True)
    r = TestClient(app).post("/api/observability/test-sentry")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["message_id"].startswith("test-")
