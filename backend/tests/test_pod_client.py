import httpx

import backend.pod_client as pc
from backend.config import PodConfig


def _pod(url="http://pod.test"):
    return PodConfig(url=url)


def test_health_returns_none_without_pod_url():
    assert pc.health(_pod(url=""), "") is None


def test_turn_posts_messages(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"answer": "ok", "candidates": [], "trackers": {}, "reliable": True}

    def fake_post(url, **kw):
        captured["url"] = url
        captured.update(kw)
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    out = pc.turn([{"role": "user", "content": "hi"}], _pod(), "secret", max_new=512)
    assert out["answer"] == "ok"
    assert captured["url"] == "http://pod.test/turn"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["max_new"] == 512


def test_sae_features_path(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"answer": "a", "candidates": [], "reliable": True}

    monkeypatch.setattr(httpx, "post", lambda url, **kw: captured.update({"url": url, **kw}) or FakeResp())
    pc.sae_features([{"role": "user", "content": "x"}], _pod(), "", max_new=64, cap=20)
    assert captured["url"] == "http://pod.test/sae/features"
    assert captured["json"]["cap"] == 20


def test_post_raises_pod_error_on_non_200(monkeypatch):
    class FakeResp:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    try:
        pc.inference([{"role": "user", "content": "x"}], _pod(), "", max_new=512)
        assert False, "expected PodError"
    except pc.PodError as e:
        assert e.status == 500


def test_health_raises_on_failure(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    try:
        pc.health(_pod(), "")
        assert False, "expected PodError"
    except pc.PodError as e:
        assert e.status == 0


def test_pod_error_carries_no_body():
    e = pc.PodError(500, "turn")
    assert "turn" in str(e) and "500" in str(e)


def test_post_non_200_body_not_in_exception(monkeypatch):
    class FakeResp:
        status_code = 503
        text = "secret model answer that must not leak"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    try:
        pc.turn([{"role": "user", "content": "x"}], _pod(), "", max_new=512)
        assert False
    except pc.PodError as e:
        assert "secret model answer" not in str(e)
