import pytest

from backend import config, pod_client
from backend.pod_client import PodError


def test_health_returns_none_without_pod_url(monkeypatch):
    monkeypatch.setattr(config, "POD_URL", "")
    assert pod_client.health() is None


def test_turn_posts_messages(monkeypatch):
    monkeypatch.setattr(config, "POD_URL", "http://pod.test")
    monkeypatch.setattr(config, "POD_TOKEN", "secret")
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "answer": "yes",
                "candidates": [{"index": 1, "act": 0.5, "source": "s"}],
                "trackers": {},
                "reliable": True,
            }

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    out = pod_client.turn([{"role": "user", "content": "hi"}], max_new=32)
    assert captured["url"] == "http://pod.test/turn"
    assert captured["json"]["max_new"] == 32
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert out["answer"] == "yes"
    assert out["candidates"][0]["index"] == 1


def test_sae_features_path(monkeypatch):
    monkeypatch.setattr(config, "POD_URL", "http://pod.test")
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"answer": "a", "candidates": [], "reliable": True}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda url, **kw: captured.update({"url": url, **kw}) or FakeResp())
    pod_client.sae_features([{"role": "user", "content": "x"}], attribution=True, cap=10)
    assert captured["url"] == "http://pod.test/sae/features"
    assert captured["json"]["attribution"] is True
    assert captured["json"]["cap"] == 10


def test_post_raises_pod_error_on_non_200(monkeypatch):
    monkeypatch.setattr(config, "POD_URL", "http://pod.test")

    class FakeResp:
        status_code = 503
        text = "busy"

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    with pytest.raises(PodError, match="503"):
        pod_client.inference([{"role": "user", "content": "hi"}])


def test_health_raises_on_failure(monkeypatch):
    monkeypatch.setattr(config, "POD_URL", "http://pod.test")
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(PodError):
        pod_client.health()


# Task 9 — privacy: PodError must carry ONLY status code + stage label, never the response body.
def test_pod_error_carries_no_body():
    from backend.pod_client import PodError
    e = PodError(500, "turn")
    s = str(e)
    assert "500" in s and "turn" in s
    # simulate a body that echoes the answer — it must NOT be in the exception
    assert "RESPONSE_BODY" not in s


def test_pod_error_str_format():
    """str(PodError(503, 'inference')) == 'pod inference failed: HTTP 503'"""
    from backend.pod_client import PodError
    e = PodError(503, "inference")
    assert str(e) == "pod inference failed: HTTP 503"


def test_post_non_200_body_not_in_exception(monkeypatch):
    """_post must not embed r.text in PodError — only status + stage."""
    monkeypatch.setattr(config, "POD_URL", "http://pod.test")

    class FakeResp:
        status_code = 500
        text = "RESPONSE_BODY patient answer here"

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    with pytest.raises(PodError) as exc_info:
        pod_client.inference([{"role": "user", "content": "hi"}])
    s = str(exc_info.value)
    assert "500" in s
    assert "RESPONSE_BODY" not in s
    assert "patient answer here" not in s
