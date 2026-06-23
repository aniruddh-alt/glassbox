import pytest
import torch
from fastapi.testclient import TestClient

from backend import gpu_service
from backend.config import AppConfig


@pytest.fixture(autouse=True)
def _state_config():
    """Every test gets a fresh default AppConfig on app.state (gpu_service handlers read
    request.app.state.config). The startup hook also builds one, but tests instantiate
    TestClient lazily — set it here so even pre-startup reads succeed."""
    gpu_service.app.state.config = AppConfig()
    yield


@pytest.fixture
def _skip_pod_load(monkeypatch):
    monkeypatch.setattr(
        gpu_service,
        "_attempt_load",
        lambda cfg: gpu_service.STATE.update(mode="real", model_loaded=True, sae_loaded=True),
    )
    # Pod auth is opt-in for tests: clear pod_token so endpoints don't 401 on the ambient
    # .env/default token. test_auth_rejects_bad_token sets its own token to exercise auth.
    gpu_service.app.state.config.pod_token = ""


def _fake_capture(messages, max_new, *, attribution=None, cfg=None):
    tok = type("T", (), {"convert_tokens_to_ids": staticmethod(lambda t: hash(t) % 1000)})()
    acts = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    out_ids = torch.tensor([10, 11, 12, 13])
    return {
        "answer": "fake answer",
        "acts": acts,
        "grad": torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.3, 0.0]]),
        "out_ids": out_ids,
        "resp_start": 1,
        "tok": tok,
    }


def _install_stubs(monkeypatch):
    gpu_service.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(gpu_service, "_capture", _fake_capture)
    monkeypatch.setattr(
        gpu_service,
        "_sae_candidates",
        lambda res, resp_acts, resp_grad, *, cap, baseline=None, cfg=None: [
            {"index": 42, "act": 1.0, "attr": 0.5, "source": "s"}
        ],
    )
    monkeypatch.setattr(gpu_service, "_score_trackers", lambda a, b: {
        "harmful": {"score": 0.1, "flag": False},
        "over_confidence": {"score": 0.2, "flag": False},
    })
    # _baseline_vec is gated on mode==real + contrast_baseline; stub to None so no engine/sae load.
    monkeypatch.setattr(gpu_service, "_baseline_vec", lambda cfg: None)


def test_health_shape(_skip_pod_load):
    client = TestClient(gpu_service.app)
    r = client.get("/health")
    assert r.status_code == 200
    p = r.json()
    assert {"mode", "model_loaded", "sae_loaded", "model", "layer", "d_sae", "trackers"} <= set(p)


def test_inference_returns_answer(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    client = TestClient(gpu_service.app)
    r = client.post("/inference", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json() == {"answer": "fake answer"}


def test_activations_returns_pooled_vectors(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    client = TestClient(gpu_service.app)
    r = client.post("/activations", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "fake answer"
    assert body["resp_start"] == 1
    assert len(body["act_last"]) == 2
    assert len(body["act_resp"]) == 2


def test_sae_features_returns_candidates(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    client = TestClient(gpu_service.app)
    r = client.post("/sae/features", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["reliable"] is True
    assert body["candidates"][0]["index"] == 42


def test_turn_composes_candidates_and_trackers(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "fake answer"
    assert body["candidates"][0]["index"] == 42
    assert "harmful" in body["trackers"]
    assert "over_confidence" in body["trackers"]
    assert body["reliable"] is True


def test_auth_rejects_bad_token(monkeypatch, _skip_pod_load):
    _install_stubs(monkeypatch)
    gpu_service.app.state.config.pod_token = "secret"
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": []})
    assert r.status_code == 401
    r = client.post(
        "/turn",
        json={"messages": []},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200


def test_turn_returns_timings(monkeypatch, _skip_pod_load):
    """Contract test: /turn response must include additive 'timings' key with capture/sae/trackers floats."""
    _install_stubs(monkeypatch)
    client = TestClient(gpu_service.app)
    r = client.post("/turn", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert "timings" in body, "timings key missing from /turn response"
    t = body["timings"]
    for key in ("capture", "sae", "trackers"):
        assert key in t, f"timings.{key} missing"
        assert isinstance(t[key], float) and t[key] >= 0.0, f"timings.{key} must be float >= 0"
