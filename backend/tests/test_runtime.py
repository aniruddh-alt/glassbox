from backend import runtime
from backend.config import AppConfig


def _cfg(pod_url="", poll=5.0):
    cfg = AppConfig()
    cfg.pod.url = pod_url
    cfg.pod.poll_interval = poll
    return cfg


def test_no_pod_url_stays_fallback():
    runtime.STATE.clear()
    runtime.STATE.update(mode="loading")
    runtime.start_loading(_cfg(pod_url=""))
    assert runtime.STATE["mode"] == "fallback"
    assert runtime.STATE["pod_reachable"] is False


def test_apply_pod_health_real():
    runtime._apply_pod_health(
        {
            "mode": "real",
            "model_loaded": True,
            "sae_loaded": True,
            "d_sae": 16384,
            "trackers": ["uncertainty"],
            "sae_recon_cosine": 0.95,
            "sae_recon_ok": True,
        },
        reachable=True,
    )
    assert runtime.STATE["mode"] == "real"
    assert runtime.STATE["model_loaded"] is True
    assert runtime.STATE["pod_reachable"] is True


def test_apply_pod_health_unreachable():
    runtime._apply_pod_health(None, reachable=False)
    assert runtime.STATE["mode"] == "fallback"
    assert runtime.STATE["pod_reachable"] is False


def test_poll_pod_once_uses_client(monkeypatch):
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    state = runtime._poll_pod_once(_cfg(pod_url="http://pod.test"))
    assert state["mode"] == "real"
    assert state["pod_reachable"] is True


def test_poll_pod_once_falls_back_on_error(monkeypatch):
    import backend.pod_client as pc

    def boom(pod, pod_token):
        raise pc.PodError(0, "health")

    monkeypatch.setattr(pc, "health", boom)
    state = runtime._poll_pod_once(_cfg(pod_url="http://pod.test"))
    assert state["mode"] == "fallback"
    assert state["pod_reachable"] is False


def test_health_payload_shape():
    runtime.STATE.update(
        mode="fallback",
        model_loaded=False,
        sae_loaded=False,
        pod_reachable=False,
        pod_health=None,
    )
    p = runtime.health_payload(_cfg(pod_url="http://pod.test"))
    assert {
        "mode",
        "model_loaded",
        "sae_loaded",
        "model",
        "layer",
        "d_sae",
        "trackers",
        "sae_recon_cosine",
        "sae_recon_ok",
        "pod_reachable",
        "pod_url_configured",
    } <= set(p)
    assert p["layer"] == 17
    assert isinstance(p["trackers"], list)


def test_start_loading_eager_polls_pod(monkeypatch):
    monkeypatch.setenv("GLASSBOX_EAGER_LOAD", "1")
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: None, raising=False)
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert runtime.STATE["mode"] == "real"


def test_eager_load_also_keeps_polling(monkeypatch):
    monkeypatch.setenv("GLASSBOX_EAGER_LOAD", "1")
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda pod, pod_token: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    started: list[bool] = []
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: started.append(True), raising=False)
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert runtime.STATE["mode"] == "real"
    assert started == [True]


def test_non_eager_starts_poll_loop(monkeypatch):
    monkeypatch.delenv("GLASSBOX_EAGER_LOAD", raising=False)
    started: list[bool] = []
    monkeypatch.setattr(runtime, "_ensure_poll_loop", lambda cfg: started.append(True), raising=False)
    runtime.start_loading(_cfg(pod_url="http://pod.test"))
    assert started == [True]
