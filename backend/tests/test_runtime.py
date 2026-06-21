from backend import runtime


def test_no_pod_url_stays_fallback(monkeypatch):
    monkeypatch.setattr(runtime.config, "POD_URL", "")
    runtime.STATE.clear()
    runtime.STATE.update(mode="loading")
    runtime.start_loading()
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
    monkeypatch.setattr(runtime.config, "POD_URL", "http://pod.test")
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "health", lambda: {"mode": "real", "model_loaded": True, "sae_loaded": True})
    state = runtime._poll_pod_once()
    assert state["mode"] == "real"
    assert state["pod_reachable"] is True


def test_poll_pod_once_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(runtime.config, "POD_URL", "http://pod.test")
    import backend.pod_client as pc

    def boom():
        raise pc.PodError("down")

    monkeypatch.setattr(pc, "health", boom)
    state = runtime._poll_pod_once()
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
    p = runtime.health_payload()
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
    monkeypatch.setattr(runtime.config, "POD_URL", "http://pod.test")
    import backend.pod_client as pc

    monkeypatch.setattr(
        pc,
        "health",
        lambda: {"mode": "real", "model_loaded": True, "sae_loaded": True},
    )
    runtime.start_loading()
    assert runtime.STATE["mode"] == "real"
