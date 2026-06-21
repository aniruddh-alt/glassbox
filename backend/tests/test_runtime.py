from backend import runtime


def test_attempt_load_falls_back_when_torch_absent():
    state = runtime._attempt_load(torch_available=False)
    assert state["mode"] == "fallback"
    assert state["model_loaded"] is False
    assert state["sae_loaded"] is False


def test_attempt_load_goes_real_with_injected_loaders():
    state = runtime._attempt_load(
        torch_available=True, load_engine=lambda: None, load_sae=lambda: None
    )
    assert state["mode"] == "real"
    assert state["model_loaded"] is True
    assert state["sae_loaded"] is True


def test_attempt_load_falls_back_on_loader_error():
    def boom():
        raise RuntimeError("no weights")

    state = runtime._attempt_load(torch_available=True, load_engine=boom)
    assert state["mode"] == "fallback"


def test_health_payload_shape():
    p = runtime.health_payload()
    assert set(p) == {"mode", "model_loaded", "sae_loaded", "model", "layer", "trackers"}
    assert p["layer"] == 17
    assert isinstance(p["trackers"], list)
