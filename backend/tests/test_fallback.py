from backend import fallback
from backend.config import AppConfig


def test_synth_turn_is_deterministic_and_neutral():
    cfg = AppConfig()
    msgs = [{"role": "user", "content": "How do rainbows form?"}]
    a1, f1 = fallback.synth_turn(msgs, cfg.sae, np_source=cfg.np_source())
    a2, f2 = fallback.synth_turn(msgs, cfg.sae, np_source=cfg.np_source())
    assert a1 == a2  # deterministic
    assert "clinical" not in a1.lower() and "patient" not in a1.lower()
    assert len(f1) >= 1
    for feat in f1:
        assert 0 <= feat["index"] < cfg.sae.d_sae
        assert feat["source"] == "17-gemmascope-2-res-16k"
        assert "clinical" not in feat["label"].lower()


def test_is_synthetic_response_detects_template():
    cfg = AppConfig()
    answer, _ = fallback.synth_turn([{"role": "user", "content": "x"}], cfg.sae, np_source=cfg.np_source())
    assert fallback.is_synthetic_response(answer) is True
    assert fallback.is_synthetic_response("totally unrelated text") is False
