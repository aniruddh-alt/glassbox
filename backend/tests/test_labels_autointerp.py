"""Auto-interp label coverage: the window builder (pure) and get_feature_stats' resolution order
(Neuronpedia explanation -> Claude auto-interp -> 'feature N'), with httpx + Claude mocked so no
network or API call happens."""

import backend.labels as labels
from backend.config import AppConfig


def test_highlighted_windows_wraps_peak_and_trims():
    acts = [{"tokens": ["a", "b", "PEAK", "c", "d"], "values": [0, 0, 5, 0, 0], "maxValueTokenIndex": 2}]
    assert labels._highlighted_windows(acts, n=15, radius=1) == ["b<<PEAK>>c"]


def test_highlighted_windows_infers_peak_when_index_missing():
    acts = [{"tokens": ["x", "HOT", "y"], "values": [1, 9, 2]}]  # no maxValueTokenIndex
    assert labels._highlighted_windows(acts, n=15, radius=5) == ["x<<HOT>>y"]


def test_highlighted_windows_skips_empty_and_caps_n():
    acts = [{"tokens": [], "values": []}, {"tokens": ["p"], "values": [1], "maxValueTokenIndex": 0}]
    assert labels._highlighted_windows(acts, n=1, radius=2) == []  # n=1 takes only the empty one


def _fake_resp(payload):
    class R:
        status_code = 200

        def json(self):
            return payload

    return R()


def _isolate(monkeypatch, tmp_path):
    labels._stats.clear()
    monkeypatch.setattr(labels, "_disk", {})
    monkeypatch.setattr(labels, "_CACHE_PATH", str(tmp_path / "cache.json"))


def _default_cfg():
    cfg = AppConfig()
    cfg.anthropic_api_key = "sk-test"
    return cfg


def test_get_feature_stats_autointerps_when_unlabeled(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cfg = _default_cfg()
    payload = {
        "explanations": [],
        "maxActApprox": 12.0,
        "frac_nonzero": 0.05,
        "activations": [{"tokens": ["a", "PEAK", "b"], "values": [0, 5, 0], "maxValueTokenIndex": 1}],
    }
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _fake_resp(payload))
    monkeypatch.setattr(
        labels, "_autointerp",
        lambda acts, fc, key: {"label": "endocrine hormones", "is_structural": False},
    )
    s = labels.get_feature_stats(897, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())
    assert s["label"] == "endocrine hormones"
    assert s["is_structural"] is False
    assert s["max_act"] == 12.0 and s["density"] == 0.05
    assert labels._disk["897"]["label"] == "endocrine hormones"  # resolved label persisted


def test_get_feature_stats_marks_structural_from_autointerp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cfg = _default_cfg()
    payload = {"explanations": [], "maxActApprox": None, "frac_nonzero": 0.1,
               "activations": [{"tokens": ["x", "y"], "values": [0, 9], "maxValueTokenIndex": 1}]}
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _fake_resp(payload))
    monkeypatch.setattr(
        labels, "_autointerp",
        lambda acts, fc, key: {"label": "paragraph breaks", "is_structural": True},
    )
    s = labels.get_feature_stats(788, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())
    assert s["is_structural"] is True


def test_get_feature_stats_uses_neuronpedia_and_skips_autointerp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cfg = _default_cfg()
    payload = {
        "explanations": [{"description": "pregnancy and childbirth"}],
        "maxActApprox": 700.0, "frac_nonzero": 0.001,
        "activations": [{"tokens": ["x"], "values": [1], "maxValueTokenIndex": 0}],
    }
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _fake_resp(payload))

    def boom(acts, fc, key):
        raise AssertionError("auto-interp must not run when Neuronpedia already has a label")

    monkeypatch.setattr(labels, "_autointerp", boom)
    s = labels.get_feature_stats(11270, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())
    assert s["label"] == "pregnancy and childbirth"
    assert s["is_structural"] is False


def test_get_feature_stats_fallback_not_persisted(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cfg = _default_cfg()

    def fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(labels.httpx, "get", fail)
    s = labels.get_feature_stats(999, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())
    assert s["label"] == "feature 999"
    assert s["is_structural"] is False
    assert "999" not in labels._disk  # transient failure must be retryable next run


def test_get_feature_stats_retries_unresolved_session_fallback(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cfg = _default_cfg()
    calls = {"n": 0}

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary network failure")
        return _fake_resp(
            {
                "explanations": [{"description": "pregnancy and childbirth"}],
                "maxActApprox": 700.0,
                "frac_nonzero": 0.001,
                "activations": [],
            }
        )

    monkeypatch.setattr(labels.httpx, "get", flaky_get)

    assert labels.get_feature_stats(999, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())["label"] == "feature 999"
    assert labels.get_feature_stats(999, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source())["label"] == "pregnancy and childbirth"
    assert calls["n"] == 2


class _NPResp:
    """Neuronpedia response with NO explanation but WITH activating examples -> autointerp fires."""

    status_code = 200

    def json(self):
        return {
            "explanations": [],
            "maxActApprox": 8.0,
            "frac_nonzero": 0.001,
            "activations": [{"tokens": ["the", "cat"], "values": [0.0, 5.0], "maxValueTokenIndex": 1}],
        }


def test_autointerp_prompt_is_neutral(monkeypatch):
    """The auto-interp Claude prompt must not describe the model as a medical chatbot."""
    captured = {}

    class FakeMsg:
        # one tool_use block, mirroring the record_feature_label tool contract
        content = [type("B", (), {"type": "tool_use", "name": "record_feature_label",
                                  "input": {"label": "x", "is_structural": False}})()]

    class FakeMessages:
        def create(self, **kw):
            captured["system"] = kw.get("system", "")
            return FakeMsg()

    class FakeClient:
        def __init__(self, **kw):
            self.messages = FakeMessages()

    labels._stats.clear()  # force the network/autointerp branch (skip the session cache)
    monkeypatch.setattr(labels, "_disk_cache", lambda: {})  # skip the disk cache too
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _NPResp())
    monkeypatch.setattr(labels, "_anthropic_client", lambda key: FakeClient(), raising=False)

    cfg = AppConfig()
    cfg.anthropic_api_key = "sk-test"
    out = labels.get_label(
        4242, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()
    )
    assert isinstance(out, str)
    assert "medical" not in captured.get("system", "").lower()
    assert "clinical" not in captured.get("system", "").lower()


def test_get_label_no_autointerp_without_key(monkeypatch):
    """No Claude key -> autointerp returns None; label degrades to 'feature N' (never raises)."""
    labels._stats.clear()
    monkeypatch.setattr(labels, "_disk_cache", lambda: {})
    monkeypatch.setattr(labels.httpx, "get", lambda *a, **k: _NPResp())
    cfg = AppConfig()  # anthropic_api_key == ""
    out = labels.get_label(
        4343, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()
    )
    assert out == "feature 4343"  # degraded fallback, no Claude call
