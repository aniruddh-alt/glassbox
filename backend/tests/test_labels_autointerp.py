"""Auto-interp label coverage: the window builder (pure) and get_feature_stats' resolution order
(Neuronpedia explanation -> Claude auto-interp -> 'feature N'), with httpx + Claude mocked so no
network or API call happens."""

import httpx

from backend import labels


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


def test_get_feature_stats_autointerps_when_unlabeled(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(labels.config, "AUTOINTERP", True)
    payload = {
        "explanations": [],
        "maxActApprox": 12.0,
        "frac_nonzero": 0.05,
        "activations": [{"tokens": ["a", "PEAK", "b"], "values": [0, 5, 0], "maxValueTokenIndex": 1}],
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(payload))
    monkeypatch.setattr(labels, "_autointerp", lambda acts: {"label": "endocrine hormones", "is_structural": False})
    s = labels.get_feature_stats(897)
    assert s["label"] == "endocrine hormones"
    assert s["is_structural"] is False
    assert s["max_act"] == 12.0 and s["density"] == 0.05
    assert labels._disk["897"]["label"] == "endocrine hormones"  # resolved label persisted


def test_get_feature_stats_marks_structural_from_autointerp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(labels.config, "AUTOINTERP", True)
    payload = {"explanations": [], "maxActApprox": None, "frac_nonzero": 0.1,
               "activations": [{"tokens": ["x", "y"], "values": [0, 9], "maxValueTokenIndex": 1}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(payload))
    monkeypatch.setattr(labels, "_autointerp", lambda acts: {"label": "paragraph breaks", "is_structural": True})
    s = labels.get_feature_stats(788)
    assert s["is_structural"] is True


def test_get_feature_stats_uses_neuronpedia_and_skips_autointerp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    payload = {
        "explanations": [{"description": "pregnancy and childbirth"}],
        "maxActApprox": 700.0, "frac_nonzero": 0.001,
        "activations": [{"tokens": ["x"], "values": [1], "maxValueTokenIndex": 0}],
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(payload))

    def boom(acts):
        raise AssertionError("auto-interp must not run when Neuronpedia already has a label")

    monkeypatch.setattr(labels, "_autointerp", boom)
    s = labels.get_feature_stats(11270)
    assert s["label"] == "pregnancy and childbirth"
    assert s["is_structural"] is False


def test_get_feature_stats_fallback_not_persisted(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(labels.config, "AUTOINTERP", True)

    def fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(httpx, "get", fail)
    s = labels.get_feature_stats(999)
    assert s["label"] == "feature 999"
    assert s["is_structural"] is False
    assert "999" not in labels._disk  # transient failure must be retryable next run


def test_get_feature_stats_retries_unresolved_session_fallback(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
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

    monkeypatch.setattr(httpx, "get", flaky_get)

    assert labels.get_feature_stats(999)["label"] == "feature 999"
    assert labels.get_feature_stats(999)["label"] == "pregnancy and childbirth"
    assert calls["n"] == 2
