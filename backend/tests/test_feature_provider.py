"""Tests for feature_provider privacy guard.

Note: This file does NOT import torch at module level. The NeuronpediaProvider guard
test must work without torch installed (it's blocked before any network/torch code runs).
"""
import pytest


def test_neuronpedia_provider_forbidden_by_default(monkeypatch):
    monkeypatch.delenv("GLASSBOX_ALLOW_REMOTE_FEATURES", raising=False)
    from backend.science.feature_provider import NeuronpediaProvider
    with pytest.raises(RuntimeError, match="forbidden"):
        NeuronpediaProvider().features_for(["msg"], cap=5)


def test_neuronpedia_provider_allowed_when_env_set(monkeypatch):
    """When GLASSBOX_ALLOW_REMOTE_FEATURES is set, the guard should not raise
    (it may fail for other reasons, but not the forbidden guard)."""
    monkeypatch.setenv("GLASSBOX_ALLOW_REMOTE_FEATURES", "1")
    from backend.science.feature_provider import NeuronpediaProvider

    # We expect no RuntimeError("forbidden"). It may raise something else (network, etc.)
    # or succeed. We use a monkeypatched httpx to keep it fast.
    import importlib
    import backend.science.feature_provider as fp
    importlib.reload(fp)  # pick up env change if cached at import time

    called = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"results": []}

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (called.update({"called": True}) or FakeResp()))

    provider = fp.NeuronpediaProvider()
    result = provider.features_for(["test message"], cap=5)
    assert called.get("called"), "httpx.post should have been called when env var is set"
    assert isinstance(result, list)
