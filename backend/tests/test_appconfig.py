"""WS0 AppConfig backbone — models, loader, secret exclusion."""
from __future__ import annotations

from pathlib import Path


def test_submodels_have_gemma_defaults():
    from backend.config import ModelConfig, SAEConfig, FeatureCloudConfig, ProbeConfig

    m = ModelConfig()
    assert m.model_id == "unsloth/gemma-3-4b-it"
    assert m.layer == 17
    assert m.sae_layers == [9, 17, 22, 29]
    assert m.mask_tokens == ["<bos>", "<start_of_turn>", "<end_of_turn>"]
    assert m.preamble_skip == 12
    assert m.max_new_tokens == 512

    s = SAEConfig()
    assert s.release == "gemma-scope-2-4b-it-res"
    assert s.sae_id_pattern == "layer_{layer}_width_16k_l0_medium"
    assert s.d_in == 2560
    assert s.d_sae == 16384
    assert s.np_model == "gemma-3-4b-it"
    assert s.np_source_pattern == "{layer}-gemmascope-2-res-16k"

    fc = FeatureCloudConfig()
    assert fc.topk == 15
    assert fc.topk_event == 30
    assert fc.topk_candidates == 50
    assert fc.rank_method == "attribution"

    p = ProbeConfig()
    assert p.enabled == ["harmful", "harmful_prompt", "over_confidence"]
    assert p.disabled == ["uncertainty", "hallucination", "risk_awareness"]
    assert p.builder.auroc_threshold == 0.75


def test_default_system_prompt_is_neutral_not_medical():
    from backend.config import ModelConfig

    sp = ModelConfig().system_prompt.lower()
    assert "helpful" in sp and "honest" in sp
    assert "clinical" not in sp and "medical" not in sp and "patient" not in sp


def test_recon_probe_default_is_neutral():
    from backend.config import SAEConfig

    assert "ibuprofen" not in SAEConfig().recon_probe.lower()
    assert "pregnancy" not in SAEConfig().recon_probe.lower()


def test_secrets_read_from_env_only(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("POD_TOKEN", "pod-secret")
    from backend.config import Secrets

    s = Secrets()
    assert s.anthropic_api_key == "sk-test-123"
    assert s.pod_token == "pod-secret"
