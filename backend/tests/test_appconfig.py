"""WS0 AppConfig backbone — models, loader, secret exclusion."""
from __future__ import annotations


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


def test_appconfig_assembles_and_derives_ids():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.model.layer == 17
    assert cfg.sae.release == "gemma-scope-2-4b-it-res"
    assert cfg.feature_cloud.topk == 15
    assert cfg.runtime.product_name == "GlassBox"
    # secret scalars default to empty (never auto-populated by the model itself)
    assert cfg.anthropic_api_key == ""
    assert cfg.pod_token == ""
    # derived helpers substitute the layer
    assert cfg.sae_id() == "layer_17_width_16k_l0_medium"
    assert cfg.sae_id(22) == "layer_22_width_16k_l0_medium"
    assert cfg.np_source() == "17-gemmascope-2-res-16k"
    assert cfg.np_source(9) == "9-gemmascope-2-res-16k"


def test_load_config_defaults_when_no_yaml(tmp_path):
    from backend.config import load_config

    missing = tmp_path / "nope.yaml"
    cfg = load_config(missing)
    assert cfg.model.model_id == "unsloth/gemma-3-4b-it"
    assert cfg.model.layer == 17
    assert cfg.pod.url == "http://localhost:8001"


def test_load_config_yaml_then_env_override(tmp_path, monkeypatch):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "model:\n  layer: 22\nruntime:\n  product_name: Demo\n", encoding="utf-8"
    )
    cfg = load_config(yaml_path)
    assert cfg.model.layer == 22  # from YAML
    assert cfg.runtime.product_name == "Demo"

    monkeypatch.setenv("GLASSBOX__MODEL__LAYER", "29")
    cfg2 = load_config(yaml_path)
    assert cfg2.model.layer == 29  # env beats YAML


def test_load_config_ignores_secrets_in_yaml(tmp_path, monkeypatch):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "anthropic_api_key: leaked-from-yaml\npod_token: leaked\n", encoding="utf-8"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("POD_TOKEN", raising=False)
    cfg = load_config(yaml_path)
    assert cfg.anthropic_api_key == ""  # YAML secret popped, env empty
    assert cfg.pod_token == ""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    cfg2 = load_config(yaml_path)
    assert cfg2.anthropic_api_key == "sk-from-env"  # env is the only source


def test_load_config_normalizes_pod_url(tmp_path):
    from backend.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("pod:\n  url: http://pod.example/\n", encoding="utf-8")
    cfg = load_config(yaml_path)
    assert cfg.pod.url == "http://pod.example"


def test_example_yaml_loads_and_has_no_secrets():
    from pathlib import Path

    import yaml

    from backend.config import load_config

    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    assert example.exists(), "config.example.yaml must be committed"

    raw = yaml.safe_load(example.read_text())
    for secret in ("anthropic_api_key", "sentry_dsn", "sentry_auth_token", "pod_token", "hf_token"):
        assert secret not in raw, f"{secret} must NOT appear in config.example.yaml"

    cfg = load_config(example)
    assert cfg.runtime.product_name == "GlassBox"
    assert cfg.model.model_id == "unsloth/gemma-3-4b-it"
    # the active default prompt stays neutral; medical lives only in a comment block
    assert "clinical" not in cfg.model.system_prompt.lower()
