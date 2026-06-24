"""Family-A feature source with graceful degradation."""

from __future__ import annotations

import os

from ..config import SAEConfig, FeatureCloudConfig, ModelConfig
from .sae import attribution_topk, sae_topk


def _resolve_device_inline(device: str | None = None) -> str:
    """Prefer cuda > mps > cpu; returns `device` directly if given."""
    if device is not None:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class FeatureProvider:
    name = "base"
    reliable = (
        False  # True only when the local path (which also powers Family B) is available
    )

    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        grad=None,
        baseline=None,
        cap: int | None = None,
        sae: SAEConfig | None = None,
        feature_cloud: FeatureCloudConfig | None = None,
        model: ModelConfig | None = None,
        np_source: str | None = None,
    ) -> list[dict]:
        """Return up to `cap` deduped, ranked candidate {index, act, source[, attr]} dicts.
        token_ids: per-position ids aligned to `activations`. special_ids: ids to skip.
        grad: per-position dL/d(resid_post) aligned to `activations`, enabling attribution
        ranking. baseline: optional [d_sae] neutral-prompt attribution to subtract (contrastive).
        Labels are attached downstream."""
        raise NotImplementedError


class LocalSAEProvider(FeatureProvider):
    """Uses the layer-17 resid_post activations captured during local generation. Free + fast.
    Prefers attribution ranking (needs `grad`); falls back to raw-activation max-pool."""

    name = "local"

    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        grad=None,
        baseline=None,
        cap: int | None = None,
        sae: SAEConfig | None = None,
        feature_cloud: FeatureCloudConfig | None = None,
        model: ModelConfig | None = None,
        np_source: str | None = None,
    ) -> list[dict]:
        if activations is None:
            raise ValueError(
                "LocalSAEProvider needs captured activations [n_positions, d_in]"
            )
        _fc = feature_cloud if feature_cloud is not None else FeatureCloudConfig()
        _model = model if model is not None else ModelConfig()
        _sae = sae if sae is not None else SAEConfig()
        _np_source = np_source if np_source is not None else _sae.np_source_pattern.format(layer=_model.layer)

        topk = _fc.topk
        topk_event = _fc.topk_event
        rank_method = _fc.rank_method
        preamble_skip = _model.preamble_skip

        effective_cap = cap if cap is not None else topk_event

        special = set(special_ids or [])
        keep = [
            pos
            for pos in range(activations.shape[0])
            if not (token_ids is not None and pos < len(token_ids) and token_ids[pos] in special)
        ]
        # Preferred: attribution candidate pool (causal effect on the response), optionally
        # contrastive (baseline subtracted). Replaces the raw-activation pool that structurally
        # over-selects high-norm grammatical features. Drop the formulaic preamble positions so
        # opening discourse features ("Okay,", greetings) don't dominate the attribution sum.
        if grad is not None and rank_method == "attribution":
            keep_c = [p for p in keep if p >= preamble_skip] or keep
            try:
                return attribution_topk(
                    activations, grad, keep_c,
                    _sae, _fc,
                    np_source=_np_source,
                    cap=effective_cap,
                    baseline=baseline,
                )
            except Exception as e:  # noqa: BLE001 — degrade to activation ranking, never crash
                print(f"[provider] attribution_topk failed ({e}); using activation ranking")
        # Fallback: per-token top-k, max activation per feature across kept positions.
        best: dict[int, float] = {}
        for pos in keep:
            for f in sae_topk(activations[pos], _sae, _fc, np_source=_np_source, k=topk):
                best[f["index"]] = max(best.get(f["index"], 0.0), f["act"])
        top = sorted(best.items(), key=lambda kv: -kv[1])[:effective_cap]
        return [
            {"index": i, "act": round(v, 3), "source": _np_source} for i, v in top
        ]


class NeuronpediaProvider(FeatureProvider):
    """Fallback for when the local SAE provider is not available."""

    name = "neuronpedia"
    BASE = "https://www.neuronpedia.org"

    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        grad=None,
        baseline=None,
        cap: int | None = None,
        sae: SAEConfig | None = None,
        feature_cloud: FeatureCloudConfig | None = None,
        model: ModelConfig | None = None,
        np_source: str | None = None,
    ) -> list[dict]:
        if not os.getenv("GLASSBOX_ALLOW_REMOTE_FEATURES"):
            raise RuntimeError("remote feature POST forbidden for patient data")
        import httpx

        _fc = feature_cloud if feature_cloud is not None else FeatureCloudConfig()
        _model = model if model is not None else ModelConfig()
        _sae = sae if sae is not None else SAEConfig()
        _np_source = np_source if np_source is not None else _sae.np_source_pattern.format(layer=_model.layer)

        topk = _fc.topk
        topk_event = _fc.topk_event
        effective_cap = cap if cap is not None else topk_event
        np_model = _sae.np_model

        url = f"{self.BASE}/api/activation/topk-by-token"
        payload = {
            "modelId": np_model,
            "source": _np_source,
            "text": text,
            "topK": topk,
        }
        best: dict[int, float] = {}
        try:
            r = httpx.post(
                url, json=payload, headers={"User-Agent": "glassbox/0.1"}, timeout=20
            )
            if r.status_code == 200:
                for tokres in r.json().get("results") or []:
                    for f in tokres.get("topFeatures") or tokres.get("features") or []:
                        idx = f.get("index", f.get("featureIndex"))
                        act = float(f.get("activation", f.get("act", 0.0)))
                        if idx is not None:
                            best[int(idx)] = max(best.get(int(idx), 0.0), act)
        except Exception:
            return []  # degrade to no cloud rather than crash the demo
        top = sorted(best.items(), key=lambda kv: -kv[1])[:effective_cap]
        return [
            {"index": i, "act": round(v, 3), "source": _np_source} for i, v in top
        ]


def get_provider(prefer: str | None = None, *, device: str | None = None) -> FeatureProvider:
    """Pick the feature provider. 'auto' (default) = local when a GPU/MPS is present, else Neuronpedia.
    Override with FEATURE_PROVIDER=local|neuronpedia."""
    pref = (prefer or os.getenv("FEATURE_PROVIDER", "auto")).lower()
    if pref == "neuronpedia":
        return NeuronpediaProvider()
    if pref == "local":
        return LocalSAEProvider()
    try:
        dev = _resolve_device_inline(device)
        if dev != "cpu":
            return LocalSAEProvider()
    except Exception:
        pass
    return NeuronpediaProvider()
