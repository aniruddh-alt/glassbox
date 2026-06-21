"""Family-A feature source with graceful degradation.

PRIMARY  = LocalSAEProvider  — uses activations you already captured locally (free, fast, GPU).
FALLBACK = NeuronpediaProvider — post-hoc API for a no-GPU teammate / resilience (rate-limited).

IMPORTANT: this only covers Family A (the exploratory feature cloud). Family B (the reliable
uncertainty/safety probes) needs RAW residual activations and is LOCAL-ONLY — there is no
Neuronpedia fallback for it. When running on the fallback, mark the event reliable_signal=False.

OWNER: Lane B (local) + Lane A (provider selection / Neuronpedia HTTP).
"""

from __future__ import annotations

import os

from .. import config
from .sae import sae_topk


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
        k: int = config.TOPK,
        cap: int = config.TOPK_EVENT,
    ) -> list[dict]:
        """Return up to `cap` deduped, activation-ranked {index, act, source} dicts for a response.
        Labels are attached downstream by labels.py."""
        raise NotImplementedError


class LocalSAEProvider(FeatureProvider):
    """Uses the layer-12 activations already captured during local generation. Free + fast."""

    name = "local"
    reliable = True

    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        k: int = config.TOPK,
        cap: int = config.TOPK_EVENT,
    ) -> list[dict]:
        if activations is None:
            raise ValueError(
                "LocalSAEProvider needs captured activations [n_positions, d_in]"
            )
        special = set(
            token_ids or []
        )  # caller passes special-token ids to skip, if any
        best: dict[int, float] = {}
        for pos in range(activations.shape[0]):
            if token_ids is not None and token_ids[pos] in special:
                continue
            for f in sae_topk(activations[pos], k=k):
                best[f["index"]] = max(best.get(f["index"], 0.0), f["act"])
        top = sorted(best.items(), key=lambda kv: -kv[1])[:cap]
        return [
            {"index": i, "act": round(v, 3), "source": config.NP_SOURCE} for i, v in top
        ]


class NeuronpediaProvider(FeatureProvider):
    """Fallback for when the local SAE provider is not available."""

    name = "neuronpedia"
    reliable = False
    BASE = "https://www.neuronpedia.org"

    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        k: int = config.TOPK,
        cap: int = config.TOPK_EVENT,
    ) -> list[dict]:
        import httpx

        url = f"{self.BASE}/api/activation/topk-by-token"
        payload = {
            "modelId": config.NP_MODEL,
            "source": config.NP_SOURCE,
            "text": text,
            "topK": k,
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
        top = sorted(best.items(), key=lambda kv: -kv[1])[:cap]
        return [
            {"index": i, "act": round(v, 3), "source": config.NP_SOURCE} for i, v in top
        ]


def get_provider(prefer: str | None = None) -> FeatureProvider:
    """Pick the feature provider. 'auto' (default) = local when a GPU/MPS is present, else Neuronpedia.
    Override with FEATURE_PROVIDER=local|neuronpedia."""
    pref = (prefer or os.getenv("FEATURE_PROVIDER", "auto")).lower()
    if pref == "neuronpedia":
        return NeuronpediaProvider()
    if pref == "local":
        return LocalSAEProvider()
    try:
        if config.resolve_device() != "cpu":
            return LocalSAEProvider()
    except Exception:
        pass
    return NeuronpediaProvider()
