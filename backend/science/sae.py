"""Family A — Gemma Scope SAE feature cloud (EXPLORATORY; labels unreliable).

OWNER: Lane B. Load once at startup; sae_topk runs per token (one matmul, microseconds).
"""
from __future__ import annotations

from .. import config

_sae = None  # set by load_sae()


def load_sae():
    """Load the Gemma Scope SAE for the locked layer/width."""
    global _sae
    # TODO(Lane B):
    # from sae_lens import SAE
    # _sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID,
    #                            device=config.DEVICE)  # dtype float32
    return _sae


def sae_topk(act, k: int = config.TOPK) -> list[dict]:
    """act: layer-12 residual activation tensor [d_in=2304] (one token).
    Returns up to k {index, act, source} dicts (labels attached later by labels.py).
    Mask special tokens upstream — their activations are high-norm noise.
    """
    # TODO(Lane B):
    # feats = _sae.encode(act.float())              # [d_sae=16384]
    # vals, idx = feats.topk(k)
    # return [{"index": int(i), "act": float(v), "source": config.NP_SOURCE}
    #         for v, i in zip(vals.tolist(), idx.tolist())]
    raise NotImplementedError("Lane B: SAE.encode + topk")
