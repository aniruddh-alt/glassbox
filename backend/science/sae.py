"""Family A — Gemma Scope SAE feature cloud (EXPLORATORY; labels unreliable).

OWNER: Lane B. Load once at startup; sae_topk runs per token (one matmul, microseconds).
"""

from __future__ import annotations

from .. import config

_sae = None  # set by load_sae()


def load_sae(device: str | None = None):
    """Load the Gemma Scope SAE for the locked layer/width onto the resolved device."""
    global _sae
    from sae_lens import SAE

    dev = config.resolve_device(device)
    loaded = SAE.from_pretrained(
        release=config.SAE_RELEASE, sae_id=config.SAE_ID, device=dev
    )
    # SAELens has returned either an SAE or a (sae, cfg, sparsity) tuple across versions.
    _sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    _sae = _sae.to(dev).eval()
    return _sae


def _prep(act):
    """Flatten one token's activation to [d_in] float32 on the SAE's device."""
    dev = next(_sae.parameters()).device
    return act.detach().reshape(-1).float().to(dev)


def sae_topk(act, k: int = config.TOPK) -> list[dict]:
    """act: layer-12 residual activation tensor [d_in=2304] (one token).
    Returns up to k {index, act, source} dicts (labels attached later by labels.py).
    Mask special tokens upstream — their activations are high-norm noise.
    """
    import torch

    if _sae is None:
        raise RuntimeError("call load_sae() first")
    a = _prep(act)
    with torch.no_grad():
        feats = _sae.encode(a.unsqueeze(0)).squeeze(0)  # [d_sae=16384]
    vals, idx = feats.topk(k)
    return [
        {"index": int(i), "act": round(float(v), 3), "source": config.NP_SOURCE}
        for v, i in zip(vals.tolist(), idx.tolist())
        if v > 0
    ]


def reconstruction_error(act) -> dict:
    """Sanity check the layer/dtype are correct: a correctly-wired Gemma Scope SAE
    reconstructs its own training-layer activations with high cosine (~0.9+).
    A low cosine usually means an off-by-one layer (hidden_states[0] is the embedding)."""
    import torch

    a = _prep(act)
    with torch.no_grad():
        recon = _sae.decode(_sae.encode(a.unsqueeze(0))).squeeze(0)
    rel_mse = (a - recon).pow(2).mean().item() / (a.pow(2).mean().item() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(a, recon, dim=0).item()
    return {"cosine": cos, "rel_mse": rel_mse}
