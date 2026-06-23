"""Family A — Gemma Scope SAE feature cloud (EXPLORATORY; labels unreliable).

OWNER: Lane B. Load once at startup; sae_topk runs per token (one matmul, microseconds).
"""

from __future__ import annotations

from ..config import SAEConfig, FeatureCloudConfig

_sae = None  # set by load_sae()


def _resolve_device_inline() -> str:
    """Resolve device for load_sae when no AppConfig is available (direct GPU callers)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sae(sae: SAEConfig, layer: int, device: str | None = None):
    """Load the Gemma Scope SAE for `layer` onto the resolved device.

    sae: SAEConfig with release and sae_id_pattern.
    layer: the residual layer index to load.
    device: explicit device string; if None, resolved via torch availability.
    """
    global _sae
    from sae_lens import SAE

    dev = device if device is not None else _resolve_device_inline()
    sae_id = sae.sae_id_pattern.format(layer=layer)
    loaded = SAE.from_pretrained(
        release=sae.release, sae_id=sae_id, device=dev
    )
    # SAELens has returned either an SAE or a (sae, cfg, sparsity) tuple across versions.
    _sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    _sae = _sae.to(dev).eval()
    return _sae


def width() -> int | None:
    """Number of latents (d_sae) of the loaded SAE, or None if not loaded."""
    if _sae is None:
        return None
    cfg = getattr(_sae, "cfg", None)
    return int(getattr(cfg, "d_sae", 0)) or None


def _prep(act):
    """Flatten one token's activation to [d_in] float32 on the SAE's device."""
    dev = next(_sae.parameters()).device
    return act.detach().reshape(-1).float().to(dev)


def sae_topk(act, sae: SAEConfig, feature_cloud: FeatureCloudConfig, *, np_source: str, k: int | None = None) -> list[dict]:
    """act: layer-LAYER resid_post activation tensor [d_in=2560] (one token).
    Returns up to k {index, act, source} dicts (labels attached later by labels.py).
    Mask special tokens upstream — their activations are high-norm noise.
    Used by the legacy raw-activation ranking path; attribution_topk is preferred.

    sae: SAEConfig (provides metadata, not used for indexing here).
    feature_cloud: FeatureCloudConfig (provides topk default).
    np_source: Neuronpedia source slug for the source field (threaded from gpu_service via cfg.np_source(layer)).
    k: override for topk; defaults to feature_cloud.topk.
    """
    import torch

    if _sae is None:
        raise RuntimeError("call load_sae() first")
    k = k if k is not None else feature_cloud.topk
    a = _prep(act)
    with torch.no_grad():
        feats = _sae.encode(a.unsqueeze(0)).squeeze(0)  # [d_sae=16384]
    vals, idx = feats.topk(k)
    return [
        {"index": int(i), "act": round(float(v), 3), "source": np_source}
        for v, i in zip(vals.tolist(), idx.tolist())
        if v > 0
    ]


def _W_dec():
    """Decoder matrix [d_sae, d_in] — row f is feature f's residual-space direction."""
    w = getattr(_sae, "W_dec", None)
    if w is None:
        raise RuntimeError("loaded SAE exposes no W_dec; cannot compute attribution")
    return w


def feature_attribution(acts, grad, keep):
    """Per-feature signed attribution over kept positions. Returns (attr[d_sae], act_max[d_sae]).

    attr[f] = sum_p act_{f,p} · (grad_p · W_dec[f]) — the first-order (attribution-patching)
    estimate of how much feature f shaped the response-logit metric. act_max is each feature's
    peak activation over kept positions (for display)."""
    import torch

    if _sae is None:
        raise RuntimeError("call load_sae() first")
    dev = next(_sae.parameters()).device
    sel = torch.as_tensor(list(keep), dtype=torch.long, device=acts.device)
    a = acts.index_select(0, sel).detach().float().to(dev)
    g = grad.index_select(0, sel).detach().float().to(dev)
    with torch.no_grad():
        feats = _sae.encode(a).float()           # [n, d_sae] sparse JumpReLU activations
        gd = g @ _W_dec().float().t()            # [n, d_sae] grad · decoder direction per feature
        attr = (feats * gd).sum(0)               # [d_sae] attribution summed over positions
        act_max = feats.max(0).values            # [d_sae] representative activation for display
    return attr, act_max


def attribution_topk(
    acts, grad, keep,
    sae: SAEConfig,
    feature_cloud: FeatureCloudConfig,
    *,
    np_source: str,
    cap: int | None = None,
    baseline=None,
) -> list[dict]:
    """Rank features by causal effect on the response, not by raw activation.

    For each kept position p, feature f's attribution is act_{f,p} · (grad_p · W_dec[f]) — the
    first-order (attribution-patching) estimate of how much ablating that feature would change a
    response-logit metric — summed over positions. This is the method Anthropic (Scaling
    Monosemanticity) and Goodfire use to pick "the features behind THIS output"; it structurally
    down-weights high-frequency grammatical features (large activation, tiny per-token effect on
    the answer) and surfaces features that actually shape the medical response.

    baseline: optional [d_sae] attribution vector from a neutral prompt. When given, it is
    SUBTRACTED, cancelling "always-on" discourse features (greetings, "Okay, let's...") that score
    highly on every reply, leaving topic-specific features (contrastive attribution).

    acts/grad: [n_pos, d_in] response-position activations and dL/d(resid_post), ALIGNED.
    keep: positions in [0, n_pos) to include (special tokens already excluded by the caller).
    sae: SAEConfig (metadata, not used for indexing here).
    feature_cloud: FeatureCloudConfig (provides topk_candidates default).
    np_source: Neuronpedia source slug for the source field.
    cap: override for candidate pool size; defaults to feature_cloud.topk_candidates.
    Returns up to `cap` {index, act, attr, source} dicts, highest attribution first.
    """
    if _sae is None:
        raise RuntimeError("call load_sae() first")
    if not keep:
        return []
    cap = cap if cap is not None else feature_cloud.topk_candidates
    attr, act_max = feature_attribution(acts, grad, keep)
    if baseline is not None:
        attr = attr - baseline.to(attr.device)   # contrastive: cancel always-on features
    k = min(cap, attr.shape[0])
    vals, idxs = attr.topk(k)
    out: list[dict] = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if v <= 0:  # keep only features that positively contributed to producing the answer
            continue
        out.append(
            {
                "index": int(i),
                "act": round(float(act_max[i]), 3),
                "attr": round(float(v), 4),
                "source": np_source,
            }
        )
    return out


def reconstruction_error(act, sae: SAEConfig) -> dict:
    """Sanity check the layer/dtype are correct: a correctly-wired Gemma Scope SAE
    reconstructs its own training-layer activations with high cosine (~0.9+).
    A low cosine usually means an off-by-one layer (hidden_states[0] is the embedding).

    sae: SAEConfig (metadata, used for future validation checks).
    """
    import torch

    a = _prep(act)
    with torch.no_grad():
        recon = _sae.decode(_sae.encode(a.unsqueeze(0))).squeeze(0)
    rel_mse = (a - recon).pow(2).mean().item() / (a.pow(2).mean().item() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(a, recon, dim=0).item()
    return {"cosine": cos, "rel_mse": rel_mse}
