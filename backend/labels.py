"""Neuronpedia feature-label cache. Keyless GET, cached in-memory. CPU only.

OWNER: Lane A. The bulk /api/explanation/export endpoint is REMOVED (400s) — use the
per-feature GET below, or pre-pull the S3 v1 dataset dump at startup for offline speed.
"""

from __future__ import annotations

from . import config

_cache: dict[int, str] = {}
_UA = {"User-Agent": "glassbox-hackathon/0.1"}


def get_label(index: int, timeout: float = 6.0) -> str:
    """Return the auto-interp description for an SAE feature index (cached).
    Falls back to 'feature {index}' on any error — labels are caveated anyway."""
    if index in _cache:
        return _cache[index]
    import httpx

    url = config.NP_FEATURE_URL.format(
        model=config.NP_MODEL, source=config.NP_SOURCE, index=index
    )
    desc = None
    try:
        r = httpx.get(url, headers=_UA, timeout=timeout)
        if r.status_code == 200:
            exps = r.json().get("explanations") or []
            if exps:
                desc = exps[0].get("description")
    except Exception:
        desc = None
    desc = (desc or f"feature {index}").strip()
    _cache[index] = desc
    return desc


def get_labels(indices) -> dict[int, str]:
    """Fetch+cache a batch of labels. Use offline to pre-warm before a demo."""
    return {i: get_label(i) for i in indices}


def preload_from_s3() -> None:
    """Optional: bulk-load the S3 v1 dump into _cache at startup (no live calls during demo)."""
    ...
