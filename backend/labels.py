"""Neuronpedia feature-label cache. Keyless GET, cached in-memory. CPU only.

OWNER: Lane A. The bulk /api/explanation/export endpoint is REMOVED (400s) — use the
per-feature GET below, or pre-pull the S3 v1 dataset dump at startup for offline speed.
"""
from __future__ import annotations

from . import config

_cache: dict[int, str] = {}


def get_label(index: int) -> str:
    """Return the auto-interp description for an SAE feature index (cached)."""
    if index in _cache:
        return _cache[index]
    # TODO(Lane A):
    # import httpx
    # url = config.NP_FEATURE_URL.format(model=config.NP_MODEL, source=config.NP_SOURCE, index=index)
    # r = httpx.get(url, timeout=5)
    # desc = (r.json().get("explanations") or [{}])[0].get("description", "(no label)")
    # _cache[index] = desc; return desc
    return "(unfetched)"


def preload_from_s3() -> None:
    """Optional: bulk-load the S3 v1 dump into _cache at startup (no live calls during demo)."""
    ...
