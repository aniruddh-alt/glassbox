"""Orchestration-side HTTP client for the GPU pod service. CPU only — never imports torch."""

from __future__ import annotations

from . import config

_UA = {"User-Agent": "glassbox-orchestration/0.1"}


class PodError(Exception):
    """Raised when the pod is unreachable or returns a non-2xx response."""


def _headers() -> dict[str, str]:
    h = dict(_UA)
    if config.POD_TOKEN:
        h["Authorization"] = f"Bearer {config.POD_TOKEN}"
    return h


def _base_url() -> str:
    if not config.POD_URL:
        raise PodError("POD_URL is not configured")
    return config.POD_URL


def _post(path: str, payload: dict) -> dict:
    import httpx

    try:
        r = httpx.post(
            f"{_base_url()}{path}",
            json=payload,
            headers=_headers(),
            timeout=config.POD_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise PodError(str(e)) from e
    if r.status_code != 200:
        raise PodError(f"POST {path} returned {r.status_code}: {r.text[:200]}")
    return r.json()


def health() -> dict | None:
    """Poll pod /health. Returns None when POD_URL is unset; raises PodError on failure."""
    if not config.POD_URL:
        return None
    import httpx

    try:
        r = httpx.get(
            f"{_base_url()}/health",
            headers=_headers(),
            timeout=min(config.POD_TIMEOUT, 10.0),
        )
    except httpx.HTTPError as e:
        raise PodError(str(e)) from e
    if r.status_code != 200:
        raise PodError(f"GET /health returned {r.status_code}")
    return r.json()


def inference(messages: list[dict], *, max_new: int | None = None) -> dict:
    """POST /inference → {answer}."""
    return _post(
        "/inference",
        {"messages": messages, "max_new": max_new or config.MAX_NEW_TOKENS},
    )


def activations(
    messages: list[dict],
    *,
    max_new: int | None = None,
    attribution: bool = False,
) -> dict:
    """POST /activations → {answer, resp_start, act_last, act_resp}."""
    return _post(
        "/activations",
        {
            "messages": messages,
            "max_new": max_new or config.MAX_NEW_TOKENS,
            "attribution": attribution,
        },
    )


def sae_features(
    messages: list[dict],
    *,
    max_new: int | None = None,
    attribution: bool | None = None,
    cap: int | None = None,
) -> dict:
    """POST /sae/features → {answer, candidates, reliable}."""
    body: dict = {
        "messages": messages,
        "max_new": max_new or config.MAX_NEW_TOKENS,
    }
    if attribution is not None:
        body["attribution"] = attribution
    if cap is not None:
        body["cap"] = cap
    return _post("/sae/features", body)


def turn(messages: list[dict], *, max_new: int | None = None) -> dict:
    """POST /turn → {answer, candidates, trackers, reliable}."""
    return _post(
        "/turn",
        {"messages": messages, "max_new": max_new or config.MAX_NEW_TOKENS},
    )
