"""Orchestration-side HTTP client for the GPU pod service. CPU only — never imports torch."""

from __future__ import annotations

from . import config

_UA = {"User-Agent": "glassbox-orchestration/0.1"}


class PodError(Exception):
    """Raised when the pod is unreachable or returns a non-2xx response.

    `detail` may carry safe setup errors (e.g. missing API key) for whitelisted endpoints.
    Never embed /turn response bodies — they may contain model output.
    """

    def __init__(self, status: int, stage: str, *, detail: str = "") -> None:
        self.status = status
        self.stage = stage
        self.detail = detail
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.detail:
            return f"pod {self.stage} failed: {self.detail}"
        return f"pod {self.stage} failed: HTTP {self.status}"


def pod_unavailable_payload(exc: Exception) -> dict:
    """Map pod client errors to a JSON body safe for the Build UI."""
    if isinstance(exc, PodError):
        if exc.status == 0:
            detail = "Cannot reach GPU pod — start the SSH tunnel and gpu_service"
        elif exc.detail:
            detail = exc.detail
        else:
            detail = f"GPU pod error during {exc.stage} (HTTP {exc.status})"
        return {"status": "unavailable", "detail": detail}
    if isinstance(exc, ConnectionError):
        return {"status": "unavailable", "detail": "POD_URL is not configured on the backend"}
    msg = str(exc) or type(exc).__name__
    return {"status": "unavailable", "detail": msg}


def _headers() -> dict[str, str]:
    h = dict(_UA)
    if config.POD_TOKEN:
        h["Authorization"] = f"Bearer {config.POD_TOKEN}"
    return h


def _base_url() -> str:
    if not config.POD_URL:
        raise ConnectionError("POD_URL is not configured")
    return config.POD_URL


def _post(path: str, payload: dict, stage: str = "request") -> dict:
    import httpx

    try:
        r = httpx.post(
            f"{_base_url()}{path}",
            json=payload,
            headers=_headers(),
            timeout=config.POD_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise PodError(0, stage) from e
    if r.status_code != 200:
        # LOCAL log only — the body may echo the model answer; never embed in the exception.
        print(f"[pod] {stage} HTTP {r.status_code}: {r.text[:200]}")
        raise PodError(r.status_code, stage)
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
        raise PodError(0, "health") from e
    if r.status_code != 200:
        raise PodError(r.status_code, "health")
    return r.json()


def inference(messages: list[dict], *, max_new: int | None = None) -> dict:
    """POST /inference → {answer}."""
    return _post(
        "/inference",
        {"messages": messages, "max_new": max_new or config.MAX_NEW_TOKENS},
        stage="inference",
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
        stage="activations",
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
    return _post("/sae/features", body, stage="sae_features")


def turn(messages: list[dict], *, max_new: int | None = None) -> dict:
    """POST /turn → {answer, candidates, trackers, reliable}."""
    return _post(
        "/turn",
        {"messages": messages, "max_new": max_new or config.MAX_NEW_TOKENS},
        stage="turn",
    )


def track(request: str) -> dict:
    """POST /api/track → {tracker_id, status}. Body carries only the NL request (no PHI)."""
    import httpx

    try:
        r = httpx.post(
            f"{_base_url()}/api/track",
            json={"request": request},
            headers=_headers(),
            timeout=config.POD_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise PodError(0, "track") from e
    if r.status_code == 200:
        return r.json()
    # Safe to surface pod setup errors (no model I/O in these bodies).
    detail = ""
    try:
        detail = str(r.json().get("detail") or "")
    except Exception:  # noqa: BLE001
        detail = ""
    print(f"[pod] track HTTP {r.status_code}: {(detail or r.text)[:200]}")
    if detail:
        return {"status": "unavailable", "detail": detail}
    raise PodError(r.status_code, "track")


def clear_custom_trackers() -> dict:
    """POST /api/trackers/clear-custom → {removed, trackers}."""
    return _post("/api/trackers/clear-custom", {}, stage="clear_custom_trackers")


def track_status(tracker_id: str) -> dict:
    """GET /api/track/{id} → the probe-job record. A 404 from the pod maps to {"status": "unknown"}
    rather than an error, so a stale/forgotten tracker_id is reported, not raised."""
    import httpx

    try:
        r = httpx.get(
            f"{_base_url()}/api/track/{tracker_id}",
            headers=_headers(),
            timeout=min(config.POD_TIMEOUT, 15.0),
        )
    except httpx.HTTPError as e:
        raise PodError(0, "track_status") from e
    if r.status_code == 404:
        return {"status": "unknown", "detail": "job not found — pod may have restarted"}
    if r.status_code != 200:
        detail = ""
        try:
            detail = str(r.json().get("detail") or "")
        except Exception:  # noqa: BLE001
            detail = ""
        print(f"[pod] track_status HTTP {r.status_code}: {(detail or r.text)[:200]}")
        if detail:
            return {"status": "unavailable", "detail": detail}
        raise PodError(r.status_code, "track_status")
    return r.json()
