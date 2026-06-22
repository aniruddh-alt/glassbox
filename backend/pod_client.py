"""Orchestration-side HTTP client for the GPU pod service. CPU only — never imports torch."""

from __future__ import annotations

_UA = {"User-Agent": "glassbox-orchestration/0.1"}


class PodError(Exception):
    """Raised when the pod is unreachable or returns a non-2xx response."""

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


def _headers(pod_token: str) -> dict[str, str]:
    h = dict(_UA)
    if pod_token:
        h["Authorization"] = f"Bearer {pod_token}"
    return h


def _base_url(pod) -> str:
    if not pod.url:
        raise ConnectionError("pod.url is not configured")
    return pod.url


def _post(path: str, payload: dict, pod, pod_token: str, stage: str = "request") -> dict:
    import httpx

    try:
        r = httpx.post(
            f"{_base_url(pod)}{path}",
            json=payload,
            headers=_headers(pod_token),
            timeout=pod.timeout,
        )
    except httpx.HTTPError as e:
        raise PodError(0, stage) from e
    if r.status_code != 200:
        print(f"[pod] {stage} HTTP {r.status_code}: {r.text[:200]}")
        raise PodError(r.status_code, stage)
    return r.json()


def health(pod, pod_token: str) -> dict | None:
    """Poll pod /health. Returns None when pod.url is unset; raises PodError on failure."""
    if not pod.url:
        return None
    import httpx

    try:
        r = httpx.get(
            f"{_base_url(pod)}/health",
            headers=_headers(pod_token),
            timeout=min(pod.timeout, 10.0),
        )
    except httpx.HTTPError as e:
        raise PodError(0, "health") from e
    if r.status_code != 200:
        raise PodError(r.status_code, "health")
    return r.json()


def inference(messages: list[dict], pod, pod_token: str, *, max_new: int) -> dict:
    """POST /inference → {answer}."""
    return _post("/inference", {"messages": messages, "max_new": max_new}, pod, pod_token, stage="inference")


def activations(messages: list[dict], pod, pod_token: str, *, max_new: int, attribution: bool = False) -> dict:
    """POST /activations → {answer, resp_start, act_last, act_resp}."""
    return _post(
        "/activations",
        {"messages": messages, "max_new": max_new, "attribution": attribution},
        pod,
        pod_token,
        stage="activations",
    )


def sae_features(
    messages: list[dict],
    pod,
    pod_token: str,
    *,
    max_new: int,
    attribution: bool | None = None,
    cap: int | None = None,
) -> dict:
    """POST /sae/features → {answer, candidates, reliable}."""
    body: dict = {"messages": messages, "max_new": max_new}
    if attribution is not None:
        body["attribution"] = attribution
    if cap is not None:
        body["cap"] = cap
    return _post("/sae/features", body, pod, pod_token, stage="sae_features")


def turn(messages: list[dict], pod, pod_token: str, *, max_new: int) -> dict:
    """POST /turn → {answer, candidates, trackers, reliable}."""
    return _post("/turn", {"messages": messages, "max_new": max_new}, pod, pod_token, stage="turn")


def track(request: str, pod, pod_token: str) -> dict:
    """POST /api/track → {tracker_id, status}. Body carries only the NL request (no PHI)."""
    import httpx

    try:
        r = httpx.post(
            f"{_base_url(pod)}/api/track",
            json={"request": request},
            headers=_headers(pod_token),
            timeout=pod.timeout,
        )
    except httpx.HTTPError as e:
        raise PodError(0, "track") from e
    if r.status_code == 200:
        return r.json()
    detail = ""
    try:
        detail = str(r.json().get("detail") or "")
    except Exception:  # noqa: BLE001
        detail = ""
    print(f"[pod] track HTTP {r.status_code}: {(detail or r.text)[:200]}")
    if detail:
        return {"status": "unavailable", "detail": detail}
    raise PodError(r.status_code, "track")


def clear_custom_trackers(pod, pod_token: str) -> dict:
    """POST /api/trackers/clear-custom → {removed, trackers}."""
    return _post("/api/trackers/clear-custom", {}, pod, pod_token, stage="clear_custom_trackers")


def track_status(tracker_id: str, pod, pod_token: str) -> dict:
    """GET /api/track/{id} → the probe-job record."""
    import httpx

    try:
        r = httpx.get(
            f"{_base_url(pod)}/api/track/{tracker_id}",
            headers=_headers(pod_token),
            timeout=min(pod.timeout, 15.0),
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
