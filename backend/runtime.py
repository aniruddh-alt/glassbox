"""Backend readiness. Polls the GPU pod when pod.url is set; otherwise stays on synthetic fallback.
State is read per-request by the API.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import os
import threading
import time

STATE: dict = {
    "mode": "loading",
    "model_loaded": False,
    "sae_loaded": False,
    "pod_reachable": False,
    "pod_health": None,
}


def _apply_pod_health(h: dict | None, *, reachable: bool) -> dict:
    STATE["pod_reachable"] = reachable
    STATE["pod_health"] = h
    if not reachable or h is None:
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
        return dict(STATE)
    STATE["model_loaded"] = bool(h.get("model_loaded"))
    STATE["sae_loaded"] = bool(h.get("sae_loaded"))
    STATE["sae_recon_cosine"] = h.get("sae_recon_cosine")
    STATE["sae_recon_ok"] = h.get("sae_recon_ok")
    if h.get("mode") == "real" and STATE["model_loaded"] and STATE["sae_loaded"]:
        STATE["mode"] = "real"
    elif h.get("mode") == "loading":
        STATE["mode"] = "loading"
    else:
        STATE["mode"] = "fallback"
    return dict(STATE)


def _poll_pod_once(cfg) -> dict:
    """Fetch pod /health once. Never raises."""
    if not cfg.pod.url:
        return _apply_pod_health(None, reachable=False)
    try:
        from . import pod_client

        return _apply_pod_health(pod_client.health(cfg.pod, cfg.pod_token), reachable=True)
    except Exception as e:  # noqa: BLE001
        print(f"[runtime] pod health poll failed ({e})")
        return _apply_pod_health(None, reachable=False)


def _pod_poll_loop(cfg) -> None:
    while True:
        _poll_pod_once(cfg)
        time.sleep(cfg.pod.poll_interval)


_poll_thread: threading.Thread | None = None


def _ensure_poll_loop(cfg) -> None:
    """Start the background pod-health poll loop once (idempotent)."""
    global _poll_thread
    if _poll_thread is not None and _poll_thread.is_alive():
        return
    _poll_thread = threading.Thread(target=_pod_poll_loop, args=(cfg,), daemon=True)
    _poll_thread.start()


def refresh_pod_health(cfg) -> dict:
    """Re-check pod readiness (e.g. after a failed turn)."""
    return _poll_pod_once(cfg)


def start_loading(cfg) -> None:
    """Bring the real path online. Called once at FastAPI startup."""
    if not cfg.pod.url:
        STATE.update(
            mode="fallback",
            model_loaded=False,
            sae_loaded=False,
            pod_reachable=False,
            pod_health=None,
        )
        return

    STATE["mode"] = "loading"
    if os.getenv("GLASSBOX_EAGER_LOAD") == "1":
        _poll_pod_once(cfg)
    _ensure_poll_loop(cfg)


def health_payload(cfg) -> dict:
    ph = STATE.get("pod_health") or {}
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": cfg.model.model_id,
        "layer": cfg.model.layer,
        "d_sae": ph.get("d_sae", cfg.sae.d_sae),
        "trackers": ph.get("trackers", []),
        "sae_recon_cosine": STATE.get("sae_recon_cosine", ph.get("sae_recon_cosine")),
        "sae_recon_ok": STATE.get("sae_recon_ok", ph.get("sae_recon_ok")),
        "pod_reachable": STATE.get("pod_reachable", False),
        "pod_url_configured": bool(cfg.pod.url),
        "anthropic_configured": ph.get("anthropic_configured"),
        "active_probe_jobs": ph.get("active_probe_jobs"),
    }
