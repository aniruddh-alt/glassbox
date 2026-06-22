"""Backend readiness. Polls the GPU pod when POD_URL is set; otherwise stays on synthetic fallback.
State is read per-request by the API.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import os
import threading
import time

from . import config

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


def _poll_pod_once() -> dict:
    """Fetch pod /health once. Never raises."""
    if not config.POD_URL:
        return _apply_pod_health(None, reachable=False)
    try:
        from . import pod_client

        return _apply_pod_health(pod_client.health(), reachable=True)
    except Exception as e:  # noqa: BLE001
        print(f"[runtime] pod health poll failed ({e})")
        return _apply_pod_health(None, reachable=False)


def _pod_poll_loop() -> None:
    while True:
        _poll_pod_once()
        time.sleep(config.POD_POLL_INTERVAL)


_poll_thread: threading.Thread | None = None


def _ensure_poll_loop() -> None:
    """Start the background pod-health poll loop once (idempotent).

    Continuous polling is what lets a pod that blips offline and then recovers self-heal back
    to mode=real. Without it the backend can get trapped in fallback: once mode != "real",
    analyze_turn stops calling refresh_pod_health(), so nothing ever re-checks the pod.
    """
    global _poll_thread
    if _poll_thread is not None and _poll_thread.is_alive():
        return
    _poll_thread = threading.Thread(target=_pod_poll_loop, daemon=True)
    _poll_thread.start()


def refresh_pod_health() -> dict:
    """Re-check pod readiness (e.g. after a failed turn)."""
    return _poll_pod_once()


def start_loading() -> None:
    """Bring the real path online. Called once at FastAPI startup.

    When POD_URL is set, always keep a background poll running so a transient pod blip
    self-heals. GLASSBOX_EAGER_LOAD=1 additionally does the first poll synchronously, so
    startup blocks until the initial pod health is known. When POD_URL is unset, stay on
    synthetic fallback.
    """
    if not config.POD_URL:
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
        _poll_pod_once()
    _ensure_poll_loop()


def health_payload() -> dict:
    ph = STATE.get("pod_health") or {}
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": config.MODEL_ID,
        "layer": config.LAYER,
        "d_sae": ph.get("d_sae", config.D_SAE),
        "trackers": ph.get("trackers", []),
        "sae_recon_cosine": STATE.get("sae_recon_cosine", ph.get("sae_recon_cosine")),
        "sae_recon_ok": STATE.get("sae_recon_ok", ph.get("sae_recon_ok")),
        "pod_reachable": STATE.get("pod_reachable", False),
        "pod_url_configured": bool(config.POD_URL),
        "anthropic_configured": ph.get("anthropic_configured"),
        "active_probe_jobs": ph.get("active_probe_jobs"),
    }
