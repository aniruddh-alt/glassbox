"""Backend readiness. A background thread tries to load the real model + SAE; until/unless
that succeeds, requests use the synthetic fallback. State is read per-request by the API.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import threading

from . import config

STATE: dict = {"mode": "loading", "model_loaded": False, "sae_loaded": False}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _default_load_engine() -> None:
    from . import engine

    engine.load_engine()


def _default_load_sae() -> None:
    from .science import sae

    sae.load_sae()


def _attempt_load(*, torch_available=None, load_engine=None, load_sae=None) -> dict:
    """Try to bring the real path online. Never raises — failure => fallback."""
    avail = _torch_available() if torch_available is None else torch_available
    if not avail:
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
        return dict(STATE)
    try:
        (load_engine or _default_load_engine)()
        STATE["model_loaded"] = True
        (load_sae or _default_load_sae)()
        STATE["sae_loaded"] = True
        STATE["mode"] = "real"
    except Exception as e:  # noqa: BLE001 — degrade, never crash the app
        print(f"[runtime] real load failed ({e}); using synthetic fallback")
        STATE.update(mode="fallback")
    return dict(STATE)


def start_loading() -> None:
    """Kick off the load off the request path. Called once at FastAPI startup."""
    STATE["mode"] = "loading"
    threading.Thread(target=_attempt_load, daemon=True).start()


def health_payload() -> dict:
    from .science.persona import _trackers

    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": config.MODEL_ID,
        "layer": config.LAYER,
        "trackers": list(_trackers.keys()),
    }
