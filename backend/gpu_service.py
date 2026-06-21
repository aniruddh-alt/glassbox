"""GPU pod FastAPI service — owns torch, model, and SAE. Run via:
  uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config

STATE: dict = {
    "mode": "loading",
    "model_loaded": False,
    "sae_loaded": False,
    "sae_recon_cosine": None,
    "sae_recon_ok": None,
}

# Cached contrastive-attribution baseline: per-feature attribution on a neutral prompt, computed
# once and subtracted from each turn so always-on discourse features cancel. {"vec": tensor|None}.
_BASELINE: dict = {"vec": None, "tried": False}

app = FastAPI(title="GlassBox GPU")


def _require_auth(request: Request) -> None:
    token = config.POD_TOKEN
    if not token:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _run_recon_check() -> None:
    try:
        from . import engine
        from .science import sae

        rec = sae.reconstruction_error(engine.probe_activation(config.RECON_PROBE))
        cos = round(float(rec["cosine"]), 3)
        STATE["sae_recon_cosine"] = cos
        STATE["sae_recon_ok"] = cos >= config.RECON_MIN_COSINE
        if STATE["sae_recon_ok"]:
            print(f"[gpu_service] SAE reconstruction cosine {cos} [OK]")
        else:
            print(
                f"[gpu_service] WARNING SAE reconstruction cosine {cos} < "
                f"{config.RECON_MIN_COSINE} — feature cloud may be unreliable"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[gpu_service] recon check skipped ({e})")
        STATE["sae_recon_cosine"] = None
        STATE["sae_recon_ok"] = None


def _attempt_load() -> None:
    try:
        from . import engine
        from .science import persona, sae

        engine.load_engine()
        STATE["model_loaded"] = True
        sae.load_sae()
        STATE["sae_loaded"] = True
        loaded = persona.load_artifacts(include=config.ENABLED_TRACKERS)
        if loaded:
            print(f"[gpu_service] loaded probe trackers: {', '.join(loaded)}")
        STATE["mode"] = "real"
        _run_recon_check()
    except Exception as e:  # noqa: BLE001
        print(f"[gpu_service] load failed ({e})")
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def _capture(messages: list[dict], max_new: int, *, attribution: bool | None = None) -> dict:
    from . import engine

    if STATE["mode"] != "real":
        raise HTTPException(status_code=503, detail="model not loaded")
    return engine.generate_and_capture(messages, max_new=max_new, attribution=attribution)


def _pooled_activations(res: dict) -> tuple:
    acts, rs = res["acts"], res["resp_start"]
    resp_acts = acts[rs:]
    act_last = acts[rs - 1] if rs > 0 else None
    act_resp = resp_acts.float().mean(0) if resp_acts.shape[0] > 0 else None
    return resp_acts, act_last, act_resp


def _baseline_vec():
    """Cached per-feature attribution on a neutral prompt, subtracted to cancel always-on
    discourse features (greetings, "Okay, let's..."). Computed once on first use; None if
    disabled or the backward pass is unavailable. Requires the model loaded (mode == real)."""
    if not config.CONTRAST_BASELINE or STATE["mode"] != "real":
        return None
    if not _BASELINE["tried"]:
        _BASELINE["tried"] = True
        try:
            from . import engine
            from .science import sae

            res = engine.generate_and_capture(
                [{"role": "user", "content": config.CONTRAST_PROMPT}],
                max_new=config.CONTRAST_MAX_NEW,
                attribution=True,
            )
            g = res.get("grad")
            if g is not None:
                rs = res["resp_start"]
                tok = res["tok"]
                special = {tok.convert_tokens_to_ids(t) for t in config.MASK_TOKENS}
                ids = res["out_ids"][rs:].tolist()
                resp_acts = res["acts"][rs:]
                keep = [p for p in range(resp_acts.shape[0]) if not (p < len(ids) and ids[p] in special)]
                keep = [p for p in keep if p >= config.PREAMBLE_SKIP] or keep  # match the turn's preamble skip
                attr, _ = sae.feature_attribution(resp_acts, g[rs:], keep)
                _BASELINE["vec"] = attr.detach()
                print(f"[gpu_service] contrastive baseline cached ({int((attr > 0).sum())} active features)")
            else:
                print("[gpu_service] baseline backward unavailable; contrast disabled")
        except Exception as e:  # noqa: BLE001 — contrast is an enhancement, never block a turn
            print(f"[gpu_service] baseline computation failed ({e}); contrast disabled")
    return _BASELINE["vec"]


def _sae_candidates(
    res: dict,
    resp_acts,
    resp_grad,
    *,
    cap: int,
    baseline=None,
) -> list[dict]:
    from .science.feature_provider import LocalSAEProvider

    tok = res["tok"]
    special = {tok.convert_tokens_to_ids(t) for t in config.MASK_TOKENS}
    resp_ids = res["out_ids"][res["resp_start"] :].tolist()
    return LocalSAEProvider().features_for(
        res["answer"],
        activations=resp_acts,
        token_ids=resp_ids,
        special_ids=special,
        grad=resp_grad,
        baseline=baseline,
        cap=cap,
    )


def _score_trackers(act_last, act_resp) -> dict:
    from .science.persona import score_all_trackers

    return score_all_trackers(act_last, act_resp)


def _tensor_list(t) -> list[float] | None:
    if t is None:
        return None
    return t.detach().float().cpu().tolist()


@app.on_event("startup")
def _startup() -> None:
    _attempt_load()


@app.get("/health")
def health() -> dict:
    from .science import sae
    from .science.persona import _trackers

    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": config.MODEL_ID,
        "layer": config.LAYER,
        "d_sae": sae.width() or config.D_SAE,
        "trackers": list(_trackers.keys()),
        "sae_recon_cosine": STATE.get("sae_recon_cosine"),
        "sae_recon_ok": STATE.get("sae_recon_ok"),
    }


@app.post("/inference", dependencies=[Depends(_require_auth)])
def inference(body: dict) -> dict:
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or config.MAX_NEW_TOKENS)
    res = _capture(messages, max_new, attribution=False)
    return {"answer": res["answer"]}


@app.post("/activations", dependencies=[Depends(_require_auth)])
def activations(body: dict) -> dict:
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or config.MAX_NEW_TOKENS)
    attribution = bool(body.get("attribution", False))
    res = _capture(messages, max_new, attribution=attribution)
    _, act_last, act_resp = _pooled_activations(res)
    return {
        "answer": res["answer"],
        "resp_start": res["resp_start"],
        "act_last": _tensor_list(act_last),
        "act_resp": _tensor_list(act_resp),
    }


@app.post("/sae/features", dependencies=[Depends(_require_auth)])
def sae_features(body: dict) -> dict:
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or config.MAX_NEW_TOKENS)
    cap = int(body.get("cap") or config.TOPK_CANDIDATES)
    attribution = body.get("attribution")
    if attribution is None:
        attribution = config.RANK_METHOD == "attribution"
    contrast = body.get("contrast")
    if contrast is None:
        contrast = config.CONTRAST_BASELINE
    res = _capture(messages, max_new, attribution=attribution)
    resp_acts, _, _ = _pooled_activations(res)
    grad = res.get("grad")
    resp_grad = grad[res["resp_start"] :] if grad is not None else None
    baseline = _baseline_vec() if (attribution and contrast) else None
    candidates = _sae_candidates(res, resp_acts, resp_grad, cap=cap, baseline=baseline)
    return {"answer": res["answer"], "candidates": candidates, "reliable": True}


@app.post("/turn", dependencies=[Depends(_require_auth)])
def turn(body: dict) -> dict:
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or config.MAX_NEW_TOKENS)
    attribution = config.RANK_METHOD == "attribution"

    _t0 = time.perf_counter()
    res = _capture(messages, max_new, attribution=attribution)
    capture_ms = (time.perf_counter() - _t0) * 1000.0

    _t1 = time.perf_counter()
    resp_acts, act_last, act_resp = _pooled_activations(res)
    grad = res.get("grad")
    resp_grad = grad[res["resp_start"] :] if grad is not None else None
    baseline = _baseline_vec() if attribution else None
    candidates = _sae_candidates(
        res, resp_acts, resp_grad, cap=config.TOPK_CANDIDATES, baseline=baseline
    )
    sae_ms = (time.perf_counter() - _t1) * 1000.0

    _t2 = time.perf_counter()
    trackers = _score_trackers(act_last, act_resp)
    trackers_ms = (time.perf_counter() - _t2) * 1000.0

    return {
        "answer": res["answer"],
        "candidates": candidates,
        "trackers": trackers,
        "reliable": True,
        "timings": {"capture": capture_ms, "sae": sae_ms, "trackers": trackers_ms},
    }


def _launch_agent(tracker_id: str) -> None:
    """Run the blocking interp-agent pipeline (pod-local gemma + Claude) off the event loop."""
    from .agent.interp_agent import run_interp_agent
    from .science import concept_synth as cs

    def _run():
        try:
            run_interp_agent(tracker_id)
        except Exception as e:  # noqa: BLE001 - surface failure in the job record
            cs.update_job(tracker_id, status="error", error=str(e))

    asyncio.create_task(asyncio.to_thread(_run))


@app.post("/api/track", dependencies=[Depends(_require_auth)])
async def track(body: dict) -> dict:
    """Submit a natural-language monitoring request. Trains a persona-vector probe in the
    background ON THIS POD (local gemma generation + Claude design/judge) and, if it clears the
    AUROC gate, registers it into THIS process's live `persona._trackers` — the same dict /turn
    scores against, so the next chat turn picks it up. Returns immediately; poll the GET below."""
    from .science import concept_synth as cs

    if STATE["mode"] != "real":
        raise HTTPException(status_code=503, detail="model not loaded")
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set on pod")
    request = (body.get("request") or body.get("concept") or body.get("name") or "").strip()
    if not request:
        raise HTTPException(status_code=400, detail="request is required")
    tracker_id = cs.create_job(request)
    _launch_agent(tracker_id)
    return {"tracker_id": tracker_id, "status": "pending"}


@app.get("/api/track/{tracker_id}", dependencies=[Depends(_require_auth)])
async def track_status(tracker_id: str):
    from .science import concept_synth as cs

    job = cs.get_job(tracker_id)
    if job is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return job
