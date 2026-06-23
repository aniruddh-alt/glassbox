"""GPU pod FastAPI service — owns torch, model, and SAE. Run via:
  uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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
    cfg = request.app.state.config
    token = cfg.pod_token
    if not token:
        return  # WS0 keeps the "empty token allows" behavior; WS3 hardens this to fail closed.
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _run_recon_check(cfg) -> None:
    try:
        from . import engine
        from .science import sae

        rec = sae.reconstruction_error(
            engine.probe_activation(cfg.sae.recon_probe, cfg.model), cfg.sae
        )
        cos = round(float(rec["cosine"]), 3)
        STATE["sae_recon_cosine"] = cos
        STATE["sae_recon_ok"] = cos >= cfg.sae.recon_min_cosine
        if STATE["sae_recon_ok"]:
            print(f"[gpu_service] SAE reconstruction cosine {cos} [OK]")
        else:
            print(
                f"[gpu_service] WARNING SAE reconstruction cosine {cos} < "
                f"{cfg.sae.recon_min_cosine} — feature cloud may be unreliable"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[gpu_service] recon check skipped ({e})")
        STATE["sae_recon_cosine"] = None
        STATE["sae_recon_ok"] = None


def _attempt_load(cfg) -> None:
    try:
        from . import engine
        from .science import persona, sae

        device = cfg.resolve_device()
        engine.load_engine(cfg.model, device)
        STATE["model_loaded"] = True
        sae.load_sae(cfg.sae, cfg.model.layer, device)
        STATE["sae_loaded"] = True
        loaded = persona.load_artifacts(cfg.probes)
        if loaded:
            print(f"[gpu_service] loaded probe trackers: {', '.join(loaded)}")
        from .science import concept_synth as cs

        n = cs.load_persisted_jobs(mark_orphans=True)
        if n:
            print(f"[gpu_service] restored {n} probe job record(s) from disk")
        STATE["mode"] = "real"
        _run_recon_check(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[gpu_service] load failed ({e})")
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def _capture(messages: list[dict], max_new: int, *, attribution: bool | None = None, cfg) -> dict:
    from . import engine

    if STATE["mode"] != "real":
        raise HTTPException(status_code=503, detail="model not loaded")
    return engine.generate_and_capture(
        messages, max_new=max_new, attribution=attribution,
        model=cfg.model, feature_cloud=cfg.feature_cloud,
    )


def _pooled_activations(res: dict) -> tuple:
    acts, rs = res["acts"], res["resp_start"]
    resp_acts = acts[rs:]
    act_last = acts[rs - 1] if rs > 0 else None
    act_resp = resp_acts.float().mean(0) if resp_acts.shape[0] > 0 else None
    return resp_acts, act_last, act_resp


def _baseline_vec(cfg):
    """Cached per-feature attribution on a neutral prompt, subtracted to cancel always-on
    discourse features (greetings, "Okay, let's..."). Computed once on first use; None if
    disabled or the backward pass is unavailable. Requires the model loaded (mode == real)."""
    if not cfg.feature_cloud.contrast_baseline or STATE["mode"] != "real":
        return None
    if not _BASELINE["tried"]:
        _BASELINE["tried"] = True
        try:
            from . import engine
            from .science import sae

            res = engine.generate_and_capture(
                [{"role": "user", "content": cfg.feature_cloud.contrast_prompt}],
                max_new=cfg.feature_cloud.contrast_max_new,
                attribution=True,
                model=cfg.model,
                feature_cloud=cfg.feature_cloud,
            )
            g = res.get("grad")
            if g is not None:
                rs = res["resp_start"]
                tok = res["tok"]
                special = {tok.convert_tokens_to_ids(t) for t in cfg.model.mask_tokens}
                ids = res["out_ids"][rs:].tolist()
                resp_acts = res["acts"][rs:]
                keep = [p for p in range(resp_acts.shape[0]) if not (p < len(ids) and ids[p] in special)]
                keep = [p for p in keep if p >= cfg.model.preamble_skip] or keep  # match the turn's preamble skip
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
    cfg,
) -> list[dict]:
    from .science.feature_provider import LocalSAEProvider

    tok = res["tok"]
    special = {tok.convert_tokens_to_ids(t) for t in cfg.model.mask_tokens}
    resp_ids = res["out_ids"][res["resp_start"] :].tolist()
    return LocalSAEProvider().features_for(
        res["answer"],
        activations=resp_acts,
        token_ids=resp_ids,
        special_ids=special,
        grad=resp_grad,
        baseline=baseline,
        cap=cap,
        sae=cfg.sae,
        feature_cloud=cfg.feature_cloud,
        model=cfg.model,
        np_source=cfg.np_source(cfg.model.layer),
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
    from .config import load_config

    app.state.config = load_config()
    _attempt_load(app.state.config)


@app.get("/health")
def health(request: Request) -> dict:
    from .science import concept_synth as cs
    from .science import sae
    from .science.persona import _trackers

    cfg = request.app.state.config
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": cfg.model.model_id,
        "layer": cfg.model.layer,
        "d_sae": sae.width() or cfg.sae.d_sae,
        "trackers": list(_trackers.keys()),
        "sae_recon_cosine": STATE.get("sae_recon_cosine"),
        "sae_recon_ok": STATE.get("sae_recon_ok"),
        "anthropic_configured": bool(cfg.anthropic_api_key),
        "active_probe_jobs": cs.active_job_count(),
    }


@app.post("/inference", dependencies=[Depends(_require_auth)])
def inference(body: dict, request: Request) -> dict:
    cfg = request.app.state.config
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or cfg.model.max_new_tokens)
    res = _capture(messages, max_new, attribution=False, cfg=cfg)
    return {"answer": res["answer"]}


@app.post("/activations", dependencies=[Depends(_require_auth)])
def activations(body: dict, request: Request) -> dict:
    cfg = request.app.state.config
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or cfg.model.max_new_tokens)
    attribution = bool(body.get("attribution", False))
    res = _capture(messages, max_new, attribution=attribution, cfg=cfg)
    _, act_last, act_resp = _pooled_activations(res)
    return {
        "answer": res["answer"],
        "resp_start": res["resp_start"],
        "act_last": _tensor_list(act_last),
        "act_resp": _tensor_list(act_resp),
    }


@app.post("/sae/features", dependencies=[Depends(_require_auth)])
def sae_features(body: dict, request: Request) -> dict:
    cfg = request.app.state.config
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or cfg.model.max_new_tokens)
    cap = int(body.get("cap") or cfg.feature_cloud.topk_candidates)
    attribution = body.get("attribution")
    if attribution is None:
        attribution = cfg.feature_cloud.rank_method == "attribution"
    contrast = body.get("contrast")
    if contrast is None:
        contrast = cfg.feature_cloud.contrast_baseline
    res = _capture(messages, max_new, attribution=attribution, cfg=cfg)
    resp_acts, _, _ = _pooled_activations(res)
    grad = res.get("grad")
    resp_grad = grad[res["resp_start"] :] if grad is not None else None
    baseline = _baseline_vec(cfg) if (attribution and contrast) else None
    candidates = _sae_candidates(res, resp_acts, resp_grad, cap=cap, baseline=baseline, cfg=cfg)
    return {"answer": res["answer"], "candidates": candidates, "reliable": True}


@app.post("/turn", dependencies=[Depends(_require_auth)])
def turn(body: dict, request: Request) -> dict:
    cfg = request.app.state.config
    messages = body.get("messages") or []
    max_new = int(body.get("max_new") or cfg.model.max_new_tokens)
    attribution = cfg.feature_cloud.rank_method == "attribution"

    _t0 = time.perf_counter()
    res = _capture(messages, max_new, attribution=attribution, cfg=cfg)
    capture_ms = (time.perf_counter() - _t0) * 1000.0

    _t1 = time.perf_counter()
    resp_acts, act_last, act_resp = _pooled_activations(res)
    grad = res.get("grad")
    resp_grad = grad[res["resp_start"] :] if grad is not None else None
    baseline = _baseline_vec(cfg) if attribution else None
    candidates = _sae_candidates(
        res, resp_acts, resp_grad, cap=cfg.feature_cloud.topk_candidates, baseline=baseline, cfg=cfg
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


def _launch_agent(tracker_id: str, cfg) -> None:
    """Run the blocking interp-agent pipeline (pod-local gemma + Claude) off the event loop."""
    from .agent.interp_agent import run_interp_agent
    from .science import concept_synth as cs

    def _run():
        try:
            run_interp_agent(tracker_id, cfg.probes.builder, cfg.anthropic_api_key, model=cfg.model)
        except Exception as e:  # noqa: BLE001 - surface failure in the job record
            msg = str(e) or f"{type(e).__name__} during probe build"
            print(f"[gpu_service] interp agent failed ({tracker_id}): {msg}")
            cs.update_job(tracker_id, status="error", error=msg)

    asyncio.create_task(asyncio.to_thread(_run))


@app.post("/api/trackers/clear-custom", dependencies=[Depends(_require_auth)])
async def clear_custom_trackers(request: Request) -> dict:
    """Remove user-built probes from live scoring and delete their artifact JSON files."""
    from .science import persona

    cfg = request.app.state.config
    removed = persona.clear_custom_trackers(cfg.probes)
    remaining = list(persona._trackers.keys())
    print(f"[gpu_service] cleared custom trackers: {', '.join(removed) or '(none)'}")
    return {"removed": removed, "trackers": remaining}


@app.post("/api/track", dependencies=[Depends(_require_auth)])
async def track(body: dict, request: Request) -> dict:
    """Submit a natural-language monitoring request. Trains a persona-vector probe in the
    background ON THIS POD (local gemma generation + Claude design/judge) and, if it clears the
    AUROC gate, registers it into THIS process's live `persona._trackers` — the same dict /turn
    scores against, so the next chat turn picks it up. Returns immediately; poll the GET below."""
    from .science import concept_synth as cs

    cfg = request.app.state.config
    if STATE["mode"] != "real":
        raise HTTPException(status_code=503, detail="model not loaded")
    if not cfg.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set on pod")
    req = (body.get("request") or body.get("concept") or body.get("name") or "").strip()
    if not req:
        raise HTTPException(status_code=400, detail="request is required")
    tracker_id = cs.create_job(req)
    _launch_agent(tracker_id, cfg)
    return {"tracker_id": tracker_id, "status": "pending"}


@app.get("/api/track/{tracker_id}", dependencies=[Depends(_require_auth)])
async def track_status(tracker_id: str, request: Request):
    from .science import concept_synth as cs

    job = cs.get_job(tracker_id)
    if job is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return job
