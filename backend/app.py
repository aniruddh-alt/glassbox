"""FastAPI backend — the A↔C seam. OWNER: Lane A."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import labels, observability, runtime, sentry_api
from .analyze import analyze_turn
from .fanout import fanout, init_sponsors, capture_cognition_alarm, sentry_enabled, _flag_reason

_TOKEN_CADENCE_S = 0.012  # replay the (already-generated) answer at a readable typing pace


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring up per-request readiness loading + the sponsor observability surfaces at startup.

    runtime.start_loading() kicks off the background model/SAE load (per-request readiness;
    requests use the synthetic fallback until it's ready). init_sponsors() is the SINGLE
    sponsor seam (contract #4): Sentry (incident view) + Phoenix (analytics view), each
    independently optional so a missing DSN or Phoenix sidecar logs a warning and the app
    still serves.
    """
    from .config import load_config

    app.state.config = load_config()
    cfg = app.state.config
    runtime.start_loading(cfg)
    init_sponsors(cfg.observability, cfg.sentry_dsn)
    yield


app = FastAPI(title="GlassBox", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/api/health")
def health(request: Request) -> dict:
    return runtime.health_payload(request.app.state.config)


@app.get("/api/observability")
async def observability_endpoint(request: Request):
    """Return the in-process store snapshot merged with health, Sentry, and Phoenix UI URL.
    Never contains prompt or response text (store holds only redacted views)."""
    cfg = request.app.state.config
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload(cfg)
    snap["sentry"] = {
        "emit_configured": bool(cfg.sentry_dsn),
        "configured": bool(cfg.sentry_auth_token),
        "deep_link": sentry_api.deep_link(cfg.observability.sentry, cfg.sentry_auth_token),
        "issues": await sentry_api.list_recent_issues(cfg.observability.sentry, cfg.sentry_auth_token),
    }
    snap["phoenix_ui_url"] = None  # WS1: remove this key
    return snap


@app.post("/api/observability/eval")
async def observability_eval():
    """Run a Phoenix batch coherence eval (labels-only, no prompt/response).
    coherence_eval is imported lazily so this task ships before that module exists."""
    from starlette.concurrency import run_in_threadpool
    from . import coherence_eval  # noqa: PLC0415 — intentionally lazy
    return await run_in_threadpool(coherence_eval.run_eval)


@app.post("/api/observability/test-sentry")
async def observability_test_sentry(request: Request):
    """Fire a synthetic flagged-turn alarm through Sentry (no GPU, no PHI).

    Use this to verify SENTRY_DSN wiring. Real chat turns alarm automatically via fanout()
    whenever any probe sets event.flag=true."""
    import time
    import uuid

    cfg = request.app.state.config
    if not sentry_enabled(cfg.sentry_dsn):
        return JSONResponse(
            {"ok": False, "reason": "SENTRY_DSN not configured — set it in .env and restart the backend."},
            status_code=503,
        )
    message_id = f"test-{uuid.uuid4().hex[:8]}"
    event = {
        "message_id": message_id,
        "ts": time.time(),
        "model": "glassbox-test",
        "layer": 17,
        "flag": True,
        "severity": "warning",
        "uncertainty": 0.91,
        "uncertainty_proj": 1.2,
        "trackers": {
            "over_confidence": {"score": 0.91, "proj": 1.2, "flag": True, "reliable": True},
            "harmful": {"score": 0.12, "flag": False, "reliable": True},
        },
        "features": [{"index": 0, "label": "synthetic test alarm (no PHI)", "act": 1.0}],
        "io": {"user_msg": "[synthetic test — not a real patient]", "response": "[synthetic test]"},
    }
    sent = capture_cognition_alarm(event, cfg.observability, flush=True)
    return {
        "ok": sent,
        "message_id": message_id,
        "flag_reason": _flag_reason(event),
        "hint": "Check Sentry Issues for 'Confident-wrong answer'. Chat turns alarm the same way when a probe flags.",
    }


@app.post("/api/observability/replay-sentry")
async def observability_replay_sentry(request: Request, body: dict | None = None):
    """Re-emit Sentry for a flagged turn already in the in-process store (e.g. if an alarm was missed)."""
    cfg = request.app.state.config
    if not sentry_enabled(cfg.sentry_dsn):
        return JSONResponse({"ok": False, "reason": "SENTRY_DSN not configured"}, status_code=503)
    message_id = (body or {}).get("message_id")
    if message_id:
        view = observability.STORE.get_turn(message_id)
    else:
        flagged = observability.STORE.snapshot()["confident_wrong"]
        view = observability.STORE.get_turn(flagged[-1]["message_id"]) if flagged else None
    if view is None or not view.get("flag"):
        return JSONResponse({"ok": False, "reason": "turn not found or not flagged"}, status_code=404)
    sent = capture_cognition_alarm({**view, "io": {}}, cfg.observability, flush=True)
    return {"ok": sent, "message_id": view["message_id"], "flag_reason": _flag_reason(view)}


def _chunks(text: str) -> list[str]:
    """Split into word-with-trailing-space chunks for the streamed typing effect."""
    return re.findall(r"\S+\s*", text) or [text]


@app.post("/api/chat")
async def chat(request: Request, body: dict):
    """Generate a turn, stream status + token replay, then one CognitionEvent line."""
    cfg = request.app.state.config
    messages = body.get("messages") or []
    turn_start_ns = time.time_ns()

    async def gen():
        yield json.dumps({"type": "status", "text": "Generating response and running probes…"}) + "\n"
        answer, event, perf = await run_in_threadpool(analyze_turn, messages, cfg)
        perf["t0_ns"] = turn_start_ns
        payload = event.model_dump()
        try:
            fanout(payload, perf, obs=cfg.observability, probes=cfg.probes)
        except Exception as e:
            print(f"[app] fanout failed: {e}")
        for chunk in _chunks(answer):
            yield json.dumps(
                {"type": "token", "text": chunk, "top_features": [], "uncertainty": None}
            ) + "\n"
            await asyncio.sleep(_TOKEN_CADENCE_S)
        yield json.dumps(payload) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze(request: Request, body: dict):
    """Post-hoc / non-streaming variant: returns the CognitionEvent as JSON."""
    cfg = request.app.state.config
    messages = body.get("messages") or []
    turn_start_ns = time.time_ns()
    _, event, perf = await run_in_threadpool(analyze_turn, messages, cfg)
    perf["t0_ns"] = turn_start_ns
    payload = event.model_dump()
    try:
        fanout(payload, perf, obs=cfg.observability, probes=cfg.probes)
    except Exception as e:
        print(f"[app] fanout failed: {e}")
    return JSONResponse(payload)


@app.post("/api/track")
async def track(request: Request, body: dict):
    """Proxy an NL monitoring request to the GPU pod."""
    from . import pod_client

    cfg = request.app.state.config
    req_str = body.get("request") or body.get("concept") or body.get("name") or ""
    try:
        return await run_in_threadpool(pod_client.track, req_str, cfg.pod, cfg.pod_token)
    except Exception as e:  # noqa: BLE001
        print(f"[app] track proxy failed: {e}")
        return JSONResponse(pod_client.pod_unavailable_payload(e), status_code=503)


@app.post("/api/trackers/clear-custom")
async def clear_custom_trackers(request: Request):
    """Proxy custom-probe cleanup to the GPU pod."""
    from . import pod_client

    cfg = request.app.state.config
    try:
        return await run_in_threadpool(pod_client.clear_custom_trackers, cfg.pod, cfg.pod_token)
    except Exception as e:  # noqa: BLE001
        print(f"[app] clear_custom_trackers proxy failed: {e}")
        return JSONResponse(pod_client.pod_unavailable_payload(e), status_code=503)


@app.get("/api/track/{tracker_id}")
async def track_status(request: Request, tracker_id: str):
    """Proxy probe-job status from the pod."""
    from . import pod_client

    cfg = request.app.state.config
    try:
        return await run_in_threadpool(pod_client.track_status, tracker_id, cfg.pod, cfg.pod_token)
    except Exception as e:  # noqa: BLE001
        print(f"[app] track_status proxy failed: {e}")
        return JSONResponse(pod_client.pod_unavailable_payload(e), status_code=503)


@app.get("/api/feature/{index}")
async def feature(request: Request, index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    cfg = request.app.state.config
    return {
        "index": index,
        "label": labels.get_label(index, cfg.sae, cfg.feature_cloud, cfg.anthropic_api_key, np_source=cfg.np_source()),
        "source": cfg.np_source(),
        "caveat": "auto-interp label, may be unreliable",
    }
