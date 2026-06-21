"""FastAPI backend — the A↔C seam. OWNER: Lane A."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import config, labels, observability, runtime, sentry_api
from .analyze import analyze_turn
from .fanout import fanout, init_sponsors

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
    runtime.start_loading()
    init_sponsors()
    yield


app = FastAPI(title="GlassBox", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/api/health")
def health() -> dict:
    return runtime.health_payload()


@app.get("/api/observability")
async def observability_endpoint():
    """Return the in-process store snapshot merged with health, Sentry, and Phoenix UI URL.
    Never contains prompt or response text (store holds only redacted views)."""
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload()
    snap["sentry"] = {
        "configured": bool(config.SENTRY_AUTH_TOKEN),
        "deep_link": sentry_api.deep_link(),
        "issues": await sentry_api.list_recent_issues(),
    }
    snap["phoenix_ui_url"] = config.PHOENIX_UI_URL
    return snap


@app.post("/api/observability/eval")
async def observability_eval():
    """Run a Phoenix batch coherence eval (labels-only, no prompt/response).
    coherence_eval is imported lazily so this task ships before that module exists."""
    from starlette.concurrency import run_in_threadpool
    from . import coherence_eval  # noqa: PLC0415 — intentionally lazy
    return await run_in_threadpool(coherence_eval.run_eval)


def _chunks(text: str) -> list[str]:
    """Split into word-with-trailing-space chunks for the streamed typing effect."""
    return re.findall(r"\S+\s*", text) or [text]


@app.post("/api/chat")
async def chat(body: dict):
    """Generate (or synthesize) a turn, stream token lines, then exactly one event line.
    fanout() runs AFTER the event line so nothing blocks the stream."""
    messages = body.get("messages") or []
    turn_start_ns = time.time_ns()
    answer, event, perf = await run_in_threadpool(analyze_turn, messages)
    perf["t0_ns"] = turn_start_ns
    payload = event.model_dump()

    async def gen():
        for chunk in _chunks(answer):
            yield json.dumps(
                {"type": "token", "text": chunk, "top_features": [], "uncertainty": None}
            ) + "\n"
            await asyncio.sleep(_TOKEN_CADENCE_S)
        yield json.dumps(payload) + "\n"
        try:
            fanout(payload, perf)
        except Exception as e:  # never let a sponsor error break the completed stream
            print(f"[app] fanout failed: {e}")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze(body: dict):
    """Post-hoc / non-streaming variant: returns the CognitionEvent as JSON."""
    messages = body.get("messages") or []
    _, event, _perf = await run_in_threadpool(analyze_turn, messages)
    return JSONResponse(event.model_dump())


@app.post("/api/track")
async def track(body: dict):
    """Proxy an NL monitoring request to the GPU pod, which trains and registers the probe where
    the model and live trackers live (gpu_service /api/track). Non-fatal: a missing or unreachable
    pod returns an 'unavailable' status instead of crashing the request."""
    from . import pod_client

    request = body.get("request") or body.get("concept") or body.get("name") or ""
    try:
        return await run_in_threadpool(pod_client.track, request)
    except Exception as e:  # noqa: BLE001 - pod down / not configured
        print(f"[app] track proxy failed: {e}")
        return JSONResponse({"status": "unavailable"}, status_code=503)


@app.get("/api/track/{tracker_id}")
async def track_status(tracker_id: str):
    """Proxy probe-job status from the pod."""
    from . import pod_client

    try:
        return await run_in_threadpool(pod_client.track_status, tracker_id)
    except Exception as e:  # noqa: BLE001 - pod down / not configured
        print(f"[app] track_status proxy failed: {e}")
        return JSONResponse({"status": "unavailable"}, status_code=503)


@app.get("/api/feature/{index}")
async def feature(index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    return {
        "index": index,
        "label": labels.get_label(index),
        "source": config.NP_SOURCE,
        "caveat": "auto-interp label, may be unreliable",
    }
