"""FastAPI backend — the A↔C seam. OWNER: Lane A."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import config, labels, runtime
from .analyze import analyze_turn
from .fanout import fanout, init_sponsors
from .science import concept_synth as _cs

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


def _chunks(text: str) -> list[str]:
    """Split into word-with-trailing-space chunks for the streamed typing effect."""
    return re.findall(r"\S+\s*", text) or [text]


@app.post("/api/chat")
async def chat(body: dict):
    """Generate (or synthesize) a turn, stream token lines, then exactly one event line.
    fanout() runs AFTER the event line so nothing blocks the stream."""
    messages = body.get("messages") or []
    answer, event = await run_in_threadpool(analyze_turn, messages)
    payload = event.model_dump()

    async def gen():
        for chunk in _chunks(answer):
            yield json.dumps(
                {"type": "token", "text": chunk, "top_features": [], "uncertainty": None}
            ) + "\n"
            await asyncio.sleep(_TOKEN_CADENCE_S)
        yield json.dumps(payload) + "\n"
        try:
            fanout(payload)
        except Exception as e:  # never let a sponsor error break the completed stream
            print(f"[app] fanout failed: {e}")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze(body: dict):
    """Post-hoc / non-streaming variant: returns the CognitionEvent as JSON."""
    messages = body.get("messages") or []
    _, event = await run_in_threadpool(analyze_turn, messages)
    return JSONResponse(event.model_dump())


def _launch_agent(tracker_id: str) -> None:
    """Run the blocking agent pipeline off the event loop."""
    from .agent.interp_agent import run_interp_agent

    def _run():
        try:
            run_interp_agent(tracker_id)
        except Exception as e:  # noqa: BLE001 - surface failure in the job record
            _cs.update_job(tracker_id, status="error", error=str(e))

    asyncio.create_task(asyncio.to_thread(_run))


@app.post("/api/track")
async def track(body: dict):
    """Submit a natural-language monitoring request. Returns immediately; runs in the bg."""
    request = body.get("request") or body.get("concept") or body.get("name") or ""
    tracker_id = _cs.create_job(request)
    _launch_agent(tracker_id)
    return {"tracker_id": tracker_id, "status": "pending"}


@app.get("/api/track/{tracker_id}")
async def track_status(tracker_id: str):
    job = _cs.get_job(tracker_id)
    if job is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return job


@app.get("/api/feature/{index}")
async def feature(index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    return {
        "index": index,
        "label": labels.get_label(index),
        "source": config.NP_SOURCE,
        "caveat": "auto-interp label, may be unreliable",
    }
