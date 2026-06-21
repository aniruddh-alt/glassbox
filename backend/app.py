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


@app.post("/api/track")
async def track(body: dict):
    """User-defined concept (WIP — synth_concept unimplemented). Returns a stub status."""
    return {"tracker_id": body.get("concept", "concept"), "status": "computing"}


@app.get("/api/track/{tracker_id}")
async def track_status(tracker_id: str):
    return {"status": "ready", "auroc": None}


@app.get("/api/feature/{index}")
async def feature(index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    return {
        "index": index,
        "label": labels.get_label(index),
        "source": config.NP_SOURCE,
        "caveat": "auto-interp label, may be unreliable",
    }
