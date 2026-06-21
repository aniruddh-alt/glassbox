"""FastAPI backend"""

from __future__ import annotations

import json
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import config

app = FastAPI(title="GlassBox")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_FIXTURE = json.loads(
    (
        pathlib.Path(__file__).parents[1] / "fixtures" / "cognition_event.sample.json"
    ).read_text()
)


@app.on_event("startup")
def _startup() -> None: ...


@app.get("/api/health")
def health() -> dict:
    # TODO(Lane A): report real load state.
    return {
        "gpu": True,
        "model_loaded": False,
        "sae_loaded": False,
        "trackers": [],
        "mode": config.MODE,
    }


@app.post("/api/chat")
async def chat(body: dict):
    """SCAFFOLD: streams the fixture as NDJSON (token lines + one event line).

    Wire protocol (contract #2): zero+ {"type":"token",...} lines, then exactly one
    {"type":"event", ...CognitionEvent}. Replace with real generation in H2-5.
    """

    async def gen():
        answer = _FIXTURE["io"]["response"]
        for tok in answer.split(" "):
            yield (
                json.dumps(
                    {
                        "type": "token",
                        "text": tok + " ",
                        "top_features": [],
                        "uncertainty": None,
                    }
                )
                + "\n"
            )
        yield json.dumps(_FIXTURE) + "\n"

    # TODO(Lane A): real path — engine.generate() → hook captures layer-12 act →
    #   sae_topk(act) + score_all_trackers(act_last, act_resp) → events.build_cognition_event() →
    #   yield token lines then the event line → fanout(event) AFTER the event line.
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze(body: dict):
    """POST-hoc mode: analyze an already-produced turn (re-forward, slice response positions)."""
    # TODO(Lane A): real path. For now echo the fixture.
    return JSONResponse(_FIXTURE)


@app.post("/api/track")
async def track(body: dict):
    """User-defined concept on demand → persona vector. Returns immediately; computes async."""
    # TODO(Lane B): from .science.concept_synth import synth_concept
    return {"tracker_id": body.get("concept", "concept"), "status": "computing"}


@app.get("/api/track/{tracker_id}")
async def track_status(tracker_id: str):
    # TODO(Lane B): poll readiness.
    return {"status": "ready", "auroc": None}


@app.get("/api/feature/{index}")
async def feature(index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    # TODO(Lane A): from .labels import get_label
    return {
        "index": index,
        "label": "(unfetched)",
        "source": config.NP_SOURCE,
        "caveat": "auto-interp label, may be unreliable",
    }
