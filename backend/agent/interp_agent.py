"""The Interpretability Agent's Claude tool-use loop. Blocking (gemma + sync Claude);
run via asyncio.to_thread so it never blocks the FastAPI event loop."""
from __future__ import annotations

from .. import config
from ..science import concept_synth as cs
from . import prompts, tools

_MAX_TURNS = 12


def run_interp_agent(tracker_id: str, *, client=None, generate_fn=None) -> dict | None:
    job = cs.get_job(tracker_id)
    if job is None:
        raise ValueError(f"unknown tracker_id: {tracker_id}")
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # generate_fn may be None; the None→engine default happens inside
    # concept_synth.generate_contrastive, keeping agent/ free of any torch/engine import.
    ctx = {"tracker_id": tracker_id, "request": job["request"],
           "client": client, "generate_fn": generate_fn,
           "spec": None, "rows": None, "fit": None}
    messages = [{"role": "user", "content": f"Monitor request: {job['request']}"}]

    for _ in range(_MAX_TURNS):
        resp = client.messages.create(
            model=config.AGENT_MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=prompts.SYSTEM,
            tools=tools.TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = tools.dispatch(block.name, block.input, ctx)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        if not results:
            break
        messages.append({"role": "user", "content": results})
        if cs.get_job(tracker_id)["status"] in ("ready", "rejected"):
            break

    job = cs.get_job(tracker_id)
    if job is not None and job["status"] not in ("ready", "rejected", "error"):
        step = (job.get("progress") or {}).get("step", job["status"])
        cs.update_job(
            tracker_id,
            status="error",
            error=f"agent did not finalize within MAX_TURNS (last step: {step})",
        )
    return cs.get_job(tracker_id)
