"""The five tools the Interpretability Agent calls, plus a dispatch() that executes
them against a per-job context dict and keeps the job record in sync."""
from __future__ import annotations

from .. import config
from ..science import concept_synth as cs
from ..science import persona
from . import prompts

TOOLS = [
    prompts.SPEC_TOOL,
    {"name": "generate_contrastive", "description": "Run the model under the pos/neg prompts; capture activations.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "judge_filter", "description": "Keep only responses whose behavior matched the intended side.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "fit_and_validate", "description": "Train the probe; measure held-out AUROC vs a baseline.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "finalize", "description": "Record the plain-language verdict; deployment is decided by the AUROC gate.",
     "input_schema": {"type": "object", "properties": {"verdict": {"type": "string"}},
                      "required": ["verdict"], "additionalProperties": False}},
]


def dispatch(name: str, tool_input: dict, ctx: dict) -> str:
    tid = ctx["tracker_id"]
    if name == "submit_spec":
        ctx["spec"] = tool_input
        cs.update_job(tid, status="designing", trait_name=tool_input["trait_name"],
                      progress={"step": "spec", "pct": 20})
        return f"Spec stored: {len(tool_input['questions'])} questions. Call generate_contrastive."
    if name == "generate_contrastive":
        cs.update_job(tid, status="generating", progress={"step": "generating", "pct": 40})
        ctx["rows"] = cs.generate_contrastive(ctx["spec"], generate_fn=ctx["generate_fn"])
        return f"Generated {len(ctx['rows'])} responses. Call judge_filter."
    if name == "judge_filter":
        cs.update_job(tid, status="judging", progress={"step": "judging", "pct": 60})
        ctx["rows"] = cs.judge_filter(ctx["spec"], ctx["rows"], client=ctx["client"])
        return f"Kept {len(ctx['rows'])} clean rows. Call fit_and_validate."
    if name == "fit_and_validate":
        cs.update_job(tid, status="fitting", progress={"step": "fitting", "pct": 80})
        ctx["fit"] = cs.fit_and_validate(ctx["rows"])
        f = ctx["fit"]
        cs.update_job(tid, auroc=f["auroc"], baseline_auroc=f["baseline_auroc"], n_kept=f["n_kept"])
        if f["status"] != "ok":
            return "insufficient_data: too few clean rows or a class collapsed. Call finalize explaining this."
        return (f"held-out AUROC={f['auroc']:.2f} (baseline {f['baseline_auroc']:.2f}), "
                f"tau={config.TRACK_AUROC_TAU}. Call finalize with your verdict.")
    if name == "finalize":
        return _finalize(ctx, tool_input["verdict"])
    return f"unknown tool: {name}"


def _finalize(ctx: dict, verdict: str) -> str:
    tid = ctx["tracker_id"]
    fit = ctx.get("fit") or {}
    deployed = fit.get("status") == "ok" and (fit.get("auroc") or 0.0) >= config.TRACK_AUROC_TAU
    if deployed:
        persona._trackers[tid] = {
            "dir": fit["direction"],
            "calibrator": fit["calibrator"],
            "threshold": fit["threshold"],
            "meta": {"user_defined": True, "auroc": fit["auroc"],
                     "request": ctx["request"], "reliability": "synthetic-validated"},
        }
    cs.update_job(tid, status="ready" if deployed else "rejected", verdict=verdict,
                  progress={"step": "done", "pct": 100})
    return "deployed as a live guardrail." if deployed else "not deployed (gate not met)."
