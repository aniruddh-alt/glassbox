"""The five tools the Interpretability Agent calls, plus a dispatch() that executes
them against a per-job context dict and keeps the job record in sync."""
from __future__ import annotations

from ..science import concept_synth as cs
from ..science import persona
from . import prompts


def tools_for(builder) -> list[dict]:
    """Build the tool list for a Claude call. `builder` is a ProbeBuilderConfig; the
    submit_spec schema embeds builder.agent_max_questions."""
    return [
        prompts.spec_tool(builder),
        {"name": "generate_contrastive", "description": "Run the model under the pos/neg prompts; capture activations.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "judge_filter", "description": "Keep only responses whose behavior matched the intended side.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "fit_and_validate", "description": "Train the probe; measure held-out AUROC vs a baseline.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "finalize", "description": "Record the plain-language pipeline summary; deployment is decided by the AUROC gate.",
         "input_schema": {"type": "object", "properties": {"verdict": {"type": "string"}},
                          "required": ["verdict"], "additionalProperties": False}},
    ]


def dispatch(name: str, tool_input: dict, ctx: dict, builder, model) -> str:
    tid = ctx["tracker_id"]
    try:
        return _dispatch(name, tool_input, ctx, builder, model)
    except Exception as e:  # noqa: BLE001 - surface in job record + agent tool_result
        msg = str(e) or type(e).__name__
        cs.update_job(tid, status="error", error=f"{name}: {msg}")
        return f"ERROR: {msg}"


def _dispatch(name: str, tool_input: dict, ctx: dict, builder, model) -> str:
    tid = ctx["tracker_id"]
    if name == "submit_spec":
        ctx["spec"] = tool_input
        cap = builder.agent_max_questions
        if cap:
            ctx["spec"]["questions"] = ctx["spec"]["questions"][:cap]
        cs.update_job(tid, status="designing", trait_name=tool_input["trait_name"],
                      progress={"step": "spec", "pct": 20})
        return f"Spec stored: {len(ctx['spec']['questions'])} questions. Call generate_contrastive."
    if name == "generate_contrastive":
        cs.update_job(tid, status="generating", progress={"step": "generating", "pct": 40})

        def on_progress(done: int, total: int) -> None:
            pct = 40 + int(15 * done / max(total, 1))
            cs.update_job(tid, progress={"step": "generating", "pct": min(pct, 59)})

        ctx["rows"] = cs.generate_contrastive(
            ctx["spec"], generate_fn=ctx["generate_fn"], on_progress=on_progress
        )
        return f"Generated {len(ctx['rows'])} responses. Call judge_filter."
    if name == "judge_filter":
        cs.update_job(tid, status="judging", progress={"step": "judging", "pct": 60})
        ctx["rows"] = cs.judge_filter(
            ctx["spec"], ctx["rows"], builder, ctx["anthropic_api_key"], client=ctx["client"]
        )
        cs.update_job(tid, n_kept=len(ctx["rows"]))
        if not ctx["rows"]:
            return "Kept 0 clean rows after judge — call finalize explaining insufficient contrast."
        return f"Kept {len(ctx['rows'])} clean rows. Call fit_and_validate."
    if name == "fit_and_validate":
        cs.update_job(tid, status="fitting", progress={"step": "fitting", "pct": 80})
        ctx["fit"] = cs.fit_and_validate(ctx["rows"])
        f = ctx["fit"]
        cs.update_job(tid, auroc=f["auroc"], baseline_auroc=f["baseline_auroc"], n_kept=f["n_kept"])
        if f["status"] != "ok":
            return "insufficient_data: too few clean rows or a class collapsed. Call finalize explaining this."
        return (f"held-out AUROC={f['auroc']:.2f} (baseline {f['baseline_auroc']:.2f}), "
                f"tau={builder.auroc_threshold}. Call finalize with your verdict.")
    if name == "finalize":
        return _finalize(ctx, tool_input["verdict"], builder, model)
    return f"unknown tool: {name}"


def _finalize(ctx: dict, verdict: str, builder, model) -> str:
    tid = ctx["tracker_id"]
    fit = ctx.get("fit") or {}
    deployed = fit.get("status") == "ok" and (fit.get("auroc") or 0.0) >= builder.auroc_threshold
    if deployed:
        persona._trackers[tid] = {
            "dir": fit["direction"],
            "calibrator": fit["calibrator"],
            "threshold": fit["threshold"],
            "meta": {"user_defined": True, "auroc": fit["auroc"],
                     "request": ctx["request"], "reliability": "synthetic-validated"},
        }
        try:
            _persist_artifact(ctx, fit, model)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] artifact persist failed for {tid}: {e}")
    cs.update_job(tid, status="ready" if deployed else "rejected", verdict=verdict,
                  progress={"step": "done", "pct": 100})
    return "deployed as a live guardrail." if deployed else "not deployed (gate not met)."


def _persist_artifact(ctx: dict, fit: dict, model) -> None:
    """Serialize a deployed probe to backend/science/artifacts/<id>.json."""
    import json

    tid = ctx["tracker_id"]
    spec = ctx.get("spec") or {}
    direction = fit["direction"].detach().cpu()
    center, scale = _proj_calibration(ctx["rows"], direction)
    artifact = {
        "id": tid,
        "concept": spec.get("trait_name") or tid,
        "description": ctx["request"],
        "direction": direction.tolist(),
        "threshold": float(fit["threshold"]),
        "projection_center": center,
        "projection_scale": scale,
        "alert_direction": "high",
        "direction_method": "diff_of_means",
        "layer": model.layer,
        "auroc": fit.get("auroc"),
        "user_defined": True,
    }
    (persona.ARTIFACT_DIR / f"{tid}.json").write_text(json.dumps(artifact, indent=2))


def _proj_calibration(rows: list[dict], direction) -> tuple[float, float]:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

    X = torch.stack([r["act_resp"].float().cpu() for r in rows])
    proj = (X @ direction.float()).numpy().reshape(-1, 1)
    y = np.asarray([int(r["label"]) for r in rows], dtype="int64")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(proj, y)
    w = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    if w > 1e-8:
        return -b / w, 1.0 / w
    return float(proj.mean()), float(proj.std()) or 1.0
