"""End-to-end smoke test for the Interpretability Agent — real gemma + real Claude.

    DEVICE=mps ANTHROPIC_API_KEY=... .venv/bin/python -u -m backend.smoke_test_agent

Runs the full NL-request -> dataset -> probe -> guardrail loop on ONE concept with a
small question budget, then prints the verdict, AUROC, and whether a tracker registered.
"""
from __future__ import annotations

import os

from backend import engine
from backend.agent import interp_agent
from backend.config import load_config
from backend.science import concept_synth as cs
from backend.science import persona

REQUEST = os.getenv("REQUEST", "Watch for the model being sycophantic toward the clinician.")


def main():
    cfg = load_config()
    engine.load_engine(cfg.model, cfg.resolve_device())
    tid = cs.create_job(REQUEST)
    print(f"[agent] tracker_id={tid}  request={REQUEST!r}")
    job = interp_agent.run_interp_agent(
        tid, cfg.probes.builder, cfg.anthropic_api_key, model=cfg.model
    )
    print(f"[agent] status={job['status']}  trait={job['trait_name']}  "
          f"AUROC={job['auroc']}  baseline={job['baseline_auroc']}  n_kept={job['n_kept']}")
    print(f"[agent] verdict: {job['verdict']}")
    print(f"[agent] registered as live guardrail: {tid in persona._trackers}")
    if job["status"] == "ready":
        print("[OK] full loop green — concept is now scored on every chat message.")
    elif job["status"] == "rejected":
        print("[OK] loop ran; concept not linearly separable enough to deploy (honest reject).")
    else:
        print(f"[WARN] ended in status={job['status']} ({job.get('error')})")


if __name__ == "__main__":
    main()
