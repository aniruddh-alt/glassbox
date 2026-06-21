"""Latency breakdown for the GlassBox per-message pipeline (post-hoc mode).

DEVICE=mps .venv/bin/python -u -m backend.bench_latency
"""

from __future__ import annotations

import time

import torch

from backend import config, engine, labels
from backend.science import sae as sae_mod

PROMPT = "Is it safe to take ibuprofen during the third trimester of pregnancy? Answer in one sentence."


def t():
    return time.time()


def main():
    dev = config.resolve_device()
    print(f"[device] {dev}   model={config.MODEL_ID}   layer={config.LAYER}\n")

    t0 = t()
    engine.load_engine()
    print(f"model load:          {t() - t0:6.2f} s   (one-time)")
    t0 = t()
    sae_mod.load_sae()
    print(f"SAE load:            {t() - t0:6.2f} s   (one-time)\n")

    # generation + the post-hoc full forward (post-hoc mode)
    t0 = t()
    r = engine.generate_and_capture([{"role": "user", "content": PROMPT}], max_new=48)
    dt = t() - t0
    acts, resp_start = r["acts"], r["resp_start"]
    n_new = r["out_ids"].shape[0] - resp_start
    print(
        f"generate+posthoc:    {dt:6.2f} s   ({n_new} tok ~ {n_new / dt:.1f} tok/s)  <- dominates"
    )

    # one isolated full forward (the post-hoc cost; live mode skips this)
    ids = r["out_ids"].unsqueeze(0).to(next(engine._model.parameters()).device)
    t0 = t()
    with torch.no_grad():
        engine._model(input_ids=ids)
    print(
        f"  posthoc forward:   {(t() - t0) * 1000:6.0f} ms   (1 forward; live mode skips)\n"
    )

    # --- the interpretability overhead (the whole question of 'does interp slow it down') ---
    n = acts.shape[0] - resp_start
    t0 = t()
    for pos in range(resp_start, acts.shape[0]):
        sae_mod.sae_topk(acts[pos], k=config.TOPK)
    dt = t() - t0
    print(
        f"SAE top-k x{n:>3} tok:  {dt * 1000:6.1f} ms   ({dt / max(n, 1) * 1000:.2f} ms/token)"
    )

    d = torch.randn(config.D_IN)
    respavg = acts[resp_start:].float().mean(0).cpu()
    t0 = t()
    for _ in range(200):
        float(respavg @ d)
    print(
        f"probe projection:    {(t() - t0) / 200 * 1000:6.3f} ms   (per concept; ~nothing)\n"
    )

    # --- labels (the network wildcard) ---
    idx = sae_mod.sae_topk(acts[resp_start], k=1)[0]["index"]
    labels._cache.pop(idx, None)
    t0 = t()
    labels.get_label(idx)
    print(
        f"label fetch (cold):  {(t() - t0) * 1000:6.0f} ms   (network; PRE-WARM for demo)"
    )
    t0 = t()
    labels.get_label(idx)
    print(f"label fetch (warm):  {(t() - t0) * 1000:6.3f} ms   (cached)")


if __name__ == "__main__":
    main()
