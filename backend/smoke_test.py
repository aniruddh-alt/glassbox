"""GlassBox Family-A smoke test — exercises the REAL stack on the locked defaults.

    engine.load_engine() -> generate -> hook layer config.LAYER -> SAE top-k -> Neuronpedia labels

Run from repo root:
    DEVICE=mps .venv/bin/python -u -m backend.smoke_test

Reads everything from config.py (now Gemma Scope 2 + gemma-3-4b-it). Override e.g. LAYER=22.
"""

from __future__ import annotations

import os

from backend import config, engine, labels
from backend.science import sae as sae_mod

PROMPT = os.getenv(
    "PROMPT",
    "Is it safe to take ibuprofen during the third trimester of pregnancy? Answer in one sentence.",
)
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "48"))


def main():
    print(
        f"[cfg] model={config.MODEL_ID} layer={config.LAYER} sae={config.SAE_RELEASE}/{config.SAE_ID}"
    )
    engine.load_engine()
    print(f"[engine] {engine.info()}")
    sae_mod.load_sae()
    print(f"[sae] loaded d_in={config.D_IN}")

    r = engine.generate_and_capture(
        [{"role": "user", "content": PROMPT}], max_new=MAX_NEW
    )
    acts, resp_start, ids_full, tok = (
        r["acts"],
        r["resp_start"],
        r["out_ids"].tolist(),
        r["tok"],
    )
    print(f"[answer] {r['answer']!r}")
    print(f"[hook] residual {tuple(acts.shape)} (expect [seq, {config.D_IN}])")

    probe_pos = min(resp_start + 2, acts.shape[0] - 1)
    rec = sae_mod.reconstruction_error(acts[probe_pos])
    ok = "OK" if rec["cosine"] > 0.85 else "!! LOW (layer/dtype?)"
    print(
        f"[sanity] SAE reconstruction cosine={rec['cosine']:.3f} rel_mse={rec['rel_mse']:.3f} [{ok}]"
    )

    special = {tok.convert_tokens_to_ids(t) for t in config.MASK_TOKENS}
    best: dict[int, float] = {}
    for pos in range(resp_start, acts.shape[0]):
        if ids_full[pos] in special:
            continue
        for f in sae_mod.sae_topk(acts[pos], k=config.TOPK):
            best[f["index"]] = max(best.get(f["index"], 0.0), f["act"])
    top = sorted(best.items(), key=lambda kv: -kv[1])[: config.TOPK_EVENT]
    print(f"[cloud] {len(best)} unique features; top 12 (Gemma-Scope-2 labels):")
    for i, v in top[:12]:
        print(f"   #{i:<6} act={v:7.2f}   {labels.get_label(i)}")

    print(
        "\n[OK] Family-A stack works on the new defaults (Gemma Scope 2 + gemma-3-4b)."
    )


if __name__ == "__main__":
    main()
