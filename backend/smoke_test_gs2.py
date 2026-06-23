"""A/B smoke test — Gemma Scope 2 + Gemma 3 (vs the proven Gemma-2-2b run).

Matryoshka SAEs (cleaner features), Gemma-3-4b-it (more capable), built by DeepMind
explicitly for hallucination / sycophancy / internal-vs-stated-reasoning.

    .venv/bin/python -u -m backend.smoke_test_gs2

Defaults (override via env): unsloth/gemma-3-4b-it (ungated) · gemma-scope-2-4b-it-res ·
layer_17_width_16k_l0_medium · layer 17. Gemma 3 4b is MULTIMODAL (Gemma3ForConditionalGeneration),
so we auto-detect the loader class, the nested text-decoder layer list, and d_in.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from backend.config import load_config as _load_config

_cfg = _load_config()

MODEL_ID = os.getenv("MODEL_ID", "unsloth/gemma-3-4b-it")
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2-4b-it-res")
SAE_ID = os.getenv("SAE_ID", "layer_17_width_16k_l0_medium")
LAYER = int(os.getenv("LAYER", "17"))
NP_MODEL = os.getenv("NP_MODEL", "gemma-3-4b")
NP_SOURCE = os.getenv("NP_SOURCE", f"{LAYER}-gemmascope-2-res-16k")
PROMPT = os.getenv(
    "PROMPT",
    "Is it safe to take ibuprofen during the third trimester of pregnancy? Answer in one sentence.",
)
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "48"))


def load_model(dev, dtype):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        loader = "AutoModelForCausalLM"
    except Exception:
        m = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype)
        loader = "AutoModelForImageTextToText"
    return tok, m.to(dev).eval(), loader


def find_decoder_layers(model):
    """Find the text-decoder ModuleList regardless of multimodal nesting."""
    cands = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) > 0 and n.split(".")[-1] == "layers"
    ]
    if not cands:
        raise RuntimeError("no decoder layer list found")
    lang = [c for c in cands if "language" in c[0] or "text" in c[0]]
    name, mod = max(lang or cands, key=lambda c: len(c[1]))
    return name, mod


def get_label(index, cache):
    if index in cache:
        return cache[index]
    import httpx

    url = f"https://www.neuronpedia.org/api/feature/{NP_MODEL}/{NP_SOURCE}/{index}"
    try:
        r = httpx.get(url, headers={"User-Agent": "glassbox/0.1"}, timeout=6)
        exps = r.json().get("explanations") or [] if r.status_code == 200 else []
        cache[index] = exps[0].get("description") if exps else f"feature {index}"
    except Exception:
        cache[index] = f"feature {index}"
    return cache[index]


def main():
    dev = _cfg.resolve_device()
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    print(
        f"[device] {dev}   model={MODEL_ID}   sae={SAE_RELEASE}/{SAE_ID}   layer={LAYER}"
    )

    t0 = time.time()
    tok, model, loader = load_model(dev, dtype)
    print(f"[load] model in {time.time() - t0:.1f}s via {loader}")

    layers_name, layers = find_decoder_layers(model)
    print(
        f"[arch] decoder layers at '{layers_name}' (n={len(layers)}); hooking [{LAYER}]"
    )
    cap = {}

    def hook(_m, _i, output):
        hs = output[0] if isinstance(output, tuple) else output
        cap["act"] = hs.detach()

    handle = layers[LAYER].register_forward_hook(hook)

    from sae_lens import SAE

    loaded = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=dev)
    sae = (loaded[0] if isinstance(loaded, (tuple, list)) else loaded).to(dev).eval()
    d_in = getattr(sae.cfg, "d_in", None)
    print(f"[load] SAE d_in={d_in} d_sae={getattr(sae.cfg, 'd_sae', None)}")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    ids = enc["input_ids"]
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    n_new = out.shape[1] - ids.shape[1]
    print(
        f"[gen] {n_new} tok in {time.time() - t1:.1f}s ({n_new / max(time.time() - t1, 1e-6):.1f} tok/s)"
    )
    print(f"[answer] {tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)!r}")

    with torch.no_grad():
        model(input_ids=out)
    acts = cap["act"][0]
    print(f"[hook] captured residual {tuple(acts.shape)} (expect [seq, {d_in}])")

    def recon(a):
        a = a.detach().reshape(-1).float().to(next(sae.parameters()).device)
        with torch.no_grad():
            r = sae.decode(sae.encode(a.unsqueeze(0))).squeeze(0)
        cos = torch.nn.functional.cosine_similarity(a, r, dim=0).item()
        rel = ((a - r).pow(2).mean() / (a.pow(2).mean() + 1e-8)).item()
        return cos, rel

    resp_start = ids.shape[1]
    cos, rel = recon(acts[min(resp_start + 2, acts.shape[0] - 1)])
    ok = "OK" if cos > 0.85 else "!! LOW (layer/dtype?)"
    print(f"[sanity] reconstruction cosine={cos:.3f} rel_mse={rel:.3f} [{ok}]")

    def topk(a, k=_cfg.feature_cloud.topk):
        a = a.detach().reshape(-1).float().to(next(sae.parameters()).device)
        with torch.no_grad():
            f = sae.encode(a.unsqueeze(0)).squeeze(0)
        v, i = f.topk(k)
        return [
            (int(ii), float(vv)) for vv, ii in zip(v.tolist(), i.tolist()) if vv > 0
        ]

    special = {tok.convert_tokens_to_ids(t) for t in _cfg.model.mask_tokens}
    ids_full = out[0].tolist()
    best = {}
    for pos in range(resp_start, acts.shape[0]):
        if ids_full[pos] in special:
            continue
        for ii, vv in topk(acts[pos]):
            best[ii] = max(best.get(ii, 0.0), vv)
    top = sorted(best.items(), key=lambda kv: -kv[1])[: _cfg.feature_cloud.topk_event]
    cache = {}
    print(f"[cloud] {len(best)} unique features; top 12 with Gemma-Scope-2 labels:")
    for i, v in top[:12]:
        print(f"   #{i:<6} act={v:6.2f}   {get_label(i, cache)}")

    handle.remove()
    print(
        "\n[OK] Gemma Scope 2 + Gemma 3 end-to-end. Compare cloud cleanliness vs the Gemma-2 run."
    )


if __name__ == "__main__":
    main()
