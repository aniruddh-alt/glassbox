"""Model load + the single layer-LAYER forward hook + decode.

OWNER: Lane A. The layer activation tensor is the ONLY object that crosses the
GPU→science boundary (contract #3). GPU work only — no sponsor SDKs here.

Gemma-3-aware: handles the multimodal Gemma3ForConditionalGeneration nesting
(text decoder at model.language_model.layers), the CausalLM/ImageTextToText loader
split, and the transformers-5.x bare-tensor layer output.
"""

from __future__ import annotations

from . import config

_cap: dict = {"act": None}  # newest capture, written by the forward hook
_tok = None
_model = None
_layers = None  # the text-decoder ModuleList
_layers_path = None


def _find_decoder_layers(model):
    """Find the text-decoder ModuleList regardless of multimodal nesting."""
    import torch.nn as nn

    cands = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) > 0 and n.split(".")[-1] == "layers"
    ]
    if not cands:
        raise RuntimeError("no decoder layer list found")
    lang = [c for c in cands if "language" in c[0] or "text" in c[0]]
    return max(lang or cands, key=lambda c: len(c[1]))


def load_engine(device: str | None = None):
    """Load the model (config.MODEL_ID), register ONE hook on the decoder layer config.LAYER."""
    global _tok, _model, _layers, _layers_path
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    dev = config.resolve_device(device)
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    _tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    try:
        _model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, dtype=dtype)
    except Exception:
        _model = AutoModelForImageTextToText.from_pretrained(
            config.MODEL_ID, dtype=dtype
        )
    _model = _model.to(dev).eval()

    _layers_path, _layers = _find_decoder_layers(_model)

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        _cap["act"] = hs.detach()

    _layers[config.LAYER].register_forward_hook(hook)
    return _tok, _model


def get_layers():
    return _layers


def info() -> dict:
    return {
        "model": config.MODEL_ID,
        "layers_path": _layers_path,
        "n_layers": (len(_layers) if _layers is not None else None),
        "hook_layer": config.LAYER,
    }


def generate_and_capture(messages: list[dict], max_new: int = 48) -> dict:
    """Generate the answer, then one post-hoc forward to capture ALL response positions at LAYER.
    Returns {answer, acts:[seq,d_in], out_ids, resp_start, tok}."""
    import torch

    dev = next(_model.parameters()).device
    enc = _tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    ids = enc["input_ids"]
    with torch.no_grad():
        out = _model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    answer = _tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    with torch.no_grad():
        _model(input_ids=out)  # post-hoc full forward -> _cap holds [1, seq, d_in]
    return {
        "answer": answer,
        "acts": _cap["act"][0],
        "out_ids": out[0],
        "resp_start": ids.shape[1],
        "tok": _tok,
    }
