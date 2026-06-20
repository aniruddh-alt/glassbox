"""Model load + the single layer-12 forward hook + KV-cache decode loop.

OWNER: Lane A. The layer-12 activation tensor is the ONLY object that crosses the
GPU→science boundary (contract #3). GPU work only — no sponsor SDKs here.
"""
from __future__ import annotations

from . import config

# _cap holds the newest token's residual activation, written by the forward hook.
_cap: dict = {"act": None}


def load_engine():
    """Load gemma-2-2b-it (bf16, cuda), register ONE hook on model.model.layers[LAYER]."""
    # TODO(Lane A):
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    # model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype="bfloat16", device_map=config.DEVICE)
    # def hook(_module, _inp, out):
    #     _cap["act"] = out[0][:, -1, :].detach()   # newest token under KV-cache
    # model.model.layers[config.LAYER].register_forward_hook(hook)
    #
    # SANITY (do once, hour 0-2): confirm hidden_states[LAYER] (NOT [0], that's the embedding)
    # matches SAELens hook_resid_post layer numbering before trusting family co-location.
    raise NotImplementedError("Lane A: load model + register layer-12 hook")


def generate(messages: list[dict]):
    """Yield (token_text, layer12_activation) per step; collect response-avg + last-prompt acts."""
    # TODO(Lane A): apply chat template, prefill, manual KV-cache decode loop reading _cap["act"].
    raise NotImplementedError("Lane A: KV-cache decode loop")
