"""Model load + the single layer-LAYER forward hook + decode.

OWNER: Lane A. The layer activation tensor is the ONLY object that crosses the
GPU→science boundary (contract #3). GPU work only — no sponsor SDKs here.

Gemma-3-aware: handles the multimodal Gemma3ForConditionalGeneration nesting
(text decoder at model.language_model.layers), the CausalLM/ImageTextToText loader
split, and the transformers-5.x bare-tensor layer output.
"""

from __future__ import annotations

from .config import ModelConfig, SAEConfig

_cap: dict = {"act": None}  # newest capture, written by the forward hook
_grad_mode = False  # when True the hook keeps the live (graph-attached) tensor for attribution
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


def _resolve_device(model_cfg: ModelConfig) -> str:
    """Inline device resolution for when no AppConfig is available (direct GPU callers).
    Mirrors AppConfig.resolve_device — if device is explicitly 'cpu' return 'cpu',
    otherwise prefer cuda > mps > cpu."""
    import torch

    p = (model_cfg.device or "auto").lower()
    if p == "cuda" and torch.cuda.is_available():
        return "cuda"
    if p == "mps" and torch.backends.mps.is_available():
        return "mps"
    if p == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_engine(model: ModelConfig, device: str | None = None):
    """Load the model (model.model_id), register ONE hook on the decoder layer model.layer."""
    global _tok, _model, _layers, _layers_path
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    dev = device if device is not None else _resolve_device(model)
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    _tok = AutoTokenizer.from_pretrained(model.model_id)
    try:
        _model = AutoModelForCausalLM.from_pretrained(model.model_id, dtype=dtype)
    except Exception:
        _model = AutoModelForImageTextToText.from_pretrained(
            model.model_id, dtype=dtype
        )
    _model = _model.to(dev).eval()

    _layers_path, _layers = _find_decoder_layers(_model)

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        # Attribution mode keeps the live (graph-attached) tensor so its gradient is reachable;
        # otherwise store a detached copy (cheap, retains no autograd graph).
        _cap["act"] = hs if _grad_mode else hs.detach()

    _layers[model.layer].register_forward_hook(hook)
    return _tok, _model


def _inject_system(messages: list[dict], model: ModelConfig) -> list[dict]:
    """Prepend model.system_prompt as a leading system turn, unless one is already present or
    the prompt is empty."""
    sys = (model.system_prompt or "").strip()
    if not sys or (messages and messages[0].get("role") == "system"):
        return messages
    return [{"role": "system", "content": sys}, *messages]


def _merge_system_into_user(messages: list[dict], model: ModelConfig) -> list[dict]:
    """Fallback for chat templates that reject a system role (some Gemma builds): fold the
    system prompt into the first user turn so the model still receives it."""
    sys = (model.system_prompt or "").strip()
    if not sys:
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "user":
            m["content"] = f"{sys}\n\n{m.get('content', '')}"
            return out
    return [{"role": "user", "content": sys}, *out]


def _encode(messages: list[dict], dev, model: ModelConfig) -> dict:
    """Apply the chat template with the system prompt injected, then move to device. Falls back to
    merging the system text into the first user turn if the template has no system role."""
    try:
        enc = _tok.apply_chat_template(
            _inject_system(messages, model), add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:  # noqa: BLE001 — template rejects system role; merge into the first user turn
        enc = _tok.apply_chat_template(
            _merge_system_into_user(messages, model), add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    return {k: v.to(dev) for k, v in enc.items()}


def get_layers():
    return _layers


def info(model: ModelConfig, sae: SAEConfig) -> dict:
    return {
        "model": model.model_id,
        "layers_path": _layers_path,
        "n_layers": (len(_layers) if _layers is not None else None),
        "hook_layer": model.layer,
    }


def _acts_for_sequence(out) -> "torch.Tensor":
    """Return layer-LAYER resid_post for every token in `out`, shape [seq, d_in].

    Autoregressive generate() only leaves the hook holding the last decode step
    ([1, 1, d_in]). A full-sequence forward is required before indexing by
    resp_start. Also re-runs when attribution left a shorter capture."""
    import torch

    seq_len = int(out.shape[1])
    stored = _cap.get("act")
    if stored is not None and stored.shape[1] >= seq_len:
        return stored[0]
    with torch.no_grad():
        _model(input_ids=out, use_cache=False)
    stored = _cap.get("act")
    if stored is None or stored.shape[1] < seq_len:
        got = 0 if stored is None else int(stored.shape[1])
        raise RuntimeError(
            f"activation capture length mismatch: need {seq_len} positions, hook saw {got}"
        )
    return stored[0]


def generate_and_capture(
    messages: list[dict], max_new: int = 48, *, attribution: bool | None = None,
    model: ModelConfig | None = None, feature_cloud=None,
) -> dict:
    """Generate the answer, then capture LAYER resid_post for ALL response positions.

    Returns {answer, acts:[seq,d_in], grad:[seq,d_in]|None, out_ids, resp_start, tok}.
    When attribution is on (default: feature_cloud.rank_method == 'attribution' if provided),
    a single grad-enabled forward + one backward of a response-logit metric also yields
    grad = dL/d(resid_post), which drives attribution ranking. Any failure (e.g. OOM) degrades
    to the plain no-grad capture."""
    import torch

    if attribution is None:
        if feature_cloud is not None:
            attribution = feature_cloud.rank_method == "attribution"
        else:
            attribution = True  # default to attribution when no config provided

    dev = next(_model.parameters()).device
    enc = _encode(messages, dev, model) if model is not None else _encode_legacy(messages, dev)
    ids = enc["input_ids"]
    with torch.no_grad():
        out = _model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    answer = _tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    resp_start = ids.shape[1]

    grad = None
    if attribution:
        try:
            grad = _capture_grad(out, resp_start)
        except Exception as e:  # noqa: BLE001 — never break a turn; fall back to activation ranking
            print(f"[engine] attribution backward failed ({e}); using activation ranking")
            grad = None
    acts = _acts_for_sequence(out)

    return {
        "answer": answer,
        "acts": acts,
        "grad": grad,
        "out_ids": out[0],
        "resp_start": resp_start,
        "tok": _tok,
    }


def _encode_legacy(messages: list[dict], dev) -> dict:
    """Legacy encode without model config — no system prompt injection."""
    enc = _tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return {k: v.to(dev) for k, v in enc.items()}


def _capture_grad(out, resp_start: int):
    """One grad-enabled forward over the generated ids + one backward of a response-logit metric.
    Returns dL/d(resid_post) as [seq, d_in] and leaves _cap['act'] holding the detached acts.

    Metric L = sum over response-predicting positions of (logit[generated_token] - mean logit) —
    Goodfire's logit-difference: how strongly the model preferred each token it actually emitted."""
    import torch

    global _grad_mode
    seq = out.shape[1]
    if seq <= resp_start:  # no generated response tokens to attribute to
        return None
    _grad_mode = True
    try:
        out_obj = _model(input_ids=out)  # grad-enabled: hook stashes the live resid tensor
        hs = _cap["act"]                 # [1, seq, d_in], part of this forward's graph
        logits = out_obj.logits[0]       # [seq, vocab]
        pos = torch.arange(resp_start - 1, seq - 1, device=logits.device)
        tgt = out[0][pos + 1]            # the token actually generated at each next position
        sel = logits.index_select(0, pos)
        chosen = sel.gather(1, tgt.unsqueeze(1)).squeeze(1)
        metric = (chosen - sel.mean(dim=1)).sum()
        (g,) = torch.autograd.grad(metric, hs)  # dL/d(resid_post), [1, seq, d_in]
    finally:
        _grad_mode = False
        _cap["act"] = _cap["act"].detach()  # detach for downstream encode/display
    return g[0].detach()


def probe_activation(text: str, model: ModelConfig, pos: int | None = None):
    """One no-grad forward over a probe prompt; return a single-position resid_post [d_in].
    Used by the startup reconstruction-error wiring check (runtime._run_recon_check)."""
    import torch

    dev = next(_model.parameters()).device
    enc = _tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        _model(input_ids=enc["input_ids"])
    acts = _cap["act"][0]  # [seq, d_in]
    i = (acts.shape[0] - 1) if pos is None else pos
    return acts[i]
