# WS2 — BYO model + SAE (Gemma slice, config-shaped) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

Goal: Make the model + SAE config-driven within the Gemma + Gemma-Scope family by lifting the hardcoded SAE release / SAE-id pattern / Neuronpedia slug into the `sae` and `model` sub-configs, deriving `d_in`/`d_sae` from the loaded SAE at runtime with config fallback, and refreshing stale Gemma display strings — without implementing arbitrary-repo loading.

Architecture: GlassBox runs two FastAPI processes — a CPU orchestration backend (`backend/app.py`, never imports torch) and a GPU pod service (`backend/gpu_service.py`, owns torch/model/SAE). Both build one `AppConfig` at startup and thread sub-configs down to the science modules. WS2 sits on top of the WS0 `AppConfig` refactor: it consumes `cfg.sae`, `cfg.model`, `cfg.sae_id(layer)`, and `cfg.np_source(layer)` exactly as defined in the AppConfig contract, and adds a runtime SAE-dimension probe (`sae.dims()`) so the live SAE's true `d_in`/`d_sae` win over the config fallback.

Tech Stack: Python 3.12, FastAPI, pydantic v2 + pydantic-settings, sae-lens 6.x (`SAE.from_pretrained`), torch 2.4+ (GPU-only, GPU-gated tests), pytest 8, TypeScript/Vite (frontend display strings; no JS test runner — verified via `tsc` + `grep`).

---

## Global Constraints

- Python >= 3.12 (matches .python-version).
- Base deps (pyproject.toml): fastapi>=0.138.0, sentry-sdk[fastapi]>=2.63.0, httpx>=0.27.0.
- ML extra (uv sync --extra ml): torch>=2.4, transformers>=4.50, sae-lens>=6.0, scikit-learn>=1.5, accelerate>=0.34, anthropic>=0.40.
- pyproject.toml is the canonical dependency source; backend/requirements.txt is being retired.
- License: MIT.
- Naming/copy: KEEP the product name "GlassBox"; REMOVE medical-specific positioning (reposition as a general-purpose LLM interpretability/observability tool); medical survives only as ONE clearly-labeled optional example. Rename the per-turn event from cognition_event to introspection_event (class CognitionEvent -> IntrospectionEvent).
- Secrets (ANTHROPIC_API_KEY, SENTRY_DSN, POD_TOKEN, HF_TOKEN) load ONLY from env / .env, NEVER from config.yaml.
- COMMIT RULE (CRITICAL): commit messages MUST NOT add Claude as a co-author. No "Co-Authored-By: Claude" trailer anywhere. Use Conventional Commits style.
- Tests: keep the existing backend pytest suite green; CI runs the non-GPU subset; backend/engine.py and backend/science/sae.py are GPU-gated (untested without weights).
- Two processes: orchestration backend (backend/app.py, NEVER imports torch) and GPU pod service (backend/gpu_service.py, owns torch). AppConfig is built once per process and threaded via FastAPI app.state.

---

## Dependency on WS0 (read before starting)

WS2 is downstream of **WS0**, which lands the `AppConfig` refactor (`backend/config.py` rewrite, `config.example.yaml`, `app.state` wiring, and the §3.2 signature changes). This plan assumes WS0 has already shipped the following from the AppConfig contract, and consumes them **verbatim**:

- `SAEConfig` with fields `release`, `sae_id_pattern`, `d_in`, `d_sae`, `np_model`, `np_source_pattern`, `np_feature_url`, `recon_min_cosine`, `recon_probe`.
- `ModelConfig` with fields `model_id`, `layer`, `mask_tokens`, `preamble_skip`, `max_new_tokens`, `device`, `system_prompt`.
- The `AppConfig` helpers `cfg.sae_id(layer=None) -> str` and `cfg.np_source(layer=None) -> str`.
- The §3.2 torch-module signatures: `science/sae.load_sae(sae: SAEConfig, layer: int, device=None)`, `science/sae.reconstruction_error(act, sae: SAEConfig)`.
- The §3.2 engine signatures Task 3 calls: `engine.load_engine(model: ModelConfig, device=None)` (current real signature is `load_engine(device=None)` — engine.py:38) and `engine.probe_activation(text, model: ModelConfig, pos=None)` (current real signature is `probe_activation(text, pos=None)` — engine.py:219). `backend/engine.py` is NOT in WS2's file list — WS2 only CALLS these refactored signatures; WS0 must land the engine refactor. If WS0 left `engine.load_engine`/`engine.probe_activation` at their old signatures, Task 3's recon-check and load path break — STOP and reconcile with WS0.
- The §3.2 persona signature Task 3 calls: `science/persona.load_artifacts(probes: ProbeConfig, include=None)` (current real signature is `load_artifacts(artifact_dir=..., exclude=(), include=None)` — persona.py:198). `backend/science/persona.py` is NOT in WS2's file list — WS2 only CALLS the refactored signature at the `_attempt_load` call site; WS0 owns the persona refactor. If WS0 has not landed the `ProbeConfig` signature, Task 3's `persona.load_artifacts(cfg.probes)` call breaks — STOP and reconcile with WS0.
- The §3.2 NP_MODEL/NP_SOURCE conversion in the pure-CPU label module: WS0 OWNS converting `backend/labels.py:70` (`config.NP_FEATURE_URL.format(model=config.NP_MODEL, source=config.NP_SOURCE, index=...)`) to `cfg.sae.np_feature_url.format(model=cfg.sae.np_model, source=cfg.np_source(layer), index=...)`. WS2 does NOT touch `labels.py`; if WS0 leaves `labels.py` reading the old `config.NP_*` globals after WS2 removes them elsewhere, `labels.py` is an orphaned consumer and `/turn` label-fetch breaks — verify WS0 converted it.

If WS0 has NOT yet shipped these symbols, STOP and complete WS0 first — every task below imports them.

**Preserved, GPU-gated, NOT re-tested by WS2** (design doc §3 WS2 bullet 3): `backend/engine._find_decoder_layers` (engine.py:23) — the auto decoder-layer detection — is already model-agnostic and stays UNTOUCHED. WS2 adds no test for it; it is GPU-gated (needs model weights) and out of WS2's file scope. The headline "swap the layer via `cfg.model.layer`" feature flows as: `cfg.model.layer` → `gpu_service._attempt_load` → `sae.load_sae(cfg.sae, cfg.model.layer)` (Task 3) and `cfg.sae_id(cfg.model.layer)` / `cfg.np_source(cfg.model.layer)`; the layer-detection path inside `engine` is not on this seam and is explicitly scoped OUT (preserved, GPU-gated). No end-to-end layer-swap test is in scope because it requires weights.

**Files this plan edits that other workstreams also edit** (sequence carefully; noted again per-task):
- `backend/config.py` — WS0 owns the rewrite; WS2 only TOUCHES the `SAEConfig`/`ModelConfig` docstrings + defaults (Task 7). Land after WS0.
- `backend/schema.py` — WS1 renames `CognitionEvent -> IntrospectionEvent`; WS2 only documents the `Feature.source` / `CognitionEvent.model` / `CognitionEvent.layer` defaults as Gemma examples (Task 6). Coordinate the class name with WS1.
- `frontend/src/mock.ts` — WS1 removes `phoenix_ui_url`; WS2 only touches the `source` display string + the demo-event comment (Task 8). Coordinate.
- `backend/gpu_service.py` / `backend/runtime.py` / `backend/fallback.py` — WS0 changes their signatures to take sub-configs; WS2 changes the d_sae-derivation call sites inside them (Tasks 2, 3, 4). Land after WS0.
- `backend/analyze.py` — pure-CPU orchestration module (NEVER imports torch). WS0 threads an `AppConfig` handle into it (replacing its `config.MODEL_ID`/`config.LAYER`/`config.MAX_NEW_TOKENS` reads at lines 128, 191, 192). WS2 updates its TWO `fallback.synth_turn(messages)` call sites (lines 164, 174) to pass `cfg.sae, cfg.np_source()` when Task 5 changes the `synth_turn` signature (Task 5). This is the "UI runs with zero setup" CPU fallback path the design doc §7 requires — if it is not updated, `/api/chat` and `/api/analyze` crash at RUNTIME (not just in tests) the moment Task 5 lands. PRECONDITION: WS0 must have given `analyze` a `cfg`/`AppConfig` handle in scope (verify before Task 5 — see Task 5 NOTE).
- `backend/app.py` — pure-CPU orchestration backend (NEVER imports torch). WS0 builds the `AppConfig` here at startup and stores it (e.g. `app.state.config` / a `runtime` handle). WS2 updates its TWO `runtime.health_payload()` call sites (lines 46 and 54 — `/api/health` and `/api/observability`) to pass the `cfg` when Task 5 changes `health_payload` to require it (Task 5). If not updated, BOTH endpoints crash at RUNTIME the moment Task 5 lands. PRECONDITION: WS0 exposes a `cfg`/`AppConfig` handle reachable from these handlers (verify before Task 5).
- `backend/labels.py` — pure-CPU label-fetch module (NEVER imports torch). It reads `config.NP_MODEL`/`config.NP_SOURCE`/`config.NP_FEATURE_URL` at line 70. Per the AppConfig contract §3.2, `labels` is an NP_SOURCE/NP_MODEL consumer and WS0 OWNS its conversion to `cfg.sae.np_model`/`cfg.np_source(layer)`/`cfg.sae.np_feature_url`. WS2 does NOT touch `labels.py`; it is listed here so the orphaned-consumer risk is explicit (see "Dependency on WS0" — WS0 deliverable).

---

## File Structure

- `backend/science/sae.py` — **modify**: `load_sae` takes `(sae: SAEConfig, layer, device)`; new `dims()` returns the loaded SAE's true `(d_in, d_sae)`; `reconstruction_error` takes `(act, sae)`; `sae_topk`/`attribution_topk` take the per-layer Neuronpedia source string instead of reading `config.NP_SOURCE`.
- `backend/gpu_service.py` — **modify**: load SAE via `cfg.sae`; `/health` reports SAE-derived `d_in`/`d_sae` (`sae.dims()`) with `cfg.sae.d_in`/`cfg.sae.d_sae` fallback; recon check uses `cfg.sae.recon_probe`/`cfg.sae.recon_min_cosine`.
- `backend/runtime.py` — **modify**: `health_payload` adds `d_in` and keeps `d_sae`, both falling back to `cfg.sae.d_in`/`cfg.sae.d_sae`.
- `backend/fallback.py` — **modify**: synthetic feature indices use `cfg.sae.d_sae`; `source` uses `cfg.np_source()`.
- `backend/schema.py` — **modify**: `Feature.source` default + `CognitionEvent.model`/`layer` defaults become config-derived placeholders documented as Gemma examples (no hardcoded medical/Gemma magic strings presented as canonical).
- `frontend/src/probes.ts` — **modify**: comment pointer `backend.config.ENABLED_TRACKERS` → `cfg.probes.enabled` (stale reference).
- `frontend/src/mock.ts` — **modify**: keep the Gemma demo values but label them as the demo profile (no behavior change; string-accuracy only).
- `backend/config.py` — **modify (WS0-owned file)**: refresh the `SAEConfig`/`ModelConfig` docstrings so they read as Gemma defaults of a generic seam (stale "LOCKED"/medical language removed).
- `backend/tests/test_sae_dims.py` — **create**: unit tests for `sae.dims()` derivation + fallback (no GPU; uses a fake SAE object).
- `backend/tests/test_sae_config_seam.py` — **create**: unit tests that `load_sae`/`sae_topk`/`attribution_topk` consume the injected `SAEConfig`/source rather than module globals (no GPU; SAE-lens monkeypatched).

---

### Task 1: `sae.dims()` — derive `(d_in, d_sae)` from the loaded SAE with config fallback

Files:
- Create: `backend/tests/test_sae_dims.py`
- Modify: `backend/science/sae.py` (add `dims()` after `width()` at lines 28-33; `width()` stays for back-compat)
- Test: `backend/tests/test_sae_dims.py`

Interfaces:
- Consumes: `SAEConfig` from the AppConfig contract (fields `d_in: int = 2560`, `d_sae: int = 16384`). The loaded SAE object exposes a `cfg` attribute carrying `d_sae` and `d_in` ints (sae-lens `SAE.cfg`), as `width()` already relies on (`backend/science/sae.py:32-33`).
- Produces: `sae.dims(sae_cfg: SAEConfig) -> tuple[int, int]` returning `(d_in, d_sae)` — the live SAE's dims when loaded, else the `SAEConfig` fallback. `sae.width()` keeps its current `int | None` contract.
- ASSUMPTION (contract §1): the tests construct `SAEConfig(d_in=2560, d_sae=16384)` / `SAEConfig(d_in=..., d_sae=...)` directly, so they assume WS0 ships `SAEConfig` accepting `d_in`/`d_sae` as constructor kwargs (true per the AppConfig contract §1). If WS0 folded the dims under a nested model or different kwarg names, these constructors fail — that is a WS0 contract deviation to reconcile, not a WS2 change. The Step 2 "stop if SAEConfig import fails" guidance covers the import; this note covers the constructor-kwarg shape.

- [ ] Step 1: Write the failing test file `backend/tests/test_sae_dims.py`.
```python
"""WS2: sae.dims() derives (d_in, d_sae) from the loaded SAE, falling back to SAEConfig.

No GPU: a fake SAE object stands in for the sae-lens SAE. We poke it onto the module-level
`_sae` cache that load_sae() would set, then assert dims() reads it. The SAEConfig fallback
path is exercised with _sae=None."""

from __future__ import annotations

import pytest

from backend.config import SAEConfig
from backend.science import sae as sae_mod


class _FakeCfg:
    def __init__(self, d_in, d_sae):
        self.d_in = d_in
        self.d_sae = d_sae


class _FakeSAE:
    def __init__(self, d_in, d_sae):
        self.cfg = _FakeCfg(d_in, d_sae)


@pytest.fixture(autouse=True)
def _reset_sae():
    prev = sae_mod._sae
    sae_mod._sae = None
    yield
    sae_mod._sae = prev


def test_dims_falls_back_to_config_when_unloaded():
    cfg = SAEConfig(d_in=2560, d_sae=16384)
    assert sae_mod._sae is None
    assert sae_mod.dims(cfg) == (2560, 16384)


def test_dims_prefers_loaded_sae_over_config_fallback():
    # Loaded SAE reports DIFFERENT dims than the config fallback — the live values must win,
    # so a user who swaps to a 65k-width SAE without editing config.yaml still gets the truth.
    sae_mod._sae = _FakeSAE(d_in=3584, d_sae=65536)
    cfg = SAEConfig(d_in=2560, d_sae=16384)
    assert sae_mod.dims(cfg) == (3584, 65536)


def test_dims_uses_config_when_loaded_sae_cfg_missing_a_field():
    # A loaded SAE whose cfg exposes d_sae but not d_in must backfill d_in from config,
    # never crash or return 0.
    class _PartialCfg:
        d_sae = 65536

    class _PartialSAE:
        cfg = _PartialCfg()

    sae_mod._sae = _PartialSAE()
    cfg = SAEConfig(d_in=2560, d_sae=16384)
    assert sae_mod.dims(cfg) == (2560, 65536)
```

- [ ] Step 2: Run the test and watch it fail because `dims` does not exist.
```bash
uv run pytest backend/tests/test_sae_dims.py -q
```
Expected failure output (collection/attribute error):
```
E   AttributeError: module 'backend.science.sae' has no attribute 'dims'
...
1 error in 0.XXs
```
(If `SAEConfig` import fails instead, WS0 has not landed — stop and finish WS0.)

- [ ] Step 3: Implement `dims()` in `backend/science/sae.py`, immediately after the existing `width()` function (after line 33). Insert this block:
```python
def dims(sae_cfg) -> tuple[int, int]:
    """Return (d_in, d_sae) for the active SAE.

    Prefers the LOADED SAE's reported dims (sae-lens exposes them on `_sae.cfg`) so a swapped
    SAE is described truthfully even when config.yaml still carries the old fallback numbers.
    Each dim independently falls back to the SAEConfig value (sae_cfg.d_in / sae_cfg.d_sae)
    when the SAE is not loaded or its cfg omits that field. Never returns 0 or None."""
    cfg = getattr(_sae, "cfg", None) if _sae is not None else None
    d_in = int(getattr(cfg, "d_in", 0) or 0) or int(sae_cfg.d_in)
    d_sae = int(getattr(cfg, "d_sae", 0) or 0) or int(sae_cfg.d_sae)
    return d_in, d_sae
```

- [ ] Step 4: Run the test and watch it pass.
```bash
uv run pytest backend/tests/test_sae_dims.py -q
```
Expected pass output:
```
...                                                                      [100%]
3 passed in 0.XXs
```

- [ ] Step 5: Commit.
```bash
git add backend/science/sae.py backend/tests/test_sae_dims.py
git commit -m "feat(sae): derive d_in/d_sae from the loaded SAE with config fallback"
```

---

### Task 2: Inject `SAEConfig` into `load_sae` / `reconstruction_error` and the per-layer Neuronpedia source

Files:
- Create: (none)
- Modify: `backend/science/sae.py` (`load_sae` lines 13-25; `sae_topk` lines 42-60; `attribution_topk` lines 93-132; `reconstruction_error` lines 135-146)
- Modify (SAME commit — existing callers of the changed signatures): `backend/tests/test_attribution.py` (direct `attribution_topk` calls at lines 37, 50, 60, 81 and the `sae_mod.config.NP_SOURCE` assertion at line 41 — must pass `source=`)
- Test: `backend/tests/test_sae_config_seam.py`

Interfaces:
- Consumes: AppConfig contract §3.2 — `load_sae(sae: SAEConfig, layer: int, device=None)` and `reconstruction_error(act, sae: SAEConfig)`. `SAEConfig.release`, and the per-layer SAELens id produced by `cfg.sae_id(layer)` (pattern `SAEConfig.sae_id_pattern`). The Neuronpedia per-layer slug `cfg.np_source(layer)` (pattern `SAEConfig.np_source_pattern`).
- Produces: `load_sae(sae, layer, device=None)` (no module-global reads of `config.SAE_RELEASE`/`config.sae_id_for_layer`); `sae_topk(act, k, source: str)`, `attribution_topk(acts, grad, keep, cap, baseline=None, *, source: str)` that stamp the passed `source` onto each emitted feature dict instead of reading `config.NP_SOURCE`; `reconstruction_error(act, sae)` (signature parity with §3.2; body unchanged otherwise).
- NOTE: `backend/science/sae.py` is GPU-gated — the SAE-lens load path is untested without weights. This task's test monkeypatches `SAE.from_pretrained` and `_sae` so the *config-threading* is covered on CPU; the numeric encode/topk math stays GPU-gated.
- NOTE: callers (`gpu_service.py`, `feature_provider.py`) are updated in Task 3/Task 4; `source` is a REQUIRED keyword-only arg (no default-from-config) so the seam is enforced and callers must pass it.
- NOTE (existing-caller breakage — MUST be fixed in THIS commit so the suite is never left red): `test_attribution.py` calls `attribution_topk(...)` DIRECTLY without `source` at lines 37, 50, 60, 81, and asserts `c["source"] == sae_mod.config.NP_SOURCE` at line 41. After this task `attribution_topk` requires keyword-only `source` (TypeError otherwise) and `sae_mod.config.NP_SOURCE` may no longer exist once the `config` import is dropped. Step 3a below updates these in the same commit. (The monkeypatched `fake_attr`/`fake_topk` stubs in `test_attribution.py` and `test_provider_masking.py` belong to the `feature_provider` call path and are fixed in Task 4, where the provider starts passing `source` to them.)

- [ ] Step 1: Write the failing test file `backend/tests/test_sae_config_seam.py`.
```python
"""WS2: science/sae consumes the injected SAEConfig + per-layer source, not module globals.

No GPU: sae-lens SAE.from_pretrained is monkeypatched to a fake, and the encode path is fed a
fake _sae. We assert (a) load_sae passes release + cfg.sae_id(layer) through to from_pretrained,
and (b) sae_topk/attribution_topk stamp the *passed* source onto every feature dict."""

from __future__ import annotations

import sys
import types

import pytest

from backend.config import AppConfig, SAEConfig
from backend.science import sae as sae_mod


@pytest.fixture(autouse=True)
def _reset_sae():
    prev = sae_mod._sae
    sae_mod._sae = None
    yield
    sae_mod._sae = prev


def test_load_sae_passes_release_and_per_layer_id(monkeypatch):
    captured = {}

    class _FakeSAE:
        cfg = types.SimpleNamespace(d_in=2560, d_sae=16384)

        def to(self, _dev):
            return self

        def eval(self):
            return self

    def _fake_from_pretrained(release, sae_id, device):
        captured["release"] = release
        captured["sae_id"] = sae_id
        captured["device"] = device
        return _FakeSAE()

    fake_mod = types.ModuleType("sae_lens")
    fake_mod.SAE = types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
    monkeypatch.setitem(sys.modules, "sae_lens", fake_mod)

    cfg = AppConfig()  # Gemma defaults: release gemma-scope-2-4b-it-res, layer 17
    monkeypatch.setattr(cfg, "resolve_device", lambda pref=None: "cpu")

    sae_mod.load_sae(cfg.sae, layer=22, device="cpu")

    assert captured["release"] == "gemma-scope-2-4b-it-res"
    assert captured["sae_id"] == "layer_22_width_16k_l0_medium"
    assert captured["device"] == "cpu"


class _EncodeFakeSAE:
    """Minimal fake exposing .encode for sae_topk's no-grad path."""

    def __init__(self):
        self.cfg = types.SimpleNamespace(d_in=4, d_sae=4)

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def encode(self, x):
        import torch

        # one token -> 4 latents, descending so topk is deterministic
        return torch.tensor([[3.0, 2.0, 1.0, 0.0]])


def test_sae_topk_stamps_passed_source():
    import torch

    sae_mod._sae = _EncodeFakeSAE()
    feats = sae_mod.sae_topk(torch.zeros(4), k=2, source="22-gemmascope-2-res-65k")
    assert {f["source"] for f in feats} == {"22-gemmascope-2-res-65k"}
    assert [f["index"] for f in feats] == [0, 1]
```

- [ ] Step 2: Run the test and watch it fail because the new signatures do not exist yet.
```bash
uv run pytest backend/tests/test_sae_config_seam.py -q
```
Expected failure output (one of):
```
E   TypeError: load_sae() got an unexpected keyword argument 'layer'
...
E   TypeError: sae_topk() got an unexpected keyword argument 'source'
...
FAILED backend/tests/test_sae_config_seam.py
```

- [ ] Step 3: Rewrite the affected functions in `backend/science/sae.py`. Replace `load_sae` (lines 13-25):
```python
def load_sae(sae, layer: int, device: str | None = None):
    """Load the Gemma-Scope SAE for `layer` onto the resolved device.

    `sae` is a SAEConfig: `sae.release` + the per-layer SAELens id rendered from
    `sae.sae_id_pattern` (layer-substituted). v1 is validated on Gemma-Scope; the config seam
    is the additive hook for arbitrary HF SAEs (OSS issue #1)."""
    global _sae
    from sae_lens import SAE

    import torch

    p = (device or "auto").lower()
    if p == "cuda" and torch.cuda.is_available():
        dev = "cuda"
    elif p == "mps" and torch.backends.mps.is_available():
        dev = "mps"
    elif p == "cpu":
        dev = "cpu"
    elif torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    sae_id = sae.sae_id_pattern.format(layer=layer)
    loaded = SAE.from_pretrained(release=sae.release, sae_id=sae_id, device=dev)
    # SAELens has returned either an SAE or a (sae, cfg, sparsity) tuple across versions.
    _sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    _sae = _sae.to(dev).eval()
    return _sae
```
Replace `sae_topk` (lines 42-60) so it takes `source` and drops `config`:
```python
def sae_topk(act, k: int, source: str) -> list[dict]:
    """act: layer resid_post activation tensor [d_in] (one token).
    Returns up to k {index, act, source} dicts (labels attached later by labels.py).
    `source` is the Neuronpedia per-layer slug (caller passes cfg.np_source(layer)).
    Mask special tokens upstream — their activations are high-norm noise.
    Used by the legacy raw-activation ranking path; attribution_topk is preferred."""
    import torch

    if _sae is None:
        raise RuntimeError("call load_sae() first")
    a = _prep(act)
    with torch.no_grad():
        feats = _sae.encode(a.unsqueeze(0)).squeeze(0)  # [d_sae]
    vals, idx = feats.topk(k)
    return [
        {"index": int(i), "act": round(float(v), 3), "source": source}
        for v, i in zip(vals.tolist(), idx.tolist())
        if v > 0
    ]
```
Change `attribution_topk` (lines 93-132): update the signature to `def attribution_topk(acts, grad, keep, cap: int, baseline=None, *, source: str) -> list[dict]:` and replace the single `"source": config.NP_SOURCE,` line (line 129) with `"source": source,`. Then replace `reconstruction_error` (line 135) signature `def reconstruction_error(act) -> dict:` with `def reconstruction_error(act, sae) -> dict:` — `sae` is a REQUIRED positional arg, matching contract §3.2 `reconstruction_error(act, sae: SAEConfig)` exactly (NO `=None` default; the body is otherwise unchanged — `sae` is accepted for signature parity and the recon math reads only the loaded `_sae`). Task 3's only caller passes it. Finally remove the now-unused `from .. import config` import at line 8 ONLY IF no remaining reference to `config.` exists in the file — verify with the grep in Step 4 first; if any `config.TOPK`/`config.LAYER` defaults remain in other signatures, leave the import.

- [ ] Step 3a (SAME commit): Update the existing DIRECT callers in `backend/tests/test_attribution.py` so the suite is never red. The kernel-math tests call `attribution_topk` directly without `source`:
  - Line 37: `out = sae_mod.attribution_topk(acts, grad, keep=[0, 1], cap=10)` → add `, source="s"` before the close paren.
  - Line 50: `out = sae_mod.attribution_topk(acts, grad, keep=[0], cap=10)` → add `, source="s"`.
  - Line 60: `sae_mod.attribution_topk(acts, grad, keep=[], cap=5)` → add `, source="s"`.
  - Line 81: `out = sae_mod.attribution_topk(acts, grad, keep=[0, 1], cap=10, baseline=baseline)` → add `, source="s"`.
  - Line 41: replace `assert all(c["source"] == sae_mod.config.NP_SOURCE for c in out)` with `assert all(c["source"] == "s" for c in out)` (the passed source value; `sae_mod.config.NP_SOURCE` no longer exists once the `config` import is dropped).
  (The provider-branch tests at lines 216-290 in the same file — `fake_attr`/`fake_topk` stubs and the `fp.config` monkeypatches — are NOT touched here; they exercise the `feature_provider` path and are updated in Task 4.)

- [ ] Step 4: Confirm no `config.` references survive in the rewritten functions, then run the new test AND the existing `test_attribution.py` (which exercises the direct `attribution_topk` callers updated in Step 3a) so the suite stays green.
```bash
grep -n "config\." backend/science/sae.py
uv run pytest backend/tests/test_sae_config_seam.py backend/tests/test_attribution.py -q
```
Expected: the grep prints nothing (or only lines you intentionally left), and both files pass (the `test_attribution.py` provider-branch tests still pass because Task 4 has not yet changed the provider — they call `features_for` with the OLD signature, which still works until Task 4):
```
..                                                                       [100%]
N passed in 0.XXs
```

- [ ] Step 5: Commit (production signature change + its direct-caller test fixes together — the suite is never left red).
```bash
git add backend/science/sae.py backend/tests/test_sae_config_seam.py backend/tests/test_attribution.py
git commit -m "refactor(sae): thread SAEConfig + per-layer source into load_sae/topk/recon"
```

---

### Task 3: Update `gpu_service` call sites to the new SAE seam + SAE-derived `/health` dims

Files:
- Create: (none)
- Modify: `backend/gpu_service.py` (`_run_recon_check` lines 39-58; `_attempt_load` lines 61-83; `_baseline_vec` lines 101-134; `_sae_candidates` lines 137-158; `health` lines 178-196)
- Test: `backend/tests/test_gpu_service.py` (existing — extend; do not break)

Interfaces:
- Consumes: WS0-threaded `app.state.config` (an `AppConfig`); `cfg.sae`, `cfg.model`, `cfg.sae_id(layer)`, `cfg.np_source(layer)`; new `sae.load_sae(sae, layer, device)`, `sae.dims(sae_cfg)`, `sae.reconstruction_error(act, sae)`, `sae.sae_topk(act, k, source)`/`attribution_topk(..., source=...)` from Tasks 1-2; `science/feature_provider.LocalSAEProvider.features_for(...)` whose `sae_topk`/`attribution_topk` calls Task 4 updates to pass `source`.
- Produces: `/health` returns `d_in` AND `d_sae` derived via `sae.dims(cfg.sae)` (live SAE dims, config fallback); recon check uses `cfg.sae.recon_probe`/`cfg.sae.recon_min_cosine`; SAE loaded for `cfg.model.layer` via `cfg.sae_id`.
- NOTE (cross-WS): `backend/gpu_service.py` startup/threading is WS0-owned — WS0 stores `app.state.config` and changes `_attempt_load`/`_require_auth` to read it. This task assumes `cfg = app.state.config` is reachable in the handlers. Where WS0 introduced a module-level `_cfg()` accessor or `request.app.state.config`, use that; the test below reads `app.state.config` directly via the TestClient app instance.
- NOTE (test coverage — two failing tests, not one): this task makes FIVE production rewrites — `health()` (dims), `_run_recon_check()`, `_attempt_load()`, `_baseline_vec()`, `_sae_candidates()`. The existing `test_gpu_service.py` monkeypatches `_sae_candidates` and `_capture` AWAY (`_install_stubs`), so the real `source`/`rank_method`/`preamble_skip` wiring is never exercised by the suite. Step 1 red-greens the `/health` dims; Step 1a adds a SECOND failing test that drives the REAL `_sae_candidates` (monkeypatching only `LocalSAEProvider.features_for` to capture kwargs) so the source/ranking-knob threading — the part most likely to crash `/turn` — is test-driven too. `_run_recon_check`/`_baseline_vec` stay GPU-gated (they need `engine.probe_activation`/`engine.turn` weights); their signature changes are exercised only via the full-suite import surface (Task 9) — call this out as deliberately GPU-gated, not silently uncovered.

- [ ] Step 1: Add a failing test to `backend/tests/test_gpu_service.py` that `/health` reports SAE-derived dims. Append:
```python
def test_health_reports_sae_derived_dims(monkeypatch):
    """WS2: /health d_in/d_sae come from sae.dims(cfg.sae) — live SAE values when loaded,
    config fallback otherwise. We fake a loaded SAE reporting non-default dims and assert
    they surface, proving derivation (not the static config numbers) drives /health."""
    import types

    from fastapi.testclient import TestClient

    from backend import gpu_service
    from backend.config import AppConfig
    from backend.science import sae as sae_mod

    gpu_service.app.state.config = AppConfig()  # Gemma defaults: d_in 2560, d_sae 16384
    gpu_service.STATE.update(mode="real", model_loaded=True, sae_loaded=True)

    sae_mod._sae = types.SimpleNamespace(cfg=types.SimpleNamespace(d_in=3584, d_sae=65536))
    try:
        client = TestClient(gpu_service.app)
        body = client.get("/health").json()
    finally:
        sae_mod._sae = None
    assert body["d_in"] == 3584
    assert body["d_sae"] == 65536
```
(If `test_gpu_service.py` already builds a config fixture differently, mirror its setup; the assertion on `d_in`/`d_sae` is the load-bearing part.)

- [ ] Step 1a: Add a SECOND failing test to `backend/tests/test_gpu_service.py` that exercises the REAL `_sae_candidates` and pins the source/ranking-knob threading from `cfg`. This is the wiring `_install_stubs` hides; it must be driven by its own test. Append:
```python
def test_sae_candidates_threads_source_and_ranking_knobs(monkeypatch):
    """WS2: _sae_candidates passes cfg.np_source(layer), cfg.feature_cloud.rank_method, and
    cfg.model.preamble_skip into LocalSAEProvider.features_for — proving the real wiring
    (not the _install_stubs fake) carries the per-layer source/ranking knobs from cfg."""
    from backend import gpu_service
    from backend.config import AppConfig
    from backend.science import feature_provider as fp

    gpu_service.app.state.config = AppConfig()  # Gemma defaults: layer 17
    captured = {}

    def _fake_features_for(self, text, **kw):
        captured.update(kw)
        return [{"index": 1, "act": 1.0, "source": kw["source"]}]

    monkeypatch.setattr(fp.LocalSAEProvider, "features_for", _fake_features_for)

    class _Tok:
        def convert_tokens_to_ids(self, t):
            return 0

    res = {"tok": _Tok(), "answer": "hi", "out_ids": [0, 0, 0], "resp_start": 0}
    out = gpu_service._sae_candidates(res, resp_acts=None, resp_grad=None, cap=30)

    assert captured["source"] == "17-gemmascope-2-res-16k"          # cfg.np_source(cfg.model.layer)
    assert captured["rank_method"] == AppConfig().feature_cloud.rank_method
    assert captured["preamble_skip"] == AppConfig().model.preamble_skip
    assert out and out[0]["source"] == "17-gemmascope-2-res-16k"
```
(`out_ids` may be a list or a tensor in production; here a list keeps the test torch-free. If WS0's `feature_cloud`/`model` field names differ from `rank_method`/`preamble_skip`, mirror them — the load-bearing assertion is that the source + the two ranking knobs arrive from `cfg`, not module globals.)

- [ ] Step 2: Run both new tests and watch them fail — `/health` has no `d_in` and ignores `sae.dims`, and `_sae_candidates` does not yet pass `source`/`rank_method`/`preamble_skip`.
```bash
uv run pytest backend/tests/test_gpu_service.py::test_health_reports_sae_derived_dims backend/tests/test_gpu_service.py::test_sae_candidates_threads_source_and_ranking_knobs -q
```
Expected failure output (both fail):
```
E   KeyError: 'd_in'
...
E   KeyError: 'source'   # _sae_candidates doesn't pass source yet (or AttributeError on cfg.np_source)
...
FAILED backend/tests/test_gpu_service.py::test_health_reports_sae_derived_dims
FAILED backend/tests/test_gpu_service.py::test_sae_candidates_threads_source_and_ranking_knobs
```

- [ ] Step 3: Edit `backend/gpu_service.py`. In `health()` (lines 178-196), replace the `"d_sae": sae.width() or config.D_SAE,` line and add `d_in`. Read the threaded config (`cfg = app.state.config`) and compute dims once:
```python
@app.get("/health")
def health() -> dict:
    from .science import concept_synth as cs
    from .science import sae
    from .science.persona import _trackers

    cfg = app.state.config
    d_in, d_sae = sae.dims(cfg.sae)
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": cfg.model.model_id,
        "layer": cfg.model.layer,
        "d_in": d_in,
        "d_sae": d_sae,
        "trackers": list(_trackers.keys()),
        "sae_recon_cosine": STATE.get("sae_recon_cosine"),
        "sae_recon_ok": STATE.get("sae_recon_ok"),
        "anthropic_configured": bool(cfg.anthropic_api_key),
        "active_probe_jobs": cs.active_job_count(),
    }
```
In `_run_recon_check()` (lines 39-58), thread `cfg` and call the new signatures:
```python
def _run_recon_check() -> None:
    cfg = app.state.config
    try:
        from . import engine
        from .science import sae

        rec = sae.reconstruction_error(engine.probe_activation(cfg.sae.recon_probe, cfg.model), cfg.sae)
        cos = round(float(rec["cosine"]), 3)
        STATE["sae_recon_cosine"] = cos
        STATE["sae_recon_ok"] = cos >= cfg.sae.recon_min_cosine
        if STATE["sae_recon_ok"]:
            print(f"[gpu_service] SAE reconstruction cosine {cos} [OK]")
        else:
            print(
                f"[gpu_service] WARNING SAE reconstruction cosine {cos} < "
                f"{cfg.sae.recon_min_cosine} — feature cloud may be unreliable"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[gpu_service] recon check skipped ({e})")
        STATE["sae_recon_cosine"] = None
        STATE["sae_recon_ok"] = None
```
In `_attempt_load()` (lines 61-83) replace `engine.load_engine()` with `engine.load_engine(cfg.model)`, `sae.load_sae()` with `sae.load_sae(cfg.sae, cfg.model.layer)`, and `persona.load_artifacts(exclude=config.DISABLED_TRACKERS)` with `persona.load_artifacts(cfg.probes)` — adding `cfg = app.state.config` at the top.

In `_baseline_vec()` (lines 101-134), add `cfg = app.state.config` at the top and replace `config.CONTRAST_BASELINE`/`config.CONTRAST_PROMPT`/`config.CONTRAST_MAX_NEW` (lines 105, 114, 115) with `cfg.feature_cloud.contrast_baseline`/`cfg.feature_cloud.contrast_prompt`/`cfg.feature_cloud.contrast_max_new`, the `config.MASK_TOKENS` read (line 122) with `cfg.model.mask_tokens`, and the `config.PREAMBLE_SKIP` read (line 126) with `cfg.model.preamble_skip`. (These cloud/model knob names are WS0-owned; if WS0 chose different attribute names, match WS0's — WS2's only hard requirement here is that no `config.MASK_TOKENS`/`config.PREAMBLE_SKIP` module-global read survives.)

In `_sae_candidates()` (lines 137-158), rewrite the body to read `cfg = app.state.config`, build `special` from `cfg.model.mask_tokens`, and pass `source`/`rank_method`/`preamble_skip` into the provider (Task 4 makes `features_for` REQUIRE `source` and accept `rank_method`/`preamble_skip`). The literal rewritten function:
```python
def _sae_candidates(
    res: dict,
    resp_acts,
    resp_grad,
    *,
    cap: int,
    baseline=None,
) -> list[dict]:
    from .science.feature_provider import LocalSAEProvider

    cfg = app.state.config
    tok = res["tok"]
    special = {tok.convert_tokens_to_ids(t) for t in cfg.model.mask_tokens}
    resp_ids = res["out_ids"][res["resp_start"] :].tolist()
    return LocalSAEProvider().features_for(
        res["answer"],
        activations=resp_acts,
        token_ids=resp_ids,
        special_ids=special,
        grad=resp_grad,
        baseline=baseline,
        cap=cap,
        source=cfg.np_source(cfg.model.layer),
        rank_method=cfg.feature_cloud.rank_method,
        preamble_skip=cfg.model.preamble_skip,
    )
```
The other production caller of the ranking knob, the `attribution = config.RANK_METHOD == "attribution"` reads at lines 229 and 246 (inside the `/turn` handler), become `cfg.feature_cloud.rank_method` — read `cfg = app.state.config` in that handler (WS0 likely already did; if so, leave it). Where WS0 already converted any of these, only the SAE-specific lines (`load_sae`, `dims`, `reconstruction_error`, `recon_probe`, `np_source`, and the `_sae_candidates` `source` wiring) are WS2's responsibility — leave WS0's other conversions intact.

- [ ] Step 4: Run the new test and the full gpu_service suite.
```bash
uv run pytest backend/tests/test_gpu_service.py -q
```
Expected pass output:
```
...                                                                      [100%]
N passed in 0.XXs
```

- [ ] Step 5: Commit.
```bash
git add backend/gpu_service.py backend/tests/test_gpu_service.py
git commit -m "refactor(gpu): report SAE-derived d_in/d_sae and load SAE via cfg.sae"
```

---

### Task 4: Thread the per-layer Neuronpedia `source` through `feature_provider`

Files:
- Create: (none)
- Modify: `backend/science/feature_provider.py` (`features_for` signatures + bodies lines 17-129; `get_provider` reads `config.resolve_device` at line 141)
- Modify (SAME commit — existing callers/stubs of the changed signatures): `backend/tests/test_provider_masking.py` (the two `fake_topk` stubs at lines 16, 33 and the two `features_for` calls at lines 22, 39), `backend/tests/test_feature_provider.py` (the two `NeuronpediaProvider().features_for` calls at lines 13, 39), `backend/tests/test_attribution.py` (the provider-branch tests at lines 216-290: `fake_attr` stubs at 219/239/255, `fake_topk` stubs at 269/282, and the `fp.config` `RANK_METHOD`/`PREAMBLE_SKIP` monkeypatches at 225/244/245/260/261/287)
- Test: `backend/tests/test_feature_provider_source.py` (create)

Interfaces:
- Consumes: new `sae.sae_topk(act, k, source)` / `sae.attribution_topk(..., source=...)` from Task 2; `cfg.np_source(layer)` value passed in by `gpu_service` (Task 3) as the `source` kwarg.
- Produces: `LocalSAEProvider.features_for(..., source: str, ...)` and `NeuronpediaProvider.features_for(..., source: str, np_model: str, ...)` that thread the per-layer `source` (and Neuronpedia model) down to `sae_topk`/`attribution_topk` and stamp it on emitted dicts, instead of reading `config.NP_SOURCE`/`config.NP_MODEL`.
- NOTE: `feature_provider.py` reads several `config.*` cloud knobs WS0 converts to `cfg.feature_cloud.*` (`TOPK`, `TOPK_EVENT`). WS2 removes the `config.NP_SOURCE`/`config.NP_MODEL` reads (the SAE/label-source seam) AND lifts `config.RANK_METHOD`/`config.PREAMBLE_SKIP` into explicit `rank_method`/`preamble_skip` kwargs on `features_for` (so the per-layer source and the ranking knobs all arrive from the caller, not module globals). To keep this task self-contained and torch-free in test, the new test exercises `LocalSAEProvider` with a fake `sae_topk`.
- NOTE (existing-caller/stub breakage — MUST be fixed in THIS commit so the suite is never left red): making `source` a REQUIRED keyword-only arg on `FeatureProvider`/`LocalSAEProvider`/`NeuronpediaProvider.features_for`, and switching `RANK_METHOD`/`PREAMBLE_SKIP` from `fp.config` monkeypatches to kwargs, breaks SIX existing tests at call time:
  - `test_provider_masking.py:22,39` call `features_for(...)` without `source` → TypeError; and its `fake_topk` stubs at lines 16,33 are `def fake_topk(act, k=15)` (no `source`), so once the provider calls `sae_topk(..., source=...)` they raise `TypeError: unexpected keyword 'source'`.
  - `test_feature_provider.py:13,39` call `NeuronpediaProvider().features_for([...], cap=5)` with neither `source` nor `np_model` → TypeError BEFORE reaching the privacy guard (the `:13` test expects `RuntimeError("forbidden")`, which would now be shadowed by the TypeError).
  - `test_attribution.py:216-290` provider-branch tests call `features_for(...)` without `source`, their `fake_attr` stubs (`def fake_attr(activations, grad, keep, cap=50, baseline=None)` at 219/239/255) and `fake_topk` stubs (269/282) don't accept `source`, and they drive behavior by `monkeypatch.setattr(fp.config, "RANK_METHOD"/"PREAMBLE_SKIP", ...)` (225/244/245/260/261/287) which no longer influences the provider once it reads kwargs (default `rank_method="attribution"`/`preamble_skip=12`). All must move to passing `source=`/`rank_method=`/`preamble_skip=` and accept `source` in their stubs.
  Step 3a–3c below rewrite all three test files in the same commit as the signature change.

- [ ] Step 1: Write the failing test `backend/tests/test_feature_provider_source.py`.
```python
"""WS2: LocalSAEProvider threads the per-layer Neuronpedia `source` down to sae_topk
instead of reading config.NP_SOURCE. We monkeypatch sae_topk to capture its source kwarg
and assert the provider passes through the source it was given. Activation-ranking path only
(grad=None), which needs no torch graph."""

from __future__ import annotations

import torch

from backend.science import feature_provider as fp


def test_local_provider_threads_source(monkeypatch):
    captured = {}

    def _fake_sae_topk(act, k, source):
        captured["source"] = source
        # one feature so the provider has something to rank
        return [{"index": 7, "act": 1.0, "source": source}]

    monkeypatch.setattr(fp, "sae_topk", _fake_sae_topk)

    prov = fp.LocalSAEProvider()
    out = prov.features_for(
        "hello",
        activations=torch.zeros(1, 4),
        token_ids=[0],
        special_ids=set(),
        grad=None,
        baseline=None,
        k=15,
        cap=30,
        source="22-gemmascope-2-res-65k",
        rank_method="activation",
        preamble_skip=0,
    )
    assert captured["source"] == "22-gemmascope-2-res-65k"
    assert out and out[0]["source"] == "22-gemmascope-2-res-65k"
```

- [ ] Step 2: Run the test and watch it fail because `features_for` rejects `source`/`rank_method`/`preamble_skip`.
```bash
uv run pytest backend/tests/test_feature_provider_source.py -q
```
Expected failure output:
```
E   TypeError: features_for() got an unexpected keyword argument 'source'
...
FAILED backend/tests/test_feature_provider_source.py
```

- [ ] Step 3: Edit `backend/science/feature_provider.py`. Update `LocalSAEProvider.features_for` (lines 42-81) to accept `source: str`, `rank_method: str`, `preamble_skip: int` as explicit kwargs and use them instead of `config.NP_SOURCE`/`config.RANK_METHOD`/`config.PREAMBLE_SKIP`. Replace the method body:
```python
    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        grad=None,
        baseline=None,
        k: int = 15,
        cap: int = 30,
        *,
        source: str,
        rank_method: str = "attribution",
        preamble_skip: int = 12,
    ) -> list[dict]:
        if activations is None:
            raise ValueError(
                "LocalSAEProvider needs captured activations [n_positions, d_in]"
            )
        special = set(special_ids or [])
        keep = [
            pos
            for pos in range(activations.shape[0])
            if not (token_ids is not None and pos < len(token_ids) and token_ids[pos] in special)
        ]
        if grad is not None and rank_method == "attribution":
            keep_c = [p for p in keep if p >= preamble_skip] or keep
            try:
                return attribution_topk(activations, grad, keep_c, cap=cap, baseline=baseline, source=source)
            except Exception as e:  # noqa: BLE001 — degrade to activation ranking, never crash
                print(f"[provider] attribution_topk failed ({e}); using activation ranking")
        best: dict[int, float] = {}
        for pos in keep:
            for f in sae_topk(activations[pos], k=k, source=source):
                best[f["index"]] = max(best.get(f["index"], 0.0), f["act"])
        top = sorted(best.items(), key=lambda kv: -kv[1])[:cap]
        return [{"index": i, "act": round(v, 3), "source": source} for i, v in top]
```
Update `NeuronpediaProvider.features_for` (lines 90-129) the same way: add `*, source: str, np_model: str` kwargs, and replace `config.NP_MODEL`→`np_model`, `config.NP_SOURCE`→`source` (the `payload["modelId"]`/`payload["source"]` at lines 107-108 PLUS the `"source": config.NP_SOURCE` emit at line 128). Update the base `FeatureProvider.features_for` (lines 17-33) signature to match (add the `*, source: str` keyword so subclasses agree). In `get_provider` (line 141) leave `config.resolve_device()` for WS0 to convert; if WS0 already removed it, this line is gone. Finally remove `from .. import config` (line 7) ONLY if Step 4's grep shows no surviving `config.` references; otherwise leave it. SCOPE NOTE: the privacy-guard message at line 102 (`"remote feature POST forbidden for patient data"`) carries medical-specific wording, but genericizing it is the broader "remove medical positioning" effort's job (not the SAE/label-source seam WS2 owns) — leave it UNCHANGED here to avoid a merge conflict; the `match="forbidden"` test (Task 4 Step 3b) is unaffected either way.

- [ ] Step 3a (SAME commit): Update `backend/tests/test_provider_masking.py` so its stubs accept `source` and its calls pass `source`. Both `fake_topk` stubs (lines 16, 33) must take the `source` kwarg the provider now passes, and both `features_for` calls (lines 22, 39) must pass `source=`:
```python
def test_local_provider_skips_special_positions(monkeypatch):
    def fake_topk(act, k=15, source="s"):
        pos = act[1]
        return [{"index": 100 + pos, "act": float(10 - pos), "source": source}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    prov = fp.LocalSAEProvider()
    feats = prov.features_for(
        "txt", activations=FakeActs(4), token_ids=[1, 999, 2, 3], special_ids={999},
        source="s", rank_method="activation",
    )
    ...
```
Apply the same two changes (`def fake_topk(act, k=15, source="s")` and `..., source="s", rank_method="activation"`) to `test_local_provider_no_special_keeps_all` (lines 33, 39). Passing `rank_method="activation"` keeps these on the no-grad activation path (their intent) regardless of the new default.

- [ ] Step 3b (SAME commit): Update `backend/tests/test_feature_provider.py`. The privacy-guard tests call `NeuronpediaProvider().features_for(...)` without `source`/`np_model`. The guard must be reached BEFORE the new required kwargs trip a TypeError. Since `source`/`np_model` are now required, pass them at both call sites so the call reaches the guard:
  - Line 13: `NeuronpediaProvider().features_for(["msg"], cap=5, source="17-gemmascope-2-res-16k", np_model="gemma-3-4b-it")` — still expects `pytest.raises(RuntimeError, match="forbidden")` (the guard fires after arg-binding succeeds).
  - Line 39: `provider.features_for(["test message"], cap=5, source="17-gemmascope-2-res-16k", np_model="gemma-3-4b-it")`.
  (If you prefer the guard to fire before ANY arg binding, an alternative is to keep the guard as the first statement and the test unchanged — but `source`/`np_model` being required keyword-only means the TypeError is raised at call time, before the body runs, so the kwargs MUST be passed. Do not give them defaults — that would re-introduce the module-global coupling this task removes.)

- [ ] Step 3c (SAME commit): Update the provider-branch tests in `backend/tests/test_attribution.py` (lines 216-290). These drove `rank_method`/`preamble_skip` via `monkeypatch.setattr(fp.config, ...)`, which no longer influences the provider. Convert each to pass the kwargs and accept `source` in stubs:
  - `fake_attr` stubs (lines 219, 239, 255): change to `def fake_attr(activations, grad, keep, cap=50, baseline=None, *, source):` and emit `"source": source` instead of `"source": "s"`.
  - `fake_topk` stubs (lines 269, 282): change to `def fake_topk(act, k=15, source="s"):` and emit `"source": source`.
  - `test_provider_uses_attribution_branch_with_grad` (216-233): drop `monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")` (line 225) and pass `source="s", rank_method="attribution"` to `features_for`.
  - `test_provider_preamble_skip_filters_keep` (236-249): replace the `fp.config` `RANK_METHOD`/`PREAMBLE_SKIP` monkeypatches (244, 245) with `features_for(..., source="s", rank_method="attribution", preamble_skip=2)`.
  - `test_provider_preamble_skip_falls_back_when_too_large` (252-265): replace monkeypatches (260, 261) with `features_for(..., source="s", rank_method="attribution", preamble_skip=100)`.
  - `test_provider_falls_back_to_activation_without_grad` (268-278): pass `source="s"` (grad=None → activation path; rank_method irrelevant).
  - `test_provider_falls_back_when_attribution_raises` (281-290): replace `monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")` (287) with `features_for(..., source="s", rank_method="attribution")`.
  All assertions on `c["source"] == "s"` still hold because the tests pass `source="s"`.

- [ ] Step 4: Confirm the NP-source/model reads are gone, then run the new test PLUS every existing provider-touching test updated above, so the suite stays green in this commit.
```bash
grep -n "config.NP_SOURCE\|config.NP_MODEL" backend/science/feature_provider.py
uv run pytest backend/tests/test_feature_provider_source.py backend/tests/test_provider_masking.py backend/tests/test_feature_provider.py backend/tests/test_attribution.py -q
```
Expected: the grep prints nothing, and all four files pass:
```
.                                                                        [100%]
N passed in 0.XXs
```

- [ ] Step 5: Commit (provider signature change + ALL existing-caller/stub test fixes together — the suite is never left red).
```bash
git add backend/science/feature_provider.py backend/tests/test_feature_provider_source.py backend/tests/test_provider_masking.py backend/tests/test_feature_provider.py backend/tests/test_attribution.py
git commit -m "refactor(features): thread per-layer Neuronpedia source + ranking knobs through providers"
```

---

### Task 5: Config-derive `d_sae` + Neuronpedia source in `runtime.health_payload` and `fallback.synth_turn`

Files:
- Create: (none)
- Modify: `backend/runtime.py` (`health_payload` lines 108-124); `backend/fallback.py` (`synth_turn` lines 52-74); `backend/analyze.py` (the two `fallback.synth_turn(messages)` call sites at lines 164, 174 — production CPU fallback path, NOT a test); `backend/app.py` (the two `runtime.health_payload()` call sites at lines 46, 54 — production `/api/health` + `/api/observability`, NOT a test)
- Modify (SAME commit — existing callers of the changed signatures): `backend/tests/test_runtime.py` (line 68 `runtime.health_payload()` → pass a cfg), `backend/tests/test_fallback.py` (lines 6, 7, 17, 18 `fallback.synth_turn(...)` → pass `sae, source`)
- Test: `backend/tests/test_fallback_dims.py` (create)

Interfaces:
- Consumes: WS0-threaded `cfg: AppConfig` into `runtime.health_payload(cfg)` (per §3.2) and a `cfg.sae` / `cfg.np_source()` into `fallback.synth_turn`. `cfg.sae.d_sae` (fallback when pod reports nothing), `cfg.sae.d_in` (new health field), `cfg.np_source()`. `analyze.py`'s `cfg`/`AppConfig` handle (WS0-threaded) to source `cfg.sae`/`cfg.np_source()` for the fallback calls.
- Produces: `runtime.health_payload(cfg)` returns `d_in` (from `cfg.sae.d_in`, or `pod_health["d_in"]` when present) plus the existing `d_sae`; `fallback.synth_turn(messages, sae: SAEConfig, source: str)` deriving the synthetic index modulus from `sae.d_sae` and stamping `source` instead of `config.D_SAE`/`config.NP_SOURCE`; `analyze.py` both fallback call sites pass `cfg.sae, cfg.np_source()`.
- NOTE (cross-WS): `runtime.py` and `fallback.py` are pure-CPU modules WS0 re-signatures (`health_payload(cfg)`, etc.). WS2 only changes the d_sae/source-derivation lines inside them. `fallback.py` also carries medical canned strings WS0 genericizes — DO NOT touch those here; WS2 touches only the `index`/`source` derivation.
- NOTE (RUNTIME-CRITICAL — `analyze.py` + `app.py`): both are pure-CPU orchestration modules (NEVER import torch) behind the live HTTP API. (a) `analyze.py` calls `fallback.synth_turn(messages)` with ONE argument at lines 164 and 174 — the design doc §7 "UI runs with zero setup" CPU fallback path behind `/api/chat` and `/api/analyze`. (b) `app.py` calls `runtime.health_payload()` with NO arguments at lines 46 and 54 — `/api/health` and `/api/observability`. The moment Task 5 makes `synth_turn` require `(messages, sae, source)` and `health_payload` require `cfg`, ALL FOUR call sites crash at RUNTIME (not just in tests). These are WS0-touched shared files: WS0 must have threaded an `AppConfig` handle into both (`analyze` currently reads `config.MODEL_ID`/`config.LAYER`/`config.MAX_NEW_TOKENS` at lines 128/191/192; `app.py` does `from . import config` and reads `config.*` knobs — WS0 converts these to a `cfg`/`app.state.config` handle). PRECONDITION before Steps 3a/3b: confirm both modules have a `cfg`/`AppConfig` in scope at the call sites (module-level handle, `app.state.config`, or a param threaded into `analyze_turn`). If WS0 has NOT given them a `cfg`, STOP — that is a WS0 gap, not a WS2 workaround. Do NOT re-add `config.NP_SOURCE`/`config.D_SAE` to keep them compiling.

- [ ] Step 1: Write the failing test `backend/tests/test_fallback_dims.py`.
```python
"""WS2: fallback.synth_turn derives its synthetic feature indices from cfg.sae.d_sae and stamps
the passed Neuronpedia source — so a non-default SAE width changes the synthetic cloud's index
space and source slug. No torch (fallback is the no-torch path)."""

from __future__ import annotations

from backend.config import SAEConfig
from backend import fallback


def test_synth_turn_indices_within_configured_d_sae():
    sae = SAEConfig(d_sae=1024)
    msgs = [{"role": "user", "content": "what is the capital of france"}]
    answer, features = fallback.synth_turn(msgs, sae, source="9-gemmascope-2-res-16k")
    assert isinstance(answer, str) and answer
    assert features, "synth_turn must emit features"
    assert all(0 <= f["index"] < 1024 for f in features)
    assert {f["source"] for f in features} == {"9-gemmascope-2-res-16k"}
```

- [ ] Step 2: Run it and watch it fail because `synth_turn` takes only `messages`.
```bash
uv run pytest backend/tests/test_fallback_dims.py -q
```
Expected failure output:
```
E   TypeError: synth_turn() takes 1 positional argument but 2 were given
...
FAILED backend/tests/test_fallback_dims.py
```

- [ ] Step 3: Edit `backend/fallback.py`. Change `synth_turn` (lines 52-74) to accept the SAE config + source and use them for the index/source:
```python
def synth_turn(messages: list[dict], sae, source: str) -> tuple[str, list[dict]]:
    """Return a deterministic (answer, features) for the given conversation.

    `sae` is a SAEConfig — its d_sae sets the synthetic feature index space; `source` is the
    Neuronpedia per-layer slug stamped on each feature (caller passes cfg.np_source())."""
    seed = _seed(messages)
    rng = random.Random(seed)
    answer = rng.choice(_TEMPLATES).format(topic=_topic(messages))

    n = rng.randint(8, 12)
    labels = rng.sample(CLINICAL_LABELS, k=min(n, len(CLINICAL_LABELS)))
    features: list[dict] = []
    for i, label in enumerate(labels):
        idx = (seed >> (i * 3)) % int(sae.d_sae)
        act = round(6.4 - i * 0.42 + rng.random() * 0.25, 3)
        features.append(
            {
                "index": idx,
                "label": label,
                "act": act,
                "source": source,
                "caveat": DEFAULT_CAVEAT,
                "tracked": None,
            }
        )
    return answer, features
```
Then remove `from . import config` (line 12) IF no other `config.` reference remains in `fallback.py` (verify with grep in Step 4; the `CLINICAL_LABELS` import stays). Next edit `backend/runtime.py` `health_payload` (lines 108-124): change its signature to `def health_payload(cfg) -> dict:` (per §3.2) and add `d_in` + keep `d_sae`, both config-derived:
```python
def health_payload(cfg) -> dict:
    ph = STATE.get("pod_health") or {}
    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": cfg.model.model_id,
        "layer": cfg.model.layer,
        "d_in": ph.get("d_in", cfg.sae.d_in),
        "d_sae": ph.get("d_sae", cfg.sae.d_sae),
        "trackers": ph.get("trackers", []),
        "sae_recon_cosine": STATE.get("sae_recon_cosine", ph.get("sae_recon_cosine")),
        "sae_recon_ok": STATE.get("sae_recon_ok", ph.get("sae_recon_ok")),
        "pod_reachable": STATE.get("pod_reachable", False),
        "pod_url_configured": bool(cfg.pod.url),
        "anthropic_configured": ph.get("anthropic_configured"),
        "active_probe_jobs": ph.get("active_probe_jobs"),
    }
```
Where WS0 already converted `health_payload(cfg)` and the `model`/`layer`/`pod` reads, ONLY add the `d_in` line and switch `d_sae` to `cfg.sae.d_sae` — leave WS0's lines intact.

- [ ] Step 3a (SAME commit — RUNTIME fix, not a test): Update the two `fallback.synth_turn(messages)` call sites in `backend/analyze.py` (lines 164 and 174) to pass the SAE config + per-layer source, using the `cfg`/`AppConfig` handle WS0 threaded into `analyze` (confirm it is in scope first — see the RUNTIME-CRITICAL NOTE). Both call sites become:
```python
            answer, feats = fallback.synth_turn(messages, cfg.sae, cfg.np_source())
```
where `cfg` is whatever `AppConfig` handle `analyze` holds post-WS0 (e.g. `runtime.CFG`, a module-level `_cfg()`, or a param threaded into `analyze_turn` — match WS0's accessor). Do NOT introduce a torch import or a `config.NP_SOURCE` read. If `analyze` builds the event via `config.MODEL_ID`/`config.LAYER` at lines 191-192 and WS0 already swapped those to `cfg.model.*`, reuse the SAME `cfg`. After the edit, grep to prove no one-arg call survives (Step 4).

- [ ] Step 3b (SAME commit — RUNTIME fix, not a test): Update the two `runtime.health_payload()` call sites in `backend/app.py` (lines 46 and 54) to pass the `cfg` `health_payload` now requires, using the `AppConfig` handle WS0 exposes in the CPU backend (confirm it is in scope first — see the RUNTIME-CRITICAL NOTE). Both become:
```python
    return runtime.health_payload(cfg)          # line 46, /api/health
    ...
    snap["health"] = runtime.health_payload(cfg)  # line 54, /api/observability
```
where `cfg` is WS0's handle (e.g. `request.app.state.config`, a module-level `_cfg()`, or `runtime.CFG`). If `/api/health` is a no-arg handler, fetch `cfg` from the app/state per WS0's accessor. Do NOT re-add a `config.D_SAE`/`config.D_IN` read. After the edit, grep to prove no no-arg call survives (Step 4).

- [ ] Step 3c (SAME commit — existing-caller test fixes so the suite is never red): update the two existing tests that drive the OLD signatures:
  - `backend/tests/test_runtime.py:68`: `p = runtime.health_payload()` → `p = runtime.health_payload(AppConfig())` (add `from backend.config import AppConfig` at the top). The existing shape assertions still hold; add `"d_in"` to the asserted key set since `health_payload` now emits it.
  - `backend/tests/test_fallback.py` (lines 6, 7, 17, 18): every `fallback.synth_turn(msgs)` / `fallback.synth_turn([...])` call must pass `sae, source`. Add `from backend.config import SAEConfig` and a module constant, e.g. `_SAE = SAEConfig()` / `_SRC = "17-gemmascope-2-res-16k"`, then call `fallback.synth_turn(msgs, _SAE, _SRC)`. The deterministic-output and varies-by-prompt assertions are unaffected (same `sae`/`source` both calls); the `f["source"] == "17-gemmascope-2-res-16k"` assertion in `test_synth_turn_is_deterministic` still holds because `_SRC` is that value.

- [ ] Step 4: Confirm the config global reads are gone in fallback AND no one-arg `synth_turn` / no-arg `health_payload` calls survive in the CPU modules, then run the new test PLUS the two updated existing tests.
```bash
grep -n "config.D_SAE\|config.NP_SOURCE" backend/fallback.py
grep -n "synth_turn(messages)" backend/analyze.py || echo "NO ONE-ARG SYNTH_TURN"
grep -n "health_payload()" backend/app.py || echo "NO NO-ARG HEALTH_PAYLOAD"
uv run pytest backend/tests/test_fallback_dims.py backend/tests/test_fallback.py backend/tests/test_runtime.py -q
```
Expected: the first grep prints nothing, the second prints `NO ONE-ARG SYNTH_TURN`, the third prints `NO NO-ARG HEALTH_PAYLOAD`, and all three test files pass:
```
.                                                                        [100%]
N passed in 0.XXs
```

- [ ] Step 5: Commit (runtime + fallback config-derivation together with ALL production call-site fixes — analyze.py + app.py — and the existing-caller test fixes, so the CPU fallback/health paths never crash and the suite is never left red between commits).
```bash
git add backend/runtime.py backend/fallback.py backend/analyze.py backend/app.py backend/tests/test_fallback_dims.py backend/tests/test_fallback.py backend/tests/test_runtime.py
git commit -m "refactor(runtime): config-derive d_in/d_sae and Neuronpedia source in health + fallback"
```

---

### Task 6: Refresh stale Gemma display defaults in `schema.py`

Files:
- Create: (none)
- Modify: `backend/schema.py` (`Feature.source` line 40; `CognitionEvent.model` line 57; `CognitionEvent.layer` line 58)
- Test: `backend/tests/test_schema_defaults.py` (create)

Interfaces:
- Consumes: nothing new (schema is the wire contract; no config import — schema stays a pure pydantic module). The intent is to stop presenting `"17-gemmascope-2-res-16k"` / `"unsloth/gemma-3-4b-it"` / `17` as canonical magic strings and label them as the Gemma demo example via field descriptions, since the live values now come from `cfg`/`sae.dims`.
- Produces: `Feature.source` keeps the Gemma default value but documents it as "Gemma-Scope example; live events carry cfg.np_source(layer)"; `CognitionEvent.model`/`layer` documented as "Gemma default; overwritten per-event from cfg.model". Wire shape (field names, types, `schema_version`) UNCHANGED.
- NOTE (cross-WS): WS1 renames `class CognitionEvent -> IntrospectionEvent` in this same file. Sequence with WS1: if WS1 has landed, edit the renamed class; the field defaults this task changes are identical either way. Coordinate so the rename and the default-description edits don't conflict.

- [ ] Step 1: Write the failing test `backend/tests/test_schema_defaults.py`.
```python
"""WS2: schema defaults are labeled Gemma EXAMPLES, not canonical constants. The wire shape is
unchanged (field names/types/defaults stay), but every Gemma magic string carries a field
description marking it as the demo default that live events overwrite from cfg. This test pins
the descriptions so a future reviewer can't silently re-canonicalize the medical/Gemma strings."""

from __future__ import annotations

from backend import schema

# WS1 may rename CognitionEvent -> IntrospectionEvent; bind to whichever exists.
Event = getattr(schema, "IntrospectionEvent", None) or schema.CognitionEvent


def test_feature_source_default_is_documented_as_example():
    fld = schema.Feature.model_fields["source"]
    assert fld.default == "17-gemmascope-2-res-16k"  # wire default unchanged
    assert fld.description and "example" in fld.description.lower()


def test_event_model_and_layer_documented_as_gemma_default():
    mfld = Event.model_fields["model"]
    lfld = Event.model_fields["layer"]
    assert mfld.default == "unsloth/gemma-3-4b-it"
    assert lfld.default == 17
    assert mfld.description and "default" in mfld.description.lower()
    assert lfld.description and "default" in lfld.description.lower()
```

- [ ] Step 2: Run it and watch it fail because the fields have no descriptions.
```bash
uv run pytest backend/tests/test_schema_defaults.py -q
```
Expected failure output:
```
E   AssertionError: assert (None and ...)
...
FAILED backend/tests/test_schema_defaults.py::test_feature_source_default_is_documented_as_example
```

- [ ] Step 3: Edit `backend/schema.py`. Change the `Feature.source` field (line 40):
```python
    source: str = Field(
        "17-gemmascope-2-res-16k",
        description="Neuronpedia per-layer slug. Gemma-Scope example default; live events carry cfg.np_source(layer).",
    )
```
Change the `CognitionEvent.model` and `layer` fields (lines 57, 58) — within the class WS1 may have renamed to `IntrospectionEvent`:
```python
    model: str = Field(
        "unsloth/gemma-3-4b-it",
        description="Model under inspection. Gemma default; live events set this from cfg.model.model_id.",
    )
    layer: int = Field(
        17, description="Hooked residual layer. Gemma default; live events set this from cfg.model.layer."
    )
```
Ensure `Field` is imported (it already is at line 10).

- [ ] Step 4: Run the test.
```bash
uv run pytest backend/tests/test_schema_defaults.py -q
```
Expected pass output:
```
..                                                                       [100%]
2 passed in 0.XXs
```

- [ ] Step 5: Commit.
```bash
git add backend/schema.py backend/tests/test_schema_defaults.py
git commit -m "docs(schema): label Gemma source/model/layer defaults as demo examples"
```

---

### Task 7: Refresh stale Gemma docstrings/defaults in `config.py` (SAEConfig + ModelConfig)

Files:
- Create: (none)
- Modify: `backend/config.py` (the WS0 `SAEConfig` and `ModelConfig` docstrings + the per-field comments lifted from the old globals — the stale "LOCKED" / medical-specific comments)
- Test: `backend/tests/test_config_sae_seam.py` (create)

Interfaces:
- Consumes: the WS0 `AppConfig`/`SAEConfig`/`ModelConfig` models verbatim. This task adds no fields — it only verifies the WS2-relevant defaults are present and the helpers render correctly, and rewrites stale prose.
- Produces: a test that pins `cfg.sae_id(layer)` and `cfg.np_source(layer)` rendering against the Gemma `sae_id_pattern`/`np_source_pattern`, plus `SAEConfig` defaults (`release`, `d_in`, `d_sae`, `np_model`). No signature change.
- NOTE (cross-WS): `backend/config.py` is WS0-owned. WS2 must land AFTER WS0. This task is purely additive (new test) + prose edits to docstrings/comments; if WS0's docstrings already read as generic Gemma defaults, the prose edit is a no-op and the task is just the test + a doc-accuracy commit.
- NOTE (TDD posture): this is a DOC/REGRESSION task, NOT red-green feature TDD. `test_sae_defaults_are_gemma_scope`/`test_sae_id_and_np_source_render_for_layer` are regression PINS on WS0's defaults (they pass immediately if WS0 shipped the contract verbatim — they drive no WS2 code change). The only WS2 production change here is prose. To keep the prose change actually GUARDED (not just grepped), Step 1 ALSO adds `test_config_docstrings_are_generic`, which asserts the `SAEConfig`/`ModelConfig` docstrings contain no `LOCKED`/medical tokens — so a future re-canonicalization of the stale language fails a test, not just the manual grep.

- [ ] Step 1: Write the failing test `backend/tests/test_config_sae_seam.py`.
```python
"""WS2: the SAE seam renders the Gemma-Scope sae_id and Neuronpedia source from config patterns.
Locks the defaults WS2 relies on and proves cfg.sae_id()/cfg.np_source() substitute the layer."""

from __future__ import annotations

from backend.config import AppConfig, SAEConfig


def test_sae_defaults_are_gemma_scope():
    sae = SAEConfig()
    assert sae.release == "gemma-scope-2-4b-it-res"
    assert sae.sae_id_pattern == "layer_{layer}_width_16k_l0_medium"
    assert sae.np_source_pattern == "{layer}-gemmascope-2-res-16k"
    assert sae.np_model == "gemma-3-4b-it"
    assert (sae.d_in, sae.d_sae) == (2560, 16384)


def test_sae_id_and_np_source_render_for_layer():
    cfg = AppConfig()  # default model.layer == 17
    assert cfg.sae_id() == "layer_17_width_16k_l0_medium"
    assert cfg.sae_id(22) == "layer_22_width_16k_l0_medium"
    assert cfg.np_source() == "17-gemmascope-2-res-16k"
    assert cfg.np_source(9) == "9-gemmascope-2-res-16k"


def test_config_docstrings_are_generic():
    """Guards the WS2 prose change: the SAE/model sub-config docstrings must NOT re-introduce
    'LOCKED' or medical-positioning language. This is the only automated check on the doc edit
    (Step 4's grep is manual); it FAILS the build if someone re-canonicalizes the stale strings."""
    import re

    from backend.config import ModelConfig, SAEConfig

    banned = re.compile(r"locked|clinical|medical|patient", re.IGNORECASE)
    for cls in (SAEConfig, ModelConfig):
        doc = cls.__doc__ or ""
        assert not banned.search(doc), f"{cls.__name__} docstring still carries stale language: {doc!r}"
```

- [ ] Step 2: Run it. The two default/render tests are REGRESSION PINS — if WS0 shipped the contract verbatim they PASS immediately (they drive no WS2 code change); if any default drifted they FAIL, flagging a contract mismatch to fix. `test_config_docstrings_are_generic` FAILS first (red) if WS0 left "LOCKED"/medical language in the docstrings, and Step 3's prose edit turns it green — this is the one genuinely red-green check in the task.
```bash
uv run pytest backend/tests/test_config_sae_seam.py -q
```
Expected (contract intact; docstrings may still be red until Step 3):
```
F..   [or all green if WS0 already wrote generic docstrings]
```
If a DEFAULT test fails, the WS0 defaults diverged from the AppConfig contract — reconcile WS0 before continuing. If only `test_config_docstrings_are_generic` fails, that is the expected red state Step 3 fixes.

- [ ] Step 3: Refresh the stale prose in `backend/config.py`. Open the `SAEConfig` and `ModelConfig` classes and ensure their docstrings/comments read as Gemma DEFAULTS of a generic seam — not "LOCKED" and not medical-specific. Make the `SAEConfig` docstring exactly:
```python
class SAEConfig(BaseModel):
    """The SAE under inspection + its Neuronpedia label source (the swap seam).

    Gemma-Scope defaults. v1 is validated on Gemma + Gemma-Scope only; arbitrary HF SAE repos
    are the additive OSS follow-up (issue #1). d_in/d_sae are FALLBACKS — the live SAE's true
    dims, read via science.sae.dims(), win at runtime (WS2)."""
```
And the `ModelConfig` docstring exactly:
```python
class ModelConfig(BaseModel):
    """The LLM under inspection (the model seam). Gemma specifics — mask_tokens, preamble_skip,
    and the system-role-merge fallback — live here as Gemma DEFAULTS, not hardcoded constants."""
```
If WS0 already wrote equivalent generic docstrings, leave them; the requirement is only that no "LOCKED" or medical-positioning language survives in these two classes. Verify with the grep in Step 4.

- [ ] Step 4: Confirm no stale "LOCKED"/medical language remains in the SAE/model config prose, then re-run the test (now green, including the docstring guard).
```bash
grep -niE "LOCKED|clinical|medical|patient" backend/config.py
uv run pytest backend/tests/test_config_sae_seam.py -q
```
Expected: the grep prints nothing (the medical SYSTEM_PROMPT default was already moved to a labeled profile by WS0; if WS0 left a labeled medical example comment it is acceptable — confirm it is clearly labeled as the opt-in example), and all three tests pass:
```
...                                                                      [100%]
3 passed in 0.XXs
```

- [ ] Step 5: Commit.
```bash
git add backend/config.py backend/tests/test_config_sae_seam.py
git commit -m "docs(config): describe SAE/model sub-configs as Gemma defaults of the swap seam"
```

---

### Task 8: Refresh stale Gemma references in frontend `probes.ts` and `mock.ts`

Files:
- Create: (none)
- Modify: `frontend/src/probes.ts` (header comment line 1); `frontend/src/mock.ts` (the `source` string at line 15 + the demo-event comment at line 19)
- Test: verification by `grep` + `tsc` (no JS test runner is configured — `frontend/package.json` has no vitest/jest). NOTE: this is a grep/tsc-verified-only task — there is NO automated test coverage of the display strings. The `source` string change in `mock.ts` is purely cosmetic; it changes no exported type or shape (`Feature.source` stays `string`), so `tsc --noEmit` passes but will NOT catch a wrong literal — the Step 4 `grep` is the only guard.

Interfaces:
- Consumes: nothing — these are display-only strings. The change makes the stale backend pointer accurate (`config.ENABLED_TRACKERS` → `cfg.probes.enabled`) and labels the demo `source`/medical strings as the demo profile rather than canonical truth.
- Produces: no API/type change — `tsc` must still pass and the exported shapes are unchanged.
- NOTE (cross-WS): WS1 removes `phoenix_ui_url` from `mock.ts`. WS2 touches only the `source` string + comments — keep edits to non-overlapping lines and coordinate the commit so WS1's Phoenix removal isn't reverted.

- [ ] Step 1: Establish the baseline — show the stale references that must change.
```bash
grep -n "backend.config.ENABLED_TRACKERS" frontend/src/probes.ts
grep -n "Ibuprofen / 3rd trimester\|17-gemmascope-2-res-16k" frontend/src/mock.ts
```
Expected (pre-change) output:
```
frontend/src/probes.ts:1:// Active built-in probe set — must match backend.config.ENABLED_TRACKERS.
frontend/src/mock.ts:15:  index, label, act, source: "17-gemmascope-2-res-16k",
frontend/src/mock.ts:19:// Ibuprofen / 3rd trimester confident-wrong case — uses the live probe pair only.
```
(The first grep pattern matches the demo-event comment on line 19; the second pattern matches the `source` constant on line 15. Line numbers may differ if WS1 already removed Phoenix; rely on the matched text, not the numbers.)

- [ ] Step 2: Edit `frontend/src/probes.ts`. Replace the stale backend pointer in the header comment (line 1):
```typescript
// Active built-in probe set — must match the backend cfg.probes.enabled list.
```

- [ ] Step 3: Edit `frontend/src/mock.ts`. Update the comment above `DEMO_EVENT` (line 19) so it reads as the labeled demo profile rather than the product's framing, and annotate the Gemma demo `source` constant (line 15) as the demo default:
```typescript
const f = (index: number, label: string, act: number, suspect = false, tracked: string | null = null): Feature => ({
  index, label, act, source: "17-gemmascope-2-res-16k", // Gemma-Scope demo default; live events carry cfg.np_source
  caveat: suspect ? SUSPECT_CAVEAT : DEFAULT_CAVEAT, tracked,
});
```
And the demo-event comment:
```typescript
// Demo profile (medical example): ibuprofen / 3rd-trimester confident-wrong case — uses the live probe pair only.
```

- [ ] Step 4: Verify the stale pointer is gone and the frontend still type-checks.
```bash
grep -n "backend.config.ENABLED_TRACKERS" frontend/src/probes.ts || echo "STALE POINTER GONE"
cd frontend && npx tsc --noEmit && echo "TSC OK"
```
Expected output:
```
STALE POINTER GONE
TSC OK
```

- [ ] Step 5: Commit.
```bash
git add frontend/src/probes.ts frontend/src/mock.ts
git commit -m "docs(frontend): point probe comment at cfg.probes.enabled; label demo Gemma strings"
```

---

### Task 9: Full-suite regression gate

Files:
- Create: (none)
- Modify: (none)
- Test: the entire `backend/tests` suite (non-GPU subset must be green; GPU-gated `engine`/`sae` numeric tests are skipped without weights)

Interfaces:
- Consumes: all prior tasks' changes.
- Produces: a green non-GPU suite proving the WS2 config-seam refactor did not break any import surface or contract.

- [ ] Step 1: Run the full backend suite.
```bash
uv run pytest -q
```
Expected: every previously-passing test still passes, plus the SIX WS2 test files (`test_sae_dims.py`, `test_sae_config_seam.py`, `test_feature_provider_source.py`, `test_fallback_dims.py`, `test_schema_defaults.py`, `test_config_sae_seam.py`) pass. GPU-gated device tests in `test_persona.py` skip on CPU. Output ends:
```
N passed, M skipped in X.XXs
```

- [ ] Step 2: BACKSTOP (should already be green). Every existing test caller of a changed signature is updated in the SAME commit as its production change — so the suite is never left red between commits. The enumerated old-signature callers and where they were already fixed:
  - Provider/topk callers (the LARGEST breakage): `backend/tests/test_provider_masking.py` (lines 16, 22, 33, 39), `backend/tests/test_feature_provider.py` (lines 13, 39), `backend/tests/test_attribution.py` (direct `attribution_topk` calls at 37/50/60/81 and the `config.NP_SOURCE` assertion at 41 → Task 2 Step 3a; the provider-branch `fake_attr`/`fake_topk` stubs at 219/239/255/269/282 and the `fp.config` `RANK_METHOD`/`PREAMBLE_SKIP` monkeypatches at 225/244/245/260/261/287 → Task 4 Step 3c).
  - `synth_turn(messages)` callers: production `backend/analyze.py:164,174` → Task 5 Step 3a; tests `backend/tests/test_fallback.py:6,7,17,18` → Task 5 Step 3c.
  - `health_payload()` callers: production `backend/app.py:46,54` → Task 5 Step 3b; test `backend/tests/test_runtime.py:68` → Task 5 Step 3c. (Also verify `backend/tests/test_observability_endpoint.py` and `backend/tests/test_api.py` exercise `/api/health`/`/api/observability` through the app — if they build the app without setting `app.state.config`, they need WS0's startup wiring or a cfg fixture; reconcile here if red.)
  - `load_sae()`/`reconstruction_error(act)`: `backend/tests/test_attribution.py` direct callers → Task 2 Step 3a; GPU-gated production callers (`gpu_service`) → Task 3.
  If, despite the per-task fixes, the full suite surfaces a MISSED old-signature caller here, open that test, identify the call, and update it to the new signature established in Tasks 2-5 — do NOT revert the production change.
```bash
uv run pytest -q 2>&1 | grep -E "FAILED|Error" | head
```
Expected after fixes: no `FAILED` lines.

- [ ] Step 3: Re-run to confirm green.
```bash
uv run pytest -q
```
Expected:
```
N passed, M skipped in X.XXs
```

- [ ] Step 4: Confirm no stray `config.SAE_RELEASE`/`config.sae_id_for_layer`/`config.NP_SOURCE`/`config.D_SAE`/`config.D_IN` reads remain in the WS2-owned modules.
```bash
grep -rn "config.SAE_RELEASE\|config.sae_id_for_layer\|config.NP_SOURCE\|config.NP_MODEL\|config.D_SAE\|config.D_IN" backend/science/sae.py backend/science/feature_provider.py backend/gpu_service.py backend/fallback.py backend/runtime.py || echo "NO STALE SAE GLOBALS"
```
Expected output:
```
NO STALE SAE GLOBALS
```

- [ ] Step 5: Commit (records the regression gate as passed; if Step 2 changed any existing test, this commit carries those edits).
```bash
git add -A
git commit -m "test(ws2): green full backend suite after model/SAE config seam"
```
