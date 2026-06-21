# Chatbot ↔ SAE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the GlassBox multi-turn chat to a real generate → SAE → label → cognition-event pipeline, streamed to a frontend whose feature map illuminates the firing features, with a deterministic synthetic fallback so it runs end-to-end on any machine.

**Architecture:** A FastAPI backend picks its path per-request from a background-loaded readiness state: the **real** path runs the model + SAE and labels features via Neuronpedia; the **fallback** path produces a deterministic synthetic turn. Both assemble exactly one `CognitionEvent` (Family-B uncertainty fields null until probes are trained) and stream NDJSON token lines then one event line. The React frontend holds a multi-turn thread, renders the streamed answer, and animates `event.features` in the canvas feature field.

**Tech Stack:** Python 3.12, FastAPI, pydantic, `uv`; torch/transformers/sae-lens (optional `ml` extra, lazy-imported); React 18 + Vite + TypeScript; pytest for backend tests.

## Global Constraints

- Real-path ML imports (`torch`, `transformers`, `sae_lens`, `sklearn`) MUST stay lazy (inside functions / the load thread). Importing any `backend.*` module MUST NOT import torch. — verbatim from spec §8.
- The contract lives in three mirrored files changed in lockstep: `backend/schema.py`, `frontend/src/types.ts`, `fixtures/cognition_event.sample.json`. — spec §5.
- Family B is WIP: `uncertainty` is **null** when no probes are registered; `flag` is then `False`, `severity` `"info"`, `trackers` `{}`. — spec §1, §5.
- Event identity values come from `config`: model `unsloth/gemma-3-4b-it`, layer `17`, feature source `17-gemmascope-2-res-16k`. — spec §4.
- `/api/chat` wire protocol: zero+ `{"type":"token",...}` NDJSON lines, then exactly one `{"type":"event", ...CognitionEvent}` line; `fanout()` is called AFTER the event line. — README contract #2, spec §4.
- The "define a probe" box stays a visible, clearly-marked **preview** (no real `synth_concept`). — spec §6.
- Run backend tests with `uv run pytest` from the repo root.

---

### Task 1: Dependencies & test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `uv run pytest`; an `ml` optional-dependency group for the real path; `httpx` as a runtime dependency (needed by `labels.py` and FastAPI `TestClient`).

- [ ] **Step 1: Rewrite `pyproject.toml`**

```toml
[project]
name = "glassbox"
version = "0.1.0"
description = "Cognition-observability for medical LLMs"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.138.0",
    "sentry-sdk[fastapi]>=2.63.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
# Real inference path. Installed only where weights/GPU exist:  uv sync --extra ml
# Match the validated env in PLAN.md §1 (torch 2.12 / transformers 5.12 / sae-lens 6).
ml = [
    "torch>=2.4",
    "transformers>=4.50",
    "sae-lens>=6.0",
    "scikit-learn>=1.5",
    "accelerate>=0.34",
    "anthropic>=0.40",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
addopts = "-q"
```

- [ ] **Step 2: Create the test package + a smoke test**

`backend/tests/__init__.py`: empty file.

`backend/tests/test_smoke.py`:

```python
def test_core_modules_import_without_torch():
    """Importing the CPU surface must never pull torch (Global Constraint)."""
    import sys
    import backend.schema
    import backend.events
    import backend.config

    assert backend.schema.SCHEMA_VERSION == "1.0"
    assert "torch" not in sys.modules
```

- [ ] **Step 3: Run the test (verifies harness + the no-torch invariant)**

Run: `uv run pytest backend/tests/test_smoke.py`
Expected: 1 passed. (`uv` resolves the light env: fastapi, sentry-sdk, httpx, pytest — no torch.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock backend/tests/__init__.py backend/tests/test_smoke.py
git commit -m "build: add ml extra, httpx, pytest harness"
```

---

### Task 2: Nullable Family-B contract

**Files:**
- Modify: `backend/schema.py`
- Modify: `backend/events.py`
- Modify: `fixtures/cognition_event.sample.json`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/mock.ts`
- Create: `backend/tests/test_events.py`

**Interfaces:**
- Consumes: `backend.schema.CognitionEvent`.
- Produces: `events.build_cognition_event(*, message_id, ts, user_msg, response, trackers: dict, features: list, model: str, layer: int) -> CognitionEvent` — when `trackers` has no `"uncertainty"` key, the event's `uncertainty`/`uncertainty_proj`/`uncertainty_proj_pre` are `None`, `flag` is `False`, `severity` is `"info"`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_events.py`:

```python
from backend.events import build_cognition_event


def test_empty_trackers_yield_null_uncertainty():
    ev = build_cognition_event(
        message_id="m1", ts=1.0, user_msg="q", response="a",
        trackers={}, features=[], model="unsloth/gemma-3-4b-it", layer=17,
    )
    assert ev.uncertainty is None
    assert ev.uncertainty_proj is None
    assert ev.uncertainty_proj_pre is None
    assert ev.flag is False
    assert ev.severity == "info"
    assert ev.trackers == {}
    assert ev.model == "unsloth/gemma-3-4b-it"
    assert ev.layer == 17


def test_uncertainty_tracker_populates_meter():
    trackers = {
        "uncertainty": {
            "score": 0.83, "proj": 1.27, "proj_pre": 0.91, "flag": True,
            "reliable": True, "status": "ready", "user_defined": False,
        }
    }
    ev = build_cognition_event(
        message_id="m2", ts=1.0, user_msg="q", response="a",
        trackers=trackers, features=[], model="m", layer=17,
    )
    assert ev.uncertainty == 0.83
    assert ev.uncertainty_proj == 1.27
    assert ev.flag is True
    assert ev.severity == "warning"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest backend/tests/test_events.py`
Expected: FAIL — current `build_cognition_event` returns `uncertainty=0.0` (not `None`) and the schema rejects `None`.

- [ ] **Step 3: Make Family-B fields nullable in `backend/schema.py`**

Replace the `CognitionEvent` Family-B block and the defaults. The full edited class:

```python
class CognitionEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    type: Literal["event"] = "event"  # discriminator vs streamed {"type":"token"} lines
    message_id: str
    ts: float
    model: str = "unsloth/gemma-3-4b-it"
    layer: int = 17
    io: IO

    # --- Family B: WIP. Null until calibrated probes are registered. ---
    uncertainty: float | None = Field(None, description="calibrated prob [0,1]; None when no probes")
    uncertainty_proj: float | None = None
    uncertainty_proj_pre: float | None = None
    flag: bool = Field(False, description="uncertainty >= threshold; False when no probes")
    severity: Severity = "info"

    trackers: dict[str, Tracker] = Field(default_factory=dict)

    # --- Family A: EXPLORATORY, labels unreliable ---
    features: list[Feature] = Field(default_factory=list)

    # --- Anthropic prize, present only when flagged (filled async) ---
    adjudication: Adjudication | None = None
```

Also update `Feature.source` default in the same file:

```python
    source: str = "17-gemmascope-2-res-16k"
```

- [ ] **Step 4: Rewrite `backend/events.py`**

```python
"""Assemble the ONE cognition_event per message. Pure CPU. No torch, no tensors out.

OWNER: Lane A.
"""

from __future__ import annotations

from .config import DEFAULT_THRESHOLD
from .schema import IO, CognitionEvent


def build_cognition_event(
    *,
    message_id: str,
    ts: float,
    user_msg: str,
    response: str,
    trackers: dict,
    features: list,
    model: str,
    layer: int,
) -> CognitionEvent:
    """trackers: dict[str -> {score,proj,flag,...}] from persona.score_all_trackers ({} until
    probes exist). features: list[Feature-like dicts] with labels already attached.

    Family B is WIP: with no "uncertainty" tracker, the meter fields stay null and flag is False.
    """
    unc = trackers.get("uncertainty")
    if unc is None:
        uncertainty = uncertainty_proj = uncertainty_proj_pre = None
        flag = False
    else:
        uncertainty = unc.get("score")
        uncertainty_proj = unc.get("proj")
        uncertainty_proj_pre = unc.get("proj_pre")
        flag = bool(unc.get("flag", (uncertainty or 0.0) >= DEFAULT_THRESHOLD))
    return CognitionEvent(
        message_id=message_id,
        ts=ts,
        model=model,
        layer=layer,
        io=IO(user_msg=user_msg, response=response),
        uncertainty=uncertainty,
        uncertainty_proj=uncertainty_proj,
        uncertainty_proj_pre=uncertainty_proj_pre,
        flag=flag,
        severity="warning" if flag else "info",
        trackers=trackers,
        features=features,
        adjudication=None,
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest backend/tests/test_events.py`
Expected: 2 passed.

- [ ] **Step 6: Update `fixtures/cognition_event.sample.json` to the gemma-3 / L17 identity**

Change `"model"` to `"unsloth/gemma-3-4b-it"`, `"layer"` to `17`, and every feature `"source"` from `"12-gemmascope-res-16k"` to `"17-gemmascope-2-res-16k"`. Leave the flagged uncertainty values as-is (the fixture exercises the non-null branch). Verify it still parses:

Run: `uv run python -c "import json,backend.schema as s; s.CognitionEvent(**json.load(open('fixtures/cognition_event.sample.json')))"`
Expected: no output, exit 0.

- [ ] **Step 7: Mirror nullability in `frontend/src/types.ts`**

```typescript
  // Family B — reliable signal; null until calibrated probes are registered (WIP)
  uncertainty: number | null;
  uncertainty_proj: number | null;
  uncertainty_proj_pre?: number | null;
  flag: boolean;
  severity: Severity;
  trackers: Record<string, Tracker>;
```

- [ ] **Step 8: Align `frontend/src/mock.ts` identity**

In `DEMO_EVENT` set `model: "unsloth/gemma-3-4b-it"` and `layer: 17`; in the `f()` helper change `source: "12-gemmascope-res-16k"` to `source: "17-gemmascope-2-res-16k"`.

- [ ] **Step 9: Typecheck the frontend**

Run: `cd frontend && npm install && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 10: Commit**

```bash
git add backend/schema.py backend/events.py fixtures/cognition_event.sample.json frontend/src/types.ts frontend/src/mock.ts backend/tests/test_events.py
git commit -m "feat: nullable Family-B contract (uncertainty empty until probes exist)"
```

---

### Task 3: Runtime readiness state + health payload

**Files:**
- Create: `backend/runtime.py`
- Create: `backend/tests/test_runtime.py`

**Interfaces:**
- Consumes: `backend.config` (MODEL_ID, LAYER); lazily `backend.engine`, `backend.science.sae`, `backend.science.persona`.
- Produces:
  - `runtime.STATE: dict` with keys `mode` (`"loading"|"real"|"fallback"`), `model_loaded: bool`, `sae_loaded: bool`.
  - `runtime._attempt_load(*, torch_available=None, load_engine=None, load_sae=None) -> dict` — sets/returns STATE; never raises.
  - `runtime.start_loading() -> None` — sets mode `"loading"`, spawns a daemon thread running `_attempt_load`.
  - `runtime.health_payload() -> dict` — `{mode, model_loaded, sae_loaded, model, layer, trackers}`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_runtime.py`:

```python
from backend import runtime


def test_attempt_load_falls_back_when_torch_absent():
    state = runtime._attempt_load(torch_available=False)
    assert state["mode"] == "fallback"
    assert state["model_loaded"] is False
    assert state["sae_loaded"] is False


def test_attempt_load_goes_real_with_injected_loaders():
    state = runtime._attempt_load(
        torch_available=True, load_engine=lambda: None, load_sae=lambda: None
    )
    assert state["mode"] == "real"
    assert state["model_loaded"] is True
    assert state["sae_loaded"] is True


def test_attempt_load_falls_back_on_loader_error():
    def boom():
        raise RuntimeError("no weights")

    state = runtime._attempt_load(torch_available=True, load_engine=boom)
    assert state["mode"] == "fallback"


def test_health_payload_shape():
    p = runtime.health_payload()
    assert set(p) == {"mode", "model_loaded", "sae_loaded", "model", "layer", "trackers"}
    assert p["layer"] == 17
    assert isinstance(p["trackers"], list)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest backend/tests/test_runtime.py`
Expected: FAIL — `No module named 'backend.runtime'`.

- [ ] **Step 3: Create `backend/runtime.py`**

```python
"""Backend readiness. A background thread tries to load the real model + SAE; until/unless
that succeeds, requests use the synthetic fallback. State is read per-request by the API.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import threading

from . import config

STATE: dict = {"mode": "loading", "model_loaded": False, "sae_loaded": False}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _default_load_engine() -> None:
    from . import engine

    engine.load_engine()


def _default_load_sae() -> None:
    from .science import sae

    sae.load_sae()


def _attempt_load(*, torch_available=None, load_engine=None, load_sae=None) -> dict:
    """Try to bring the real path online. Never raises — failure => fallback."""
    avail = _torch_available() if torch_available is None else torch_available
    if not avail:
        STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
        return dict(STATE)
    try:
        (load_engine or _default_load_engine)()
        STATE["model_loaded"] = True
        (load_sae or _default_load_sae)()
        STATE["sae_loaded"] = True
        STATE["mode"] = "real"
    except Exception as e:  # noqa: BLE001 — degrade, never crash the app
        print(f"[runtime] real load failed ({e}); using synthetic fallback")
        STATE.update(mode="fallback")
    return dict(STATE)


def start_loading() -> None:
    """Kick off the load off the request path. Called once at FastAPI startup."""
    STATE["mode"] = "loading"
    threading.Thread(target=_attempt_load, daemon=True).start()


def health_payload() -> dict:
    from .science.persona import _trackers

    return {
        "mode": STATE["mode"],
        "model_loaded": STATE["model_loaded"],
        "sae_loaded": STATE["sae_loaded"],
        "model": config.MODEL_ID,
        "layer": config.LAYER,
        "trackers": list(_trackers.keys()),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest backend/tests/test_runtime.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/runtime.py backend/tests/test_runtime.py
git commit -m "feat: per-request backend readiness state + health payload"
```

---

### Task 4: Fix LocalSAEProvider special-token masking

**Files:**
- Modify: `backend/science/feature_provider.py`
- Create: `backend/tests/test_provider_masking.py`

**Interfaces:**
- Produces: `FeatureProvider.features_for(self, text, activations=None, token_ids=None, special_ids=None, k=config.TOPK, cap=config.TOPK_EVENT) -> list[dict]`. `token_ids` is per-position (aligned to `activations` rows); `special_ids` is the set of token ids to skip. Returns deduped, activation-ranked `{index, act, source}` dicts (no labels).

**Context:** the current `features_for` computes `special = set(token_ids or [])` then skips every `pos` whose `token_ids[pos] in special` — i.e. it skips *every* position and returns `[]`. This task fixes the contract to take a separate `special_ids` set.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_provider_masking.py`:

```python
from backend.science import feature_provider as fp


class FakeActs:
    """Minimal stand-in for a [n, d] activation tensor: exposes .shape and row indexing."""

    def __init__(self, n: int):
        self.shape = (n, 4)

    def __getitem__(self, pos: int):
        return ("row", pos)


def test_local_provider_skips_special_positions(monkeypatch):
    # one synthetic feature per position, index encodes the position
    def fake_topk(act, k=15):
        pos = act[1]
        return [{"index": 100 + pos, "act": float(10 - pos), "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    prov = fp.LocalSAEProvider()
    feats = prov.features_for(
        "txt", activations=FakeActs(4), token_ids=[1, 999, 2, 3], special_ids={999}
    )
    idxs = {f["index"] for f in feats}
    assert 101 not in idxs                       # position 1 (token 999) masked
    assert {100, 102, 103} <= idxs               # other positions kept
    # ranked by activation descending
    assert feats[0]["act"] >= feats[-1]["act"]


def test_local_provider_no_special_keeps_all(monkeypatch):
    def fake_topk(act, k=15):
        pos = act[1]
        return [{"index": 100 + pos, "act": 1.0, "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    prov = fp.LocalSAEProvider()
    feats = prov.features_for("txt", activations=FakeActs(3), token_ids=[1, 2, 3])
    assert {f["index"] for f in feats} == {100, 101, 102}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest backend/tests/test_provider_masking.py`
Expected: FAIL — `features_for` got an unexpected `special_ids` (and currently masks everything).

- [ ] **Step 3: Fix the base + Local + Neuronpedia signatures**

In `backend/science/feature_provider.py`, update the base method signature:

```python
    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        k: int = config.TOPK,
        cap: int = config.TOPK_EVENT,
    ) -> list[dict]:
        """Return up to `cap` deduped, activation-ranked {index, act, source} dicts.
        token_ids: per-position ids aligned to `activations`. special_ids: ids to skip.
        Labels are attached downstream."""
        raise NotImplementedError
```

Replace `LocalSAEProvider.features_for` with:

```python
    def features_for(
        self,
        text,
        activations=None,
        token_ids=None,
        special_ids=None,
        k: int = config.TOPK,
        cap: int = config.TOPK_EVENT,
    ) -> list[dict]:
        if activations is None:
            raise ValueError(
                "LocalSAEProvider needs captured activations [n_positions, d_in]"
            )
        special = set(special_ids or [])
        best: dict[int, float] = {}
        for pos in range(activations.shape[0]):
            if token_ids is not None and pos < len(token_ids) and token_ids[pos] in special:
                continue
            for f in sae_topk(activations[pos], k=k):
                best[f["index"]] = max(best.get(f["index"], 0.0), f["act"])
        top = sorted(best.items(), key=lambda kv: -kv[1])[:cap]
        return [
            {"index": i, "act": round(v, 3), "source": config.NP_SOURCE} for i, v in top
        ]
```

In `NeuronpediaProvider.features_for`, add `special_ids=None,` to the signature (after `token_ids=None,`); it is ignored (text-based path).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest backend/tests/test_provider_masking.py`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/science/feature_provider.py backend/tests/test_provider_masking.py
git commit -m "fix: LocalSAEProvider masks special tokens via explicit special_ids set"
```

---

### Task 5: Synthetic fallback + the shared analyze_turn core

**Files:**
- Create: `backend/fallback.py`
- Create: `backend/analyze.py`
- Create: `backend/tests/test_fallback.py`
- Create: `backend/tests/test_analyze.py`

**Interfaces:**
- Consumes: `runtime.STATE`, `events.build_cognition_event`, `labels.get_label`, `config`; lazily `engine`, `feature_provider`, `persona`.
- Produces:
  - `fallback.synth_turn(messages: list[dict]) -> tuple[str, list[dict]]` — deterministic `(answer, features)`; each feature is a full dict `{index,label,act,source,caveat,tracked}`.
  - `analyze.analyze_turn(messages: list[dict], *, message_id=None, ts=None) -> tuple[str, CognitionEvent]` — branches on `runtime.STATE["mode"]`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_fallback.py`:

```python
from backend import fallback


def test_synth_turn_is_deterministic():
    msgs = [{"role": "user", "content": "Is ibuprofen safe in the third trimester?"}]
    a1, f1 = fallback.synth_turn(msgs)
    a2, f2 = fallback.synth_turn(msgs)
    assert a1 == a2
    assert f1 == f2
    assert isinstance(a1, str) and a1.strip()
    assert len(f1) >= 5
    assert all(f["label"] for f in f1)
    assert all(f["source"] == "17-gemmascope-2-res-16k" for f in f1)


def test_synth_turn_varies_by_prompt():
    _, fa = fallback.synth_turn([{"role": "user", "content": "AAA"}])
    _, fb = fallback.synth_turn([{"role": "user", "content": "BBB"}])
    assert fa != fb
```

`backend/tests/test_analyze.py`:

```python
from backend import analyze, runtime
from backend.schema import CognitionEvent


def test_analyze_turn_fallback_builds_valid_event():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
    answer, event = analyze.analyze_turn(
        [{"role": "user", "content": "Is metformin safe during pregnancy?"}]
    )
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(event, CognitionEvent)
    assert event.uncertainty is None
    assert event.flag is False
    assert event.severity == "info"
    assert event.io.user_msg == "Is metformin safe during pregnancy?"
    assert event.io.response == answer
    assert len(event.features) > 0
    assert event.features[0].label
    # round-trips through a fresh validation
    CognitionEvent(**event.model_dump())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest backend/tests/test_fallback.py backend/tests/test_analyze.py`
Expected: FAIL — `No module named 'backend.fallback'` / `'backend.analyze'`.

- [ ] **Step 3: Create `backend/fallback.py`**

```python
"""Deterministic synthetic turn for when the real model/SAE is absent (no torch, no weights,
or still loading). Offline-safe: features carry their own curated labels — no network.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import hashlib
import random

from . import config
from .mock_labels import CLINICAL_LABELS

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"

_TEMPLATES = [
    "That's an important clinical question about {topic}. In general it depends on the "
    "specific dose, timing, and the patient's history — the safest course is to confirm "
    "against current guidelines or with a clinician before acting.",
    "Good question regarding {topic}. There are real trade-offs here, and the right answer "
    "varies by individual circumstances; I'd verify the latest evidence before relying on this.",
    "Regarding {topic}: the considerations are nuanced. Standard practice offers guidance, "
    "but individual risk factors matter, so corroborate with an authoritative source.",
]


def _seed(messages: list[dict]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _topic(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            words = m["content"].strip().rstrip("?.!").split()
            return " ".join(words[:8]) if words else "this question"
    return "this question"


def synth_turn(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return a deterministic (answer, features) for the given conversation."""
    seed = _seed(messages)
    rng = random.Random(seed)
    answer = rng.choice(_TEMPLATES).format(topic=_topic(messages))

    n = rng.randint(8, 12)
    labels = rng.sample(CLINICAL_LABELS, k=min(n, len(CLINICAL_LABELS)))
    features: list[dict] = []
    for i, label in enumerate(labels):
        idx = (seed >> (i * 3)) % config.D_SAE
        act = round(6.4 - i * 0.42 + rng.random() * 0.25, 3)
        features.append(
            {
                "index": idx,
                "label": label,
                "act": act,
                "source": config.NP_SOURCE,
                "caveat": DEFAULT_CAVEAT,
                "tracked": None,
            }
        )
    return answer, features
```

- [ ] **Step 4: Create `backend/mock_labels.py`**

```python
"""Curated feature labels for the offline synthetic fallback. Plausible clinical/linguistic
concepts so the feature map reads believably without any network call."""

CLINICAL_LABELS = [
    "pregnancy & gestation terms",
    "medication / drug safety",
    "anti-inflammatory (NSAID)",
    "dosage & administration",
    "hedging / expressions of caution",
    "consulting a physician",
    "contraindication & risk",
    "trimester & fetal terms",
    "reassurance · 'generally safe'",
    "clinical guidelines reference",
    "second-person address",
    "affirmation / yes",
    "temporal periods",
    "symptom description",
    "renal / hepatic clearance",
    "drug interaction",
]
```

- [ ] **Step 5: Create `backend/analyze.py`**

```python
"""Shared turn analysis: generate (or synthesize) a response and assemble its CognitionEvent.
Branches on runtime.STATE; used by both /api/chat (streaming) and /api/analyze.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint) — the real path
imports engine/science lazily inside _real_turn.
"""

from __future__ import annotations

import time
import uuid

from . import config, labels, runtime
from .events import build_cognition_event
from .schema import CognitionEvent

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _label_features(features: list[dict]) -> list[dict]:
    """Attach Neuronpedia labels to bare {index,act,source} features (real path).
    Features that already carry a label (fallback path) are left untouched."""
    for f in features:
        if not f.get("label"):
            f["label"] = labels.get_label(f["index"])
        f.setdefault("caveat", DEFAULT_CAVEAT)
        f.setdefault("tracked", None)
    return features


def _real_turn(messages: list[dict]) -> tuple[str, list[dict], dict]:
    from . import engine
    from .science.feature_provider import get_provider
    from .science.persona import score_all_trackers

    res = engine.generate_and_capture(messages)
    tok = res["tok"]
    special = {tok.convert_tokens_to_ids(t) for t in config.MASK_TOKENS}
    acts, rs = res["acts"], res["resp_start"]
    resp_acts = acts[rs:]
    resp_ids = res["out_ids"][rs:].tolist()

    feats = get_provider().features_for(
        res["answer"], activations=resp_acts, token_ids=resp_ids, special_ids=special
    )
    _label_features(feats)

    act_last = acts[rs - 1] if rs > 0 else None
    act_resp = resp_acts.float().mean(0) if resp_acts.shape[0] > 0 else None
    trackers = score_all_trackers(act_last, act_resp)  # {} until probes exist
    return res["answer"], feats, trackers


def analyze_turn(
    messages: list[dict], *, message_id: str | None = None, ts: float | None = None
) -> tuple[str, CognitionEvent]:
    message_id = message_id or uuid.uuid4().hex
    ts = time.time() if ts is None else ts

    if runtime.STATE["mode"] == "real":
        answer, feats, trackers = _real_turn(messages)
    else:
        from . import fallback

        answer, feats = fallback.synth_turn(messages)
        trackers = {}

    event = build_cognition_event(
        message_id=message_id,
        ts=ts,
        user_msg=_last_user(messages),
        response=answer,
        trackers=trackers,
        features=_label_features(feats),
        model=config.MODEL_ID,
        layer=config.LAYER,
    )
    return answer, event
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest backend/tests/test_fallback.py backend/tests/test_analyze.py`
Expected: 3 passed. (No network: fallback features already have labels, so `_label_features` makes no calls.)

- [ ] **Step 7: Commit**

```bash
git add backend/fallback.py backend/mock_labels.py backend/analyze.py backend/tests/test_fallback.py backend/tests/test_analyze.py
git commit -m "feat: synthetic fallback + shared analyze_turn core"
```

---

### Task 6: Wire FastAPI endpoints + NDJSON streaming

**Files:**
- Modify: `backend/app.py`
- Create: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `analyze.analyze_turn`, `runtime`, `fanout`, `labels`, `config`.
- Produces: `POST /api/chat` (NDJSON stream), `POST /api/analyze` (JSON event), `GET /api/health`, `GET /api/feature/{index}`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_api.py`:

```python
import json

from fastapi.testclient import TestClient

from backend import runtime
from backend.app import app
from backend.schema import CognitionEvent

client = TestClient(app)


def _force_fallback():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    p = r.json()
    assert {"mode", "model", "layer", "trackers"} <= set(p)


def test_chat_streams_tokens_then_one_event():
    _force_fallback()
    r = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Is ibuprofen safe in the third trimester?"}]},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    parsed = [json.loads(ln) for ln in lines]
    assert len(parsed) >= 2
    assert all(p["type"] == "token" for p in parsed[:-1])
    assert parsed[-1]["type"] == "event"
    ev = parsed[-1]
    CognitionEvent(**ev)  # schema-valid
    assert ev["uncertainty"] is None
    assert ev["flag"] is False
    assert len(ev["features"]) > 0
    assert ev["features"][0]["label"]


def test_analyze_returns_event():
    _force_fallback()
    r = client.post("/api/analyze", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    CognitionEvent(**r.json())


def test_feature_label(monkeypatch):
    from backend import labels

    monkeypatch.setattr(labels, "get_label", lambda i, **k: f"feat-{i}")
    r = client.get("/api/feature/123")
    assert r.json()["label"] == "feat-123"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest backend/tests/test_api.py`
Expected: FAIL — `/api/chat` still replays the fixture (no real token/event split from `analyze_turn`); `/api/health` lacks `model`.

- [ ] **Step 3: Rewrite `backend/app.py`**

```python
"""FastAPI backend — the A↔C seam. OWNER: Lane A."""

from __future__ import annotations

import asyncio
import json
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import config, labels, runtime
from .analyze import analyze_turn
from .fanout import fanout, init_sponsors

app = FastAPI(title="GlassBox")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_TOKEN_CADENCE_S = 0.012  # replay the (already-generated) answer at a readable typing pace


@app.on_event("startup")
def _startup() -> None:
    runtime.start_loading()
    init_sponsors()


@app.get("/api/health")
def health() -> dict:
    return runtime.health_payload()


def _chunks(text: str) -> list[str]:
    """Split into word-with-trailing-space chunks for the streamed typing effect."""
    return re.findall(r"\S+\s*", text) or [text]


@app.post("/api/chat")
async def chat(body: dict):
    """Generate (or synthesize) a turn, stream token lines, then exactly one event line.
    fanout() runs AFTER the event line so nothing blocks the stream."""
    messages = body.get("messages") or []
    answer, event = await run_in_threadpool(analyze_turn, messages)
    payload = event.model_dump()

    async def gen():
        for chunk in _chunks(answer):
            yield json.dumps(
                {"type": "token", "text": chunk, "top_features": [], "uncertainty": None}
            ) + "\n"
            await asyncio.sleep(_TOKEN_CADENCE_S)
        yield json.dumps(payload) + "\n"
        fanout(payload)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze(body: dict):
    """Post-hoc / non-streaming variant: returns the CognitionEvent as JSON."""
    messages = body.get("messages") or []
    _, event = await run_in_threadpool(analyze_turn, messages)
    return JSONResponse(event.model_dump())


@app.post("/api/track")
async def track(body: dict):
    """User-defined concept (WIP — synth_concept unimplemented). Returns a stub status."""
    return {"tracker_id": body.get("concept", "concept"), "status": "computing"}


@app.get("/api/track/{tracker_id}")
async def track_status(tracker_id: str):
    return {"status": "ready", "auroc": None}


@app.get("/api/feature/{index}")
async def feature(index: int):
    """Server-side Neuronpedia label proxy/cache (dodges client CORS + rate limits)."""
    return {
        "index": index,
        "label": labels.get_label(index),
        "source": config.NP_SOURCE,
        "caveat": "auto-interp label, may be unreliable",
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest backend/tests/test_api.py`
Expected: 4 passed.

- [ ] **Step 5: Run the whole backend suite**

Run: `uv run pytest`
Expected: all tests pass (smoke, events, runtime, provider, fallback, analyze, api).

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/tests/test_api.py
git commit -m "feat: wire /api/chat NDJSON streaming + health/analyze/feature to real pipeline"
```

---

### Task 7: Frontend multi-turn chat

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/ChatPanel.tsx`

**Interfaces:**
- Consumes: `useCognitionStream()` (no `mock` → real `/api/chat`), `CognitionEvent` type.
- Produces: `ChatPanel` props `{ thread: Msg[]; pending: string | null; status; modelName: string; onSend(content: string): void }`; `Msg = { role: "user" | "assistant"; content: string }` exported from `ChatPanel`.

- [ ] **Step 1: Rewrite `frontend/src/components/ChatPanel.tsx`**

```tsx
// Clinician chat: a multi-turn thread + the streaming assistant turn + the composer.
import { useEffect, useRef, useState } from "react";

export type Msg = { role: "user" | "assistant"; content: string };

export function ChatPanel({
  thread, pending, status, modelName, onSend,
}: {
  thread: Msg[];
  pending: string | null;            // the in-progress assistant text while streaming
  status: "idle" | "streaming" | "done" | "error";
  modelName: string;
  onSend: (content: string) => void;
}) {
  const [draft, setDraft] = useState("Is ibuprofen safe to take in the third trimester of pregnancy?");
  const threadRef = useRef<HTMLDivElement>(null);
  const streaming = status === "streaming";

  useEffect(() => {
    const t = threadRef.current;
    if (t) t.scrollTop = t.scrollHeight;
  }, [thread, pending]);

  function submit() {
    const content = draft.trim();
    if (!content || streaming) return;
    onSend(content);
    setDraft("");
  }

  const who = (role: Msg["role"]) => (role === "user" ? "clinician" : modelName);

  return (
    <section className="chat glass">
      <div className="thread" ref={threadRef}>
        {thread.length === 0 && !pending && (
          <div className="empty">Ask a clinical question to watch the model's features fire.</div>
        )}
        {thread.map((m, i) => (
          <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}`}>
            <span className="who">{who(m.role)}</span>
            <div className="body">{m.content}</div>
          </div>
        ))}
        {streaming && (
          <div className="msg bot">
            <span className="who">{modelName}</span>
            <div className="body">{pending}<span className="cursor" /></div>
          </div>
        )}
      </div>
      <div className="composer">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        <button className={`send ${streaming ? "running" : ""}`} disabled={streaming} onClick={submit}>
          {streaming ? "Running…" : "Send ▸"}
        </button>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Rewrite `frontend/src/App.tsx`**

```tsx
// Root view. Left: multi-turn clinician chat. Right: cognition stage (feature field + probes +
// Claude verdict) reflecting the LATEST message's CognitionEvent. Real /api/chat (no mock).
import { useEffect, useMemo, useState } from "react";

import "./styles.css";
import type { CognitionEvent } from "./types";
import { useCognitionStream } from "./useCognitionStream";
import { ChatPanel, type Msg } from "./components/ChatPanel";
import { FeatureField } from "./components/FeatureField";
import { ProbePanel } from "./components/ProbePanel";
import { AdjudicationBanner } from "./components/AdjudicationBanner";

export function App() {
  const { answer, event, status, send } = useCognitionStream();
  const [thread, setThread] = useState<Msg[]>([]);
  const [latest, setLatest] = useState<CognitionEvent | null>(null);

  function onSend(content: string) {
    const history: Msg[] = [...thread, { role: "user", content }];
    setThread(history);
    send(history);
  }

  // Commit the assistant turn + capture its event when a stream finishes.
  // The "last msg is user" guard makes this idempotent across re-renders.
  useEffect(() => {
    if (status === "done") {
      setThread((t) => (t.length && t[t.length - 1].role === "user"
        ? [...t, { role: "assistant", content: answer || "" }] : t));
      if (event) setLatest(event);
    } else if (status === "error") {
      setThread((t) => (t.length && t[t.length - 1].role === "user"
        ? [...t, { role: "assistant", content: "[generation failed]" }] : t));
    }
  }, [status]);

  const features = useMemo(() => latest?.features ?? [], [latest]);
  const trackers = useMemo(() => latest?.trackers ?? {}, [latest]);
  const modelName = latest?.model ?? "model";

  return (
    <div className="app">
      <header className="glass">
        <div className="mark"><span className="lens" /><span className="g">glass</span><b>box</b></div>
        <div className="meta"><span className="pill">{modelName} · L{latest?.layer ?? "—"}</span></div>
        <div className="live"><span className="d" />observing</div>
      </header>

      <main>
        <ChatPanel
          thread={thread}
          pending={status === "streaming" ? answer : null}
          status={status}
          modelName={modelName}
          onSend={onSend}
        />
        <section className="stage">
          <FeatureField features={features} />
          <ProbePanel trackers={trackers} />
          <AdjudicationBanner adjudication={latest?.adjudication ?? null} />
          <p className="ethos">
            <b>Surface, never suppress</b> — we flag when to double-check, never alter the answer.
          </p>
        </section>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Add an `.empty` style to `frontend/src/styles.css`**

Append:

```css
.empty{margin:auto;color:var(--muted);font-size:13px;text-align:center;max-width:280px;line-height:1.55}
```

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: `tsc` clean, `vite build` succeeds. (Header model/layer is a placeholder here; Task 8 makes it health-driven.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/ChatPanel.tsx frontend/src/styles.css
git commit -m "feat: multi-turn chat thread on real /api/chat stream"
```

---

### Task 8: Health-driven header + backend-mode badge

**Files:**
- Create: `frontend/src/health.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

**Interfaces:**
- Produces: `useHealth(): Health | null` where `Health = { mode: "loading"|"real"|"fallback"; model: string; layer: number; model_loaded: boolean; sae_loaded: boolean; trackers: string[] }`.

- [ ] **Step 1: Create `frontend/src/health.ts`**

```ts
import { useEffect, useState } from "react";

export interface Health {
  mode: "loading" | "real" | "fallback";
  model: string;
  layer: number;
  model_loaded: boolean;
  sae_loaded: boolean;
  trackers: string[];
}

// Polls /api/health; keeps polling while the backend is still warming up the model.
export function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let alive = true;
    let tries = 0;
    const poll = async () => {
      try {
        const r = await fetch("/api/health");
        const j = (await r.json()) as Health;
        if (!alive) return;
        setHealth(j);
        if (j.mode === "loading" && tries++ < 40) setTimeout(poll, 1500);
      } catch {
        if (alive && tries++ < 40) setTimeout(poll, 2000);
      }
    };
    poll();
    return () => { alive = false; };
  }, []);
  return health;
}

export const MODE_BADGE: Record<Health["mode"], string> = {
  loading: "warming up",
  real: "live model",
  fallback: "synthetic · offline",
};
```

- [ ] **Step 2: Use health in the `App` header**

In `frontend/src/App.tsx`, add the import:

```tsx
import { useHealth, MODE_BADGE } from "./health";
```

Inside `App`, add `const health = useHealth();` near the other hooks. **Replace** the existing `const modelName = latest?.model ?? "model";` line from Task 7 with the two lines below (do not add a second `modelName` declaration), so the header prefers health, then the latest event, then a placeholder:

```tsx
  const modelName = latest?.model ?? health?.model ?? "model";
  const layerLabel = latest?.layer ?? health?.layer ?? "—";
```

Replace the header `<div className="meta">…</div>` and `<div className="live">…</div>` with:

```tsx
        <div className="meta">
          <span className="pill">{modelName} · L{layerLabel}</span>
          {health && <span className={`badge ${health.mode}`}>{MODE_BADGE[health.mode]}</span>}
        </div>
        <div className="live"><span className="d" />observing</div>
```

(Keep `modelName` used by `ChatPanel` as-is; it now resolves via health before the first event.)

- [ ] **Step 3: Add badge styles to `frontend/src/styles.css`**

Append:

```css
.badge{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;padding:3px 9px;border-radius:999px;border:1px solid var(--brd2)}
.badge.real{color:var(--teal);background:rgba(91,216,199,.1);border-color:rgba(91,216,199,.25)}
.badge.fallback{color:var(--amber);background:rgba(240,180,84,.1);border-color:rgba(240,180,84,.25)}
.badge.loading{color:var(--muted);background:rgba(255,255,255,.04)}
```

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/health.ts frontend/src/App.tsx frontend/src/styles.css
git commit -m "feat: health-driven header + backend-mode badge"
```

---

### Task 9: ProbePanel WIP placeholder + marked-preview define box

**Files:**
- Modify: `frontend/src/components/ProbePanel.tsx`
- Modify: `frontend/src/styles.css`

**Interfaces:**
- Consumes: `trackers: Record<string, Tracker>` (empty until probes exist).
- Produces: when `trackers` is empty, a WIP placeholder instead of an empty list; the define box is always shown but tagged "preview".

- [ ] **Step 1: Edit `frontend/src/components/ProbePanel.tsx`**

Compute emptiness and render a placeholder. Add at the top of the returned panel body (after the `ph` header, before the rows map):

```tsx
  const noBuiltins = builtins.length === 0;
```

Replace the rows line:

```tsx
      {[...builtins, ...custom].map((row, i) => <ProbeRow key={row.id ?? `b${i}`} row={row} />)}
```

with:

```tsx
      {noBuiltins && custom.length === 0 && (
        <div className="wip">
          <b>Family B — calibrated uncertainty / safety probes</b>
          <span>In progress. Once probes are trained, the live meter (green → red) appears here per message.</span>
        </div>
      )}
      {[...builtins, ...custom].map((row, i) => <ProbeRow key={row.id ?? `b${i}`} row={row} />)}
```

In the `.define` block, tag it as preview — change the wrapper and add a label:

```tsx
      <div className="define preview">
        <span className="tag">preview</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") define(); }}
          placeholder="Define a probe in natural language…  e.g. over-confidence"
        />
        <button onClick={define}>+ track</button>
      </div>
```

- [ ] **Step 2: Add styles to `frontend/src/styles.css`**

Append:

```css
.wip{display:flex;flex-direction:column;gap:5px;padding:14px 14px;border:1px dashed var(--brd);border-radius:11px;background:rgba(255,255,255,.02)}
.wip b{font-size:13px;color:var(--soft);font-weight:600}
.wip span{font-size:12px;color:var(--muted);line-height:1.5}
.define.preview{position:relative}
.define .tag{position:absolute;top:-9px;left:10px;font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--amber);background:var(--bg2);padding:0 6px}
```

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ProbePanel.tsx frontend/src/styles.css
git commit -m "feat: ProbePanel WIP placeholder + marked-preview define box"
```

---

### Task 10: End-to-end verification + docs

**Files:**
- Modify: `README.md` (Quickstart only)

**Interfaces:** none (verification task).

- [ ] **Step 1: Run the full backend suite once more**

Run: `uv run pytest`
Expected: all green.

- [ ] **Step 2: Start the backend (fallback mode on this Mac)**

Run: `uv run uvicorn backend.app:app --port 8000`
In another shell: `curl -s localhost:8000/api/health`
Expected JSON includes `"mode": "fallback"` (no torch here), `"model": "unsloth/gemma-3-4b-it"`, `"layer": 17`, `"trackers": []`.

- [ ] **Step 3: Exercise the stream from the CLI**

Run: `curl -sN -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Is ibuprofen safe in the third trimester?"}]}'`
Expected: several `{"type":"token",...}` lines then one `{"type":"event",...}` line with `"uncertainty": null`, `"flag": false`, and a non-empty `"features"` array with labels.

- [ ] **Step 4: Run the frontend and verify the experience manually**

Run: `cd frontend && npm install && npm run dev` (Vite on :5173, proxies `/api` → :8000).
In the browser, confirm:
- Header shows `unsloth/gemma-3-4b-it · L17` and a `synthetic · offline` badge.
- Sending a question streams the answer into the thread; a second question appends a new turn (multi-turn).
- The feature field illuminates the event's features after each answer; hovering a lit point shows its label + caveat.
- The probe panel shows the **Family B … in progress** placeholder and a **preview**-tagged define box.

- [ ] **Step 5: Update the README Quickstart to match reality**

In `README.md`, update the Quickstart commands and the model/layer references so they reflect: `uv sync` (light) / `uv sync --extra ml` (real path), `uv run uvicorn backend.app:app --port 8000`, model `unsloth/gemma-3-4b-it`, layer 17, and that the backend auto-falls-back to synthetic mode when the ML stack/weights are absent. Keep edits scoped to Quickstart + the model-decision line.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: update Quickstart for uv + gemma-3 + synthetic fallback"
```

---

## Self-Review

**Spec coverage:**
- §1 goal (multi-turn chat → SAE → labels → event → feature map): Tasks 5, 6, 7. ✓
- §3 per-request readiness (background load, modes): Task 3 + Task 6 startup. ✓
- §4 data flow + real path + config identity: Tasks 4, 5, 6. ✓
- §5 contract change (nullable Family B, three files lockstep): Task 2. ✓
- §6 frontend (mock:false, multi-turn, header/badge, feature field, ProbePanel WIP, remove auto-run): Tasks 7, 8, 9 (auto-run removed by App rewrite in Task 7). ✓
- §7 testing (pure units, TestClient NDJSON, health): Tasks 2, 3, 4, 5, 6; manual frontend in Task 10. ✓
- §8 deps (ml extra, lazy): Task 1 + Global Constraints. ✓
- §9 risks: addressed by fallback (Tasks 3/5/6) and label degradation (existing `get_label`). ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step shows full content. README step (Task 10 Step 5) is a scoped prose instruction, acceptable for a docs touch-up. ✓

**Type consistency:** `analyze_turn(messages, *, message_id, ts) -> (str, CognitionEvent)` used consistently (Tasks 5, 6). `features_for(..., special_ids=...)` defined in Task 4, consumed in Task 5. `build_cognition_event(*, ..., model, layer)` signature matches across Tasks 2 and 5. `Msg`/`ChatPanel` props defined in Task 7, header `modelName` consistent in Task 8. `useHealth`/`Health` defined and consumed in Task 8. ✓
