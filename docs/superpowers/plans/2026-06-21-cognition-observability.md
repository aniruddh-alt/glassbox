# Cognition Observability Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Sentry (alarms) + Arize Phoenix (ledger + coherence eval) end-to-end behind a hard no-PII boundary, surfaced on a new frontend Observability page, with the fan-out architected as one event → many sinks.

**Architecture:** A single redacted `CognitionEvent` fans out to interchangeable `Sink`s (Sentry, Phoenix, in-process store). The store feeds `GET /api/observability` (and a future MCP server). Phoenix gets a per-turn span waterfall + a post-hoc batch coherence eval over *feature labels only*. Prompt/response never leave the box.

**Tech Stack:** FastAPI, `sentry-sdk 2.63`, `arize-phoenix 17.9` + `arize-phoenix-evals 3.1.0` + `arize-phoenix-otel 0.16.1`, `openinference`, `opentelemetry`, `httpx 0.28`, React 18 + Vite + TypeScript (no new FE deps).

## Global Constraints

- **Lane A backend files must never import torch** (`app.py`, `analyze.py`, `fanout.py`, `observability.py`, `sentry_api.py`, `coherence_eval.py`, `labels.py`, `config.py`, `runtime.py`, `events.py`, `pod_client.py`). The torch path lives only on the pod (`gpu_service.py`, `engine.py`, `science/*`).
- **`schema.py` is frozen** (contract #1). No new `CognitionEvent` fields — `perf`/timings travel out-of-band.
- **NO PII leaves the machine.** Prompt (`io.user_msg`) and response (`io.response`) must never reach Sentry, Phoenix, the store, the REST response, or any cloud LLM. Only feature **labels** + probe **scores** + ids + timings may.
- **Backend tests run without torch installed.** Sinks/eval are CPU-only.
- **Redaction is allow-list, defined once** in `observability.to_redacted_view`.
- **Phoenix is local-only** for patient data; **Sentry fires only on flagged turns.**
- Spec: `docs/superpowers/specs/2026-06-21-cognition-observability-design.md`. Canonical stage names: `pod_roundtrip, capture, sae, trackers, label_fetch, ranking` (+ `turn_ms`).

---

## File Structure

**New (Lane A):** `backend/observability.py` (redaction + store + read API), `backend/sentry_api.py` (REST issues read), `backend/coherence_eval.py` (Phoenix batch eval).
**Rewrite:** `backend/fanout.py` (Sink protocol + registry + 3 sinks + hardened init + report_error).
**Edit (Lane A):** `backend/config.py`, `backend/app.py`, `backend/analyze.py`, `backend/pod_client.py`, `backend/labels.py`, `backend/science/feature_provider.py` (provider guard).
**Edit (Lane B ⚠):** `backend/gpu_service.py` (additive stage timings).
**New/Edit (Lane C):** `frontend/src/ObservabilityPage.tsx`, `frontend/src/useObservability.ts` (new); `frontend/src/App.tsx`, `api.ts`, `types.ts`, `styles.css`, `mock.ts` (edit).
**Tests:** `backend/tests/test_observability.py`, `test_fanout.py`, `test_sentry_api.py`, `test_coherence_eval.py`, extend `test_analyze.py`, `test_pod_client.py`.

---

# WORKSTREAM 1 — Privacy + sinks core (ships first; fixes the live leak)

### Task 1: `config.py` — observability env vars

**Files:** Modify `backend/config.py` (after `PHOENIX_ENDPOINT`, ~:118). Note `AUTOINTERP_MODEL` already exists at :74.

**Interfaces:**
- Produces: `config.PHOENIX_UI_URL, SENTRY_ORG_SLUG, SENTRY_PROJECT_SLUG, SENTRY_AUTH_TOKEN, SENTRY_API_BASE, SENTRY_ORG_URL, EVAL_LLM_PROVIDER, EVAL_LLM_MODEL` (all `str`).

- [ ] **Step 1: Add the vars**

```python
# --- Observability surfaces (read paths + eval) ---
PHOENIX_UI_URL      = os.getenv("PHOENIX_UI_URL", PHOENIX_ENDPOINT)        # iframe src
SENTRY_ORG_SLUG     = os.getenv("SENTRY_ORG_SLUG", "")
SENTRY_PROJECT_SLUG = os.getenv("SENTRY_PROJECT_SLUG", "")
SENTRY_AUTH_TOKEN   = os.getenv("SENTRY_AUTH_TOKEN", "")                    # internal-integration, event:read+project:read
SENTRY_API_BASE     = os.getenv("SENTRY_API_BASE", "https://sentry.io").rstrip("/")
SENTRY_ORG_URL      = os.getenv("SENTRY_ORG_URL", "https://sentry.io")     # deep-link host
EVAL_LLM_PROVIDER   = os.getenv("EVAL_LLM_PROVIDER", "anthropic")
EVAL_LLM_MODEL      = os.getenv("EVAL_LLM_MODEL", "claude-haiku-4-5-20251001")
```

- [ ] **Step 2: Verify import** — Run: `python -c "from backend import config; print(config.SENTRY_API_BASE, config.EVAL_LLM_PROVIDER)"` Expected: `https://sentry.io anthropic`
- [ ] **Step 3: Commit** — `git add backend/config.py && git commit -m "feat(obs): add observability config vars"`

---

### Task 2: `observability.to_redacted_view` — the allow-list projector

**Files:** Create `backend/observability.py`; Test `backend/tests/test_observability.py`.

**Interfaces:**
- Produces: `to_redacted_view(event: dict) -> dict` — rebuilds a de-identified view from named safe fields; never reads `event["io"]` or `adjudication.rationale`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_observability.py
from backend.observability import to_redacted_view

SECRET_Q, SECRET_A = "PATIENT_PROMPT_42yo_warfarin", "RESPONSE_take_5mg"

def _event(**over):
    e = {"message_id": "m1", "ts": 1.0, "model": "gemma", "layer": 17,
         "io": {"user_msg": SECRET_Q, "response": SECRET_A},
         "uncertainty": 0.2, "uncertainty_proj": 1.1, "uncertainty_proj_pre": 0.9,
         "flag": True, "severity": "warning",
         "trackers": {"hallucination": {"score": 0.8, "proj": 2.0, "proj_pre": 1.0,
                                        "flag": True, "reliable": True, "user_defined": False, "status": "ready"}},
         "features": [{"index": 12, "label": "anticoagulant dosing", "act": 1.8,
                       "source": "17-gemmascope-2-res-16k", "caveat": "x", "tracked": None}],
         "adjudication": {"verdict": "likely_hallucinated", "rationale": SECRET_A, "by": "claude"}}
    e.update(over); return e

def test_redacted_view_omits_prompt_response_and_rationale():
    v = to_redacted_view(_event())
    blob = repr(v)
    assert SECRET_Q not in blob and SECRET_A not in blob
    assert "io" not in v
    assert v["adjudication"] == {"verdict": "likely_hallucinated", "by": "claude"}  # no rationale
    assert v["features"][0]["label"] == "anticoagulant dosing"
    assert v["trackers"]["hallucination"]["score"] == 0.8

def test_redacted_view_handles_no_adjudication_and_empty_trackers():
    v = to_redacted_view(_event(adjudication=None, trackers={}, features=[]))
    assert v["adjudication"] is None and v["trackers"] == {} and v["features"] == []
```

- [ ] **Step 2: Run, expect fail** — Run: `pytest backend/tests/test_observability.py -v` Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# backend/observability.py
"""Canonical cognition-observability store + the SINGLE redaction projector.
Lane A — never imports torch. Prompt/response (io.*) are NEVER read here."""
from __future__ import annotations
from collections import deque

_TRACKER_KEYS = ("score", "proj", "proj_pre", "flag", "reliable", "user_defined", "status")

def to_redacted_view(event: dict) -> dict:
    """ALLOW-LIST projection → de-identified view. io.* and adjudication.rationale are never copied,
    so prompt/response cannot leak structurally."""
    a = event.get("adjudication")
    return {
        "message_id": event["message_id"], "ts": event["ts"],
        "model": event.get("model"), "layer": event.get("layer"),
        "uncertainty": event.get("uncertainty"),
        "uncertainty_proj": event.get("uncertainty_proj"),
        "uncertainty_proj_pre": event.get("uncertainty_proj_pre"),
        "flag": event.get("flag", False), "severity": event.get("severity", "info"),
        "trackers": {tid: {k: tr.get(k) for k in _TRACKER_KEYS}
                     for tid, tr in (event.get("trackers") or {}).items()},
        "features": [{"index": f["index"], "label": f["label"], "act": f.get("act"),
                      "source": f.get("source"), "tracked": f.get("tracked")}
                     for f in (event.get("features") or [])],
        "adjudication": ({"verdict": a["verdict"], "by": a.get("by", "claude")} if a else None),
    }
```

- [ ] **Step 4: Run, expect pass** — Run: `pytest backend/tests/test_observability.py -v` Expected: PASS.
- [ ] **Step 5: Commit** — `git add backend/observability.py backend/tests/test_observability.py && git commit -m "feat(obs): allow-list redaction projector"`

---

### Task 3: `ObservabilityStore` — record (guarded) + snapshot

**Files:** Modify `backend/observability.py`; `backend/tests/test_observability.py`.

**Interfaces:**
- Consumes: `to_redacted_view` (Task 2), `runtime.health_payload` (existing).
- Produces: `STORE: ObservabilityStore` with `record(view, perf)`, `snapshot() -> dict`, `get_recent_concerns(kind=None, limit=20)`, `get_turn(message_id)`, `query_turns(filter)`, `concern_taxonomy()`.

- [ ] **Step 1: Write the failing test**

```python
from backend.observability import ObservabilityStore, to_redacted_view

def test_store_rejects_raw_event_and_snapshot_aggregates():
    s = ObservabilityStore(maxlen=5)
    raw = {"io": {"user_msg": "x"}, "message_id": "m", "ts": 1.0}
    try:
        s.record(raw, None); assert False, "should reject un-redacted view"
    except AssertionError as e:
        assert "redacted" in str(e)
    # well-formed redacted views
    for i in range(3):
        v = to_redacted_view({"message_id": f"m{i}", "ts": float(i), "model": "g", "layer": 17,
                              "uncertainty": 0.1 * i, "flag": i == 2, "severity": "warning",
                              "trackers": {"uncertainty": {"score": 0.1 * i, "flag": i == 2}},
                              "features": [{"index": 1, "label": "dosing", "act": 1.0}]})
        s.record(v, {"turn_ms": 100 + i, "stages": {"pod_roundtrip": 50}})
    snap = s.snapshot()
    assert snap["totals"]["turns"] == 3
    assert len(snap["uncertainty_series"]) == 3
    assert abs(snap["flag_rate"] - 1/3) < 1e-9
    assert snap["trackers"]["uncertainty"]["series"] == [0.0, 0.1, 0.2]
    assert snap["top_features"][0]["label"] == "dosing" and snap["top_features"][0]["count"] == 3
    assert snap["latency"]["turn_ms"]["last"] == 102
    assert len(snap["confident_wrong"]) == 1 and snap["confident_wrong"][0]["message_id"] == "m2"

def test_store_empty_snapshot_is_well_formed():
    snap = ObservabilityStore().snapshot()
    assert snap["totals"]["turns"] == 0 and snap["trackers"] == {} and snap["confident_wrong"] == []
```

- [ ] **Step 2: Run, expect fail** — Run: `pytest backend/tests/test_observability.py -k store -v` Expected: FAIL.

- [ ] **Step 3: Implement** (append to `observability.py`)

```python
def _pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]

class ObservabilityStore:
    def __init__(self, maxlen: int = 200):
        self._buf: deque[dict] = deque(maxlen=maxlen)

    def record(self, view: dict, perf: dict | None) -> None:
        assert "io" not in view, "store must hold only redacted views (io present)"
        self._buf.append({"ts": view["ts"], "view": view, "perf": perf or {}})

    def snapshot(self) -> dict:
        items = list(self._buf)
        views = [it["view"] for it in items]
        unc = [v["uncertainty"] for v in views if v.get("uncertainty") is not None]
        flags = [v for v in views if v.get("flag")]
        # per-tracker series (bounded by deque; absent → skipped)
        tracker_ids = {tid for v in views for tid in (v.get("trackers") or {})}
        trackers = {}
        for tid in tracker_ids:
            series = [v["trackers"][tid]["score"] for v in views if tid in (v.get("trackers") or {})]
            trackers[tid] = {"current": series[-1] if series else None,
                             "flag_count": sum(1 for v in views if (v.get("trackers") or {}).get(tid, {}).get("flag")),
                             "series": series}
        # feature leaderboard
        feat = {}
        for v in views:
            for f in v.get("features") or []:
                d = feat.setdefault(f["label"], {"label": f["label"], "count": 0, "_acts": []})
                d["count"] += 1
                if f.get("act") is not None: d["_acts"].append(f["act"])
        top = sorted(feat.values(), key=lambda d: -d["count"])
        for d in top:
            d["mean_act"] = round(sum(d["_acts"]) / len(d["_acts"]), 4) if d["_acts"] else None
            d.pop("_acts")
        # latency percentiles per stage
        def stage_vals(key):
            out = {}
            for it in items:
                for name, ms in (it["perf"].get("stages", {}) | it["perf"].get("pod_stages", {})).items():
                    out.setdefault(name, []).append(ms)
            return out
        turn_ms = [it["perf"].get("turn_ms") for it in items if it["perf"].get("turn_ms") is not None]
        stages = stage_vals("stages")
        latency = {"turn_ms": {"p50": _pct(turn_ms, 50), "p95": _pct(turn_ms, 95),
                               "last": turn_ms[-1] if turn_ms else None},
                   "stages": {n: {"p50": _pct(vs, 50)} for n, vs in stages.items()}}
        cw = [{"message_id": v["message_id"], "ts": v["ts"], "uncertainty": v.get("uncertainty"),
               "trackers": v.get("trackers", {}),
               "feature_labels": [f["label"] for f in v.get("features") or []]} for v in flags]
        return {"ts": views[-1]["ts"] if views else None,
                "totals": {"turns": len(views), "flags": len(flags)},
                "flag_rate": (len(flags) / len(views)) if views else 0.0,
                "uncertainty_series": unc, "trackers": trackers,
                "top_features": top, "latency": latency, "confident_wrong": cw}

    def get_recent_concerns(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        cw = self.snapshot()["confident_wrong"][-limit:]
        return cw if kind in (None, "confident_wrong") else []

    def get_turn(self, message_id: str) -> dict | None:
        return next((it["view"] for it in reversed(self._buf) if it["view"]["message_id"] == message_id), None)

    def query_turns(self, filter: dict) -> dict:
        out = [it["view"] for it in self._buf]
        if filter.get("flagged"): out = [v for v in out if v.get("flag")]
        if filter.get("min_uncertainty") is not None:
            out = [v for v in out if (v.get("uncertainty") or 0) >= filter["min_uncertainty"]]
        lim = filter.get("limit", 50)
        return {"turns": out[-lim:], "next_cursor": None}

    def concern_taxonomy(self) -> list[dict]:
        return [{"id": "confident_wrong", "description": "low-uncertainty flagged answer", "source": "tracker"},
                {"id": "instrument_unhealthy", "description": "pod/SAE/label failure", "source": "system"},
                {"id": "feature_incoherence", "description": "off-domain feature activation", "source": "phoenix-eval"}]

STORE = ObservabilityStore()
```

- [ ] **Step 4: Run, expect pass** — Run: `pytest backend/tests/test_observability.py -v` Expected: PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs): in-process store + snapshot aggregation"`

---

### Task 4: REWRITE `fanout.py` — Sink protocol + registry + isolation

**Files:** Modify `backend/fanout.py`; Test `backend/tests/test_fanout.py` (file exists — extend/replace).

**Interfaces:**
- Produces: `Sink` (Protocol, `name: str`, `emit(event, perf=None)`), `register_sink(s)`, `_SINKS: list`, `fanout(event: dict, perf: dict | None = None)`, `report_error(stage, exc, ctx=None)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_fanout.py (add)
import backend.fanout as fo

def test_fanout_isolates_failing_sink(monkeypatch):
    calls = []
    class Good: name = "good";  emit = lambda self, e, p=None: calls.append("good")
    class Bad:  name = "bad";   emit = lambda self, e, p=None: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(fo, "_SINKS", [Bad(), Good()])
    fo.fanout({"flag": False}, {"turn_ms": 1})   # must not raise; Good still runs
    assert calls == ["good"]
```

- [ ] **Step 2: Run, expect fail** — Run: `pytest backend/tests/test_fanout.py -k isolates -v` Expected: FAIL.

- [ ] **Step 3: Implement** — replace the top of `fanout.py` (keep the module docstring; remove the old `fanout`/`_to_sentry`/`_to_phoenix` bodies — they are re-added as sinks in later tasks):

```python
from __future__ import annotations
import asyncio, json
from typing import Protocol, runtime_checkable
from . import config

_sentry_on = False
_tracer = None

@runtime_checkable
class Sink(Protocol):
    name: str
    def emit(self, event: dict, perf: dict | None = None) -> None: ...

_SINKS: list[Sink] = []
def register_sink(s: Sink) -> None: _SINKS.append(s)

def fanout(event: dict, perf: dict | None = None) -> None:
    """Fan one finished cognition_event (+ optional perf) to every sink. CPU only, post-stream."""
    for s in _SINKS:
        try:
            s.emit(event, perf)
        except Exception as e:  # noqa: BLE001 — one bad sink never breaks others / the stream
            print(f"[fanout] sink {getattr(s, 'name', '?')} failed: {e}")

def report_error(stage: str, exc: BaseException, ctx: dict | None = None) -> None:
    """System-error → Sentry capture_exception, sanitized (relies on init's locals-off + sanitized excs)."""
    if not _sentry_on:
        return
    import sentry_sdk
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["glassbox", "system", stage]
        scope.set_tag("subsystem", stage)
        scope.set_level("error")
        if ctx:
            scope.set_context("system", ctx)
        sentry_sdk.capture_exception(exc)
```

- [ ] **Step 4: Run, expect pass** — Run: `pytest backend/tests/test_fanout.py -k isolates -v` Expected: PASS.
- [ ] **Step 5: Commit** — `git add backend/fanout.py backend/tests/test_fanout.py && git commit -m "refactor(fanout): Sink protocol + registry + report_error"`

---

### Task 5: Hardened Sentry init + `_scrub_pii` + `_level`

**Files:** Modify `backend/fanout.py`; `backend/tests/test_fanout.py`.

**Interfaces:**
- Produces: `init_sponsors()` (Sentry init + Phoenix register + sink registration — built incrementally across Tasks 5-8), `_scrub_pii(event, hint)`, `_level(severity) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_scrub_pii_strips_exception_frame_vars():
    from backend.fanout import _scrub_pii
    ev = {"exception": {"values": [{"stacktrace": {"frames": [
            {"vars": {"messages": "PATIENT_PROMPT", "x": 1}}]}}]},
          "contexts": {}, "extra": {"answer": "RESPONSE_TEXT"}}
    out = _scrub_pii(ev, {})
    assert out["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"] == {}
    assert "RESPONSE_TEXT" not in repr(out)

def test_level_clamps():
    from backend.fanout import _level
    assert _level("warning") == "warning" and _level("info") == "info"
    assert _level("bogus") == "warning"
```

- [ ] **Step 2: Run, expect fail** — Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** (add to `fanout.py`)

```python
_VALID_LEVELS = {"fatal", "critical", "error", "warning", "info", "debug"}
_PII_KEYS = {"question", "answer", "user_msg", "response", "prompt", "messages", "io"}

def _level(severity: str) -> str:
    return severity if severity in _VALID_LEVELS else "warning"

def _scrub_pii(event: dict, hint: dict):
    """before_send: drop ALL exception-frame locals + any forbidden-key values anywhere. Defense-in-depth."""
    for v in event.get("exception", {}).get("values", []):
        for fr in v.get("stacktrace", {}).get("frames", []):
            fr["vars"] = {}
    def walk(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in _PII_KEYS: o[k] = "[scrubbed]"
                else: walk(o[k])
        elif isinstance(o, list):
            for x in o: walk(x)
    for sect in ("contexts", "extra", "request"):
        if sect in event: walk(event[sect])
    return event

def init_sponsors() -> None:
    """Call once at startup. Each sink is independently optional."""
    global _sentry_on, _tracer
    if config.SENTRY_DSN:
        import sentry_sdk
        sentry_sdk.init(
            dsn=config.SENTRY_DSN, traces_sample_rate=0.0, environment="hackathon",
            send_default_pii=False,
            include_local_variables=False,   # CRITICAL: frame locals carry the prompt + io.*
            include_source_context=False,
            max_request_body_size="never",
            before_send=_scrub_pii,
        )
        _sentry_on = True
    # Phoenix register + sink registration added in Tasks 6-8.
```

- [ ] **Step 4: Run, expect pass** — Run: `pytest backend/tests/test_fanout.py -k "scrub or level" -v` Expected: PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(fanout): hardened Sentry init + PII scrub + level clamp"`

---

### Task 6: `SentrySink` — quiet alarm, redacted context

**Files:** Modify `backend/fanout.py`; `backend/tests/test_fanout.py`.

**Interfaces:**
- Consumes: `_level`, `_sentry_on`.
- Produces: `SentrySink` (`name="sentry"`, `emit`).

- [ ] **Step 1: Write the failing test**

```python
def test_sentry_sink_quiet_on_unflagged_and_redacted_on_flag(monkeypatch):
    import backend.fanout as fo
    captured = {}
    fake = type("S", (), {})()
    fake.capture_message = lambda msg, level=None: captured.setdefault("msgs", []).append((msg, level))
    class Scope:
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def set_tag(s, *a): pass
        def set_context(s, k, v): captured["ctx"] = v
        fingerprint = None
    fake.new_scope = lambda: Scope()
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", fake)
    monkeypatch.setattr(fo, "_sentry_on", True)
    sink = fo.SentrySink()
    ev = {"flag": False}; sink.emit(ev); assert "msgs" not in captured        # quiet
    ev = {"flag": True, "severity": "warning", "model": "g", "uncertainty": 0.2,
          "uncertainty_proj": 1.0, "trackers": {}, "io": {"user_msg": "SECRET", "response": "SECRET"},
          "features": [{"label": "dosing"}]}
    sink.emit(ev)
    assert captured["msgs"][0][1] == "warning"
    assert "SECRET" not in repr(captured["ctx"]) and captured["ctx"]["top_features"] == ["dosing"]
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** (add to `fanout.py`)

```python
def _bucket(u): return "high" if (u or 0) >= 0.66 else "med" if (u or 0) >= 0.33 else "low"

class SentrySink:
    name = "sentry"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        if not _sentry_on or not event.get("flag"):   # QUIET: only flagged turns alarm
            return
        import sentry_sdk
        with sentry_sdk.new_scope() as scope:
            scope.fingerprint = ["glassbox", "medical-cognition", "confident_wrong"]
            scope.set_tag("model", event.get("model"))
            scope.set_tag("event_type", "confident_wrong")
            scope.set_tag("uncertainty_bucket", _bucket(event.get("uncertainty")))
            scope.set_context("cognition", {        # PII-FREE — no question/answer
                "uncertainty": event.get("uncertainty"),
                "uncertainty_proj": event.get("uncertainty_proj"),
                "trackers": event.get("trackers", {}),
                "top_features": [f["label"] for f in event.get("features", [])],
            })
            sentry_sdk.capture_message("Confident-wrong medical answer", level=_level(event.get("severity", "warning")))
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(fanout): SentrySink (quiet, redacted alarm)"`

---

### Task 7: `PhoenixSink` — redacted span waterfall (epoch-ns)

**Files:** Modify `backend/fanout.py`; `backend/tests/test_fanout.py`.

**Interfaces:**
- Consumes: `perf["t0_ns"]`, `perf["turn_ms"]`, canonical stages.
- Produces: `PhoenixSink` (`name="phoenix"`, `emit`), `_stage_spans(perf) -> list[(name, kind, ms)]`. Sets `_tracer` in `init_sponsors`.

- [ ] **Step 1: Write the failing test** (uses a fake tracer; asserts no input/output + child spans)

```python
def test_phoenix_sink_redacts_and_builds_waterfall(monkeypatch):
    import backend.fanout as fo
    spans = []
    class FakeSpan:
        def __init__(s, name): s.name = name; s.attrs = {}; s.ended = None
        def set_attribute(s, k, v): s.attrs[k] = v
        def end(s, end_time=None): s.ended = end_time
    class FakeTracer:
        def start_span(s, name, context=None, start_time=None, openinference_span_kind=None):
            assert openinference_span_kind == openinference_span_kind.lower()  # lowercase contract
            sp = FakeSpan(name); sp.start = start_time; sp.kind = openinference_span_kind; spans.append(sp); return sp
    monkeypatch.setattr(fo, "_tracer", FakeTracer())
    monkeypatch.setattr(fo, "set_span_in_context", lambda sp: None, raising=False)
    ev = {"flag": True, "model": "g", "uncertainty": 0.2, "trackers": {"unc": {"score": 0.2}},
          "features": [{"label": "dosing"}], "io": {"user_msg": "SECRET", "response": "SECRET"}}
    perf = {"t0_ns": 1_000_000_000, "turn_ms": 100,
            "stages": {"pod_roundtrip": 60, "label_fetch": 10, "ranking": 5},
            "pod_stages": {"capture": 40, "sae": 15, "trackers": 3}}
    fo.PhoenixSink().emit(ev, perf)
    all_attrs = {k: v for sp in spans for k, v in sp.attrs.items()}
    assert "input.value" not in all_attrs and "output.value" not in all_attrs
    assert "SECRET" not in repr(all_attrs)
    assert spans[0].name == "chat-turn" and any(sp.name == "pod_roundtrip" for sp in spans)
    assert all_attrs["cognition.feature_labels"] == '["dosing"]'
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** (add to `fanout.py`; add import `from opentelemetry.trace import set_span_in_context` near top, guarded by the phoenix try-block in init)

```python
from opentelemetry.trace import set_span_in_context  # safe: otel always installed

_STAGE_KIND = {"pod_roundtrip": "tool", "capture": "chain", "sae": "chain",
               "trackers": "chain", "label_fetch": "tool", "ranking": "chain"}

def _stage_spans(perf: dict):
    """Ordered (name, kind, ms). pod_stages nest inside pod_roundtrip's window conceptually but are laid
    out sequentially from t0 for a readable waterfall."""
    order = ["pod_roundtrip", "capture", "sae", "trackers", "label_fetch", "ranking"]
    merged = {**perf.get("stages", {}), **perf.get("pod_stages", {})}
    return [(n, _STAGE_KIND[n], merged[n]) for n in order if n in merged]

class PhoenixSink:
    name = "phoenix"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        if _tracer is None:
            return
        MS = 1_000_000
        t0 = (perf or {}).get("t0_ns")
        if t0 is None:
            return
        parent = _tracer.start_span("chat-turn", start_time=t0, openinference_span_kind="llm")
        parent.set_attribute("cognition.flag", bool(event.get("flag")))
        parent.set_attribute("cognition.severity", event.get("severity", "info"))
        if event.get("uncertainty") is not None:
            parent.set_attribute("cognition.uncertainty", float(event["uncertainty"]))
        parent.set_attribute("cognition.feature_labels", json.dumps([f["label"] for f in event.get("features", [])]))
        for tid, tr in (event.get("trackers") or {}).items():
            if tr.get("score") is not None:
                parent.set_attribute(f"cognition.tracker.{tid}", float(tr["score"]))
        ctx = set_span_in_context(parent)
        off = t0
        for name, kind, ms in _stage_spans(perf or {}):
            child = _tracer.start_span(name, context=ctx, start_time=off, openinference_span_kind=kind)
            child.set_attribute("cognition.stage", name)
            end = off + int(ms * MS); child.end(end_time=end); off = end
        parent.end(end_time=t0 + int((perf or {}).get("turn_ms", 0) * MS))
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(fanout): PhoenixSink redacted span waterfall"`

---

### Task 8: Phoenix register (redacted + local-only) + `StoreSink` + wire registration

**Files:** Modify `backend/fanout.py`; `backend/tests/test_fanout.py`.

**Interfaces:**
- Consumes: `observability.STORE`, `to_redacted_view`.
- Produces: `StoreSink` (`name="store"`); `init_sponsors` now registers Sentry/Phoenix/Store sinks; Phoenix registers only when endpoint host is local.

- [ ] **Step 1: Write the failing test**

```python
def test_store_sink_records_redacted(monkeypatch):
    import backend.fanout as fo
    from backend import observability
    store = observability.ObservabilityStore()
    monkeypatch.setattr(observability, "STORE", store)
    fo.StoreSink().emit({"message_id": "m", "ts": 1.0, "io": {"user_msg": "SECRET"},
                         "features": [], "trackers": {}}, {"turn_ms": 5})
    assert store.snapshot()["totals"]["turns"] == 1
    assert "SECRET" not in repr(store.snapshot())

def test_phoenix_not_registered_when_remote(monkeypatch):
    import backend.fanout as fo
    monkeypatch.setattr(fo.config, "PHOENIX_ENDPOINT", "https://cloud.phoenix.example.com")
    assert fo._phoenix_is_local() is False
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** (add to `fanout.py`; complete `init_sponsors`)

```python
from urllib.parse import urlparse
from . import observability

class StoreSink:
    name = "store"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        observability.STORE.record(observability.to_redacted_view(event), perf)

def _phoenix_is_local() -> bool:
    host = (urlparse(config.PHOENIX_ENDPOINT).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", ""}
```

Then extend `init_sponsors()` after the Sentry block:

```python
    if _phoenix_is_local():
        try:
            from phoenix.otel import register
            provider = register(project_name="glassbox",
                                endpoint=f"{config.PHOENIX_ENDPOINT.rstrip('/')}/v1/traces",
                                batch=True, auto_instrument=False)
            globals()["_tracer"] = provider.get_tracer(__name__)
            register_sink(PhoenixSink())
        except Exception as e:  # noqa: BLE001 — never let a missing sidecar break the app
            print(f"[fanout] Phoenix not initialized ({e}); continuing without it.")
    else:
        print(f"[fanout] PHOENIX_ENDPOINT is non-local; PhoenixSink disabled (PII fail-closed).")
    if _sentry_on:
        register_sink(SentrySink())
    register_sink(StoreSink())   # store is always on (in-process, redacted)
```

Also set the redaction env vars at import top of `fanout.py` (belt-and-suspenders, before phoenix import):

```python
import os
os.environ.setdefault("OPENINFERENCE_HIDE_INPUTS", "true")
os.environ.setdefault("OPENINFERENCE_HIDE_OUTPUTS", "true")
```

- [ ] **Step 4: Run, expect pass** — Run: `pytest backend/tests/test_fanout.py -v` Expected: PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(fanout): StoreSink + local-only Phoenix register + sink wiring"`

---

### Task 9: `pod_client.PodError` sanitization (privacy)

**Files:** Modify `backend/pod_client.py` (`_post` ~:40, `PodError`); Test `backend/tests/test_pod_client.py`.

**Interfaces:**
- Produces: `PodError(status: int, stage: str)` whose `str()` is `f"pod {stage} failed: HTTP {status}"` — never the response body.

- [ ] **Step 1: Write the failing test**

```python
def test_pod_error_carries_no_body():
    from backend.pod_client import PodError
    e = PodError(500, "turn")
    s = str(e)
    assert "500" in s and "turn" in s
    # simulate a body that echoes the answer — it must NOT be in the exception
    assert "RESPONSE_BODY" not in s
```

- [ ] **Step 2: Run, expect fail** (current `PodError` likely embeds `r.text`).
- [ ] **Step 3: Implement** — change `PodError` to `__init__(self, status, stage)` storing both, `__str__` returns the sanitized message. In `_post`, on non-2xx: `print(f"[pod] {stage} HTTP {r.status_code}: {r.text[:200]}")` (LOCAL log only) then `raise PodError(r.status_code, stage)`. Pass `stage` (e.g. `"turn"`) into `_post`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "fix(pod_client): sanitize PodError (no response body to Sentry)"`

---

### Task 10: Provider guard (forbid raw-text POST for patient data)

**Files:** Modify `backend/science/feature_provider.py` (`get_provider` / `NeuronpediaProvider.features_for` ~:102-139); Test `backend/tests/test_feature_provider.py` (new, no torch — test the guard only).

**Interfaces:**
- Produces: a `GLASSBOX_ALLOW_REMOTE_FEATURES` env gate; `NeuronpediaProvider.features_for` raises `RuntimeError("remote feature POST forbidden for patient data")` unless the gate is set.

- [ ] **Step 1: Write the failing test**

```python
def test_neuronpedia_provider_forbidden_by_default(monkeypatch):
    monkeypatch.delenv("GLASSBOX_ALLOW_REMOTE_FEATURES", raising=False)
    from backend.science.feature_provider import NeuronpediaProvider
    import pytest
    with pytest.raises(RuntimeError, match="forbidden"):
        NeuronpediaProvider().features_for(["msg"], cap=5)
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — at the top of `NeuronpediaProvider.features_for`, add `if not os.getenv("GLASSBOX_ALLOW_REMOTE_FEATURES"): raise RuntimeError("remote feature POST forbidden for patient data")`. (Import `os` if absent.)
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "fix(science): forbid Neuronpedia raw-text POST for patient data"`

---

### Task 11: Wire `init_sponsors`/`fanout` call sites (keep app working)

**Files:** Modify `backend/app.py` (init at :29, fanout at :58). No new behavior beyond passing the 2nd arg once Task 13 lands; here just confirm the rewrite integrates.

- [ ] **Step 1:** Confirm `app.py:16` import still resolves (`from .fanout import fanout, init_sponsors`).
- [ ] **Step 2: Run smoke** — Run: `python -c "import backend.app"` Expected: no ImportError.
- [ ] **Step 3: Run full backend suite** — Run: `pytest backend/tests/ -q` Expected: PASS (existing + new).
- [ ] **Step 4: Commit** — `git add -A && git commit -m "chore(obs): integrate fanout rewrite into app"`

---

# WORKSTREAM 2 — Timings (Lane-B touch ⚠, additive-key gated)

### Task 12: `gpu_service.py` — additive stage timings

**Files:** Modify `backend/gpu_service.py` `/turn` (:230-249). **Lane B — requires torch on the pod; test via contract test with monkeypatched stages.**

**Interfaces:**
- Produces: `/turn` response gains `"timings": {"capture": ms, "sae": ms, "trackers": ms}` (floats).

- [ ] **Step 1: Write the failing contract test** (`backend/tests/test_gpu_service.py`, monkeypatch the heavy fns)

```python
def test_turn_returns_timings(monkeypatch):
    # monkeypatch engine/_capture/_sae_candidates/_score_trackers to cheap stubs, call the /turn handler,
    # assert "timings" in result with keys capture/sae/trackers, all floats >= 0.
    ...
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — wrap `_capture(...)` (`:235`) → `capture_ms`; `_pooled_activations`+`_sae_candidates` (`:240-242`) → `sae_ms`; `_score_trackers` (`:243`) → `trackers_ms` using `t=time.perf_counter()` deltas (×1000). Add `result["timings"] = {"capture": capture_ms, "sae": sae_ms, "trackers": trackers_ms}`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(gpu_service): additive per-stage timings on /turn"`

---

### Task 13: `analyze.py` perf (3-tuple) + `pod_client` timings passthrough + `app.py` t0_ns

**Files:** Modify `backend/analyze.py` (`analyze_turn` :131-179, `_real_turn` :122-128), `backend/pod_client.py` (`turn` :107-112), `backend/app.py` (:47, :58, :69). Update callers: `app.py:69`, `backend/validation/batch_medqa_observability.py`, `backend/tests/test_analyze.py`.

**Interfaces:**
- Produces: `analyze_turn(...) -> (str, CognitionEvent, dict)`; `perf = {"turn_ms": float, "stages": {pod_roundtrip, label_fetch, ranking}, "pod_stages": {capture, sae, trackers}}`; app sets `perf["t0_ns"] = time.time_ns()`.

- [ ] **Step 1: Write the failing test** (extend `test_analyze.py`)

```python
def test_analyze_turn_returns_perf(monkeypatch):
    # stub pod_client.turn to return {answer, candidates, trackers, reliable, timings:{capture:10,sae:5,trackers:2}}
    # call analyze_turn; assert 3-tuple; perf has turn_ms, stages.pod_roundtrip, pod_stages.capture==10
    ...
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** —
  - `pod_client.turn()` returns the dict including `timings` (already passthrough after Task 12; ensure not dropped).
  - `analyze.py`: wrap `pod_client.turn()` with `perf_counter` → `pod_roundtrip`; wrap `_rank_features` → split `label_fetch`/`ranking` (or measure `_rank_features` whole as `ranking` + a sub-timer for label fetch if separable; if not, record `ranking` only). Build `perf`; merge `r.get("timings", {})` into `perf["pod_stages"]`. Return `(answer, event, perf)`. Fallback path: `perf = {"turn_ms": ..., "stages": {...}, "pod_stages": {}}`.
  - `app.py`: `turn_start_ns = time.time_ns()` before the call; `answer, event, perf = analyze_turn(...)`; `perf["t0_ns"] = turn_start_ns`; `fanout(payload, perf)` at :58.
  - Update all callers to unpack 3-tuple.
- [ ] **Step 4: Run** — Run: `pytest backend/tests/test_analyze.py backend/tests/test_smoke.py -q` Expected: PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs): perf timings threaded analyze→app→fanout (3-tuple)"`

---

# WORKSTREAM 3 — Read API + Sentry REST + endpoint

### Task 14: `sentry_api.list_recent_issues`

**Files:** Create `backend/sentry_api.py`; Test `backend/tests/test_sentry_api.py`.

**Interfaces:**
- Produces: `async def list_recent_issues(limit=15) -> list[dict]`; `def deep_link() -> str | None`.

- [ ] **Step 1: Write the failing test** (monkeypatch httpx)

```python
import asyncio, backend.sentry_api as sa

def test_list_recent_issues_projects_fields(monkeypatch):
    class R:
        def raise_for_status(s): pass
        def json(s): return [{"id":"1","shortId":"G-1","title":"t","culprit":"c","level":"warning",
                              "count":5,"userCount":2,"lastSeen":"x","permalink":"p","extra":"drop"}]
    class C:
        def __init__(s,**k): pass
        async def __aenter__(s): return s
        async def __aexit__(s,*a): return False
        async def get(s,*a,**k): return R()
    monkeypatch.setattr(sa.httpx, "AsyncClient", C)
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "t"); monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "o")
    monkeypatch.setattr(sa.config, "SENTRY_PROJECT_SLUG", "p")
    out = asyncio.run(sa.list_recent_issues())
    assert out[0]["shortId"] == "G-1" and "extra" not in out[0]

def test_list_recent_issues_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "")
    assert asyncio.run(sa.list_recent_issues()) == []
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** (the spec §4 `list_recent_issues` body verbatim + `deep_link` = `f"{config.SENTRY_ORG_URL}/organizations/{config.SENTRY_ORG_SLUG}/issues/"` when configured else `None`).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs): sentry_api recent-issues read path"`

---

### Task 15: `GET /api/observability` + `POST /api/observability/eval`

**Files:** Modify `backend/app.py` (after `/api/health` :34). Test extend `backend/tests/test_smoke.py`.

**Interfaces:**
- Consumes: `STORE.snapshot`, `runtime.health_payload`, `sentry_api`, `coherence_eval` (Task 18 — guard import lazily so this task ships before it).
- Produces: `GET /api/observability` (§7 shape), `POST /api/observability/eval`.

- [ ] **Step 1: Write the failing test**

```python
def test_observability_endpoint_shape(monkeypatch, client):  # client = TestClient(app)
    from backend import observability, sentry_api
    monkeypatch.setattr(observability.STORE, "snapshot", lambda: {"totals":{"turns":0},"trackers":{},"confident_wrong":[]})
    monkeypatch.setattr(sentry_api, "list_recent_issues", lambda limit=15: [])
    r = client.get("/api/observability"); j = r.json()
    assert "health" in j and "sentry" in j and j["phoenix_ui_url"]
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**

```python
@app.get("/api/observability")
async def observability_endpoint():
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload()
    snap["sentry"] = {"configured": bool(config.SENTRY_AUTH_TOKEN),
                      "deep_link": sentry_api.deep_link(),
                      "issues": await sentry_api.list_recent_issues()}
    snap["phoenix_ui_url"] = config.PHOENIX_UI_URL
    return snap

@app.post("/api/observability/eval")
async def observability_eval():
    from starlette.concurrency import run_in_threadpool
    from . import coherence_eval
    return await run_in_threadpool(coherence_eval.run_eval)
```

- [ ] **Step 4: Run, expect pass** (the `/eval` route may 500 until Task 18; test only the GET here).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs): /api/observability snapshot endpoint"`

---

# WORKSTREAM 4 — Frontend Observability page (Lane C)

### Task 16: types + mock + api + hook

**Files:** Modify `frontend/src/types.ts`, `mock.ts`, `api.ts`; Create `frontend/src/useObservability.ts`.

**Interfaces:**
- Produces: `ObservabilitySnapshot` type; `getObservability(): Promise<ObservabilitySnapshot>`; `useObservability(): {snapshot, status}`; `DEMO_OBSERVABILITY_SNAPSHOT`.

- [ ] **Step 1:** Add `ObservabilitySnapshot` to `types.ts` mirroring §7 (totals, flag_rate, uncertainty_series, trackers, confident_wrong, top_features, latency, health, sentry, phoenix_ui_url).
- [ ] **Step 2:** Add `getObservability` to `api.ts`: `export const getObservability = () => fetch("/api/observability").then(r => r.json());`
- [ ] **Step 3:** Create `useObservability.ts`: polls every 2000ms via `setInterval` + `getObservability`, cleans up on unmount, returns `{snapshot, status}`.
- [ ] **Step 4:** Add `DEMO_OBSERVABILITY_SNAPSHOT` to `mock.ts` (non-empty: one tracker series, one confident_wrong with `feature_labels` only, latency stages, healthy `health`, 2 sentry issues).
- [ ] **Step 5: Build check** — Run: `cd frontend && npx tsc --noEmit` Expected: no type errors.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(obs-ui): snapshot types + hook + mock"`

---

### Task 17: `ObservabilityPage` + panels + App toggle

**Files:** Create `frontend/src/ObservabilityPage.tsx`; Modify `frontend/src/App.tsx`, `styles.css`.

**Interfaces:**
- Consumes: `useObservability`, `ObservabilitySnapshot`, existing style tokens.

- [ ] **Step 1:** Build `ObservabilityPage` with the six sections (TrackerStrip sparklines, ConfidentWrongFeed showing **labels+scores only**, FeatureLeaderboard, LatencyHealth bars + recon gauge + pod badge, `<iframe src={snapshot.phoenix_ui_url}>`, SentryStrip mapping issues to `<a href={permalink} target="_blank">`). Hand-rolled SVG for sparklines/bars. Reuse `.panel`, `--accent`, `--ok`.
- [ ] **Step 2:** `App.tsx`: add `const [view, setView] = useState<"chat"|"observe">("chat")` + a top-nav with two buttons. **Render both; hide chat with `style={{display: view==="chat"?"":"none"}}` so `ChatPanel`/stream never unmount.** Mount `<ObservabilityPage/>` when `view==="observe"`.
- [ ] **Step 3:** Add observe styles to `styles.css` (sparkline stroke = `--ink`, hot = `--accent`, healthy = `--ok`; iframe `width:100%;height:520px;border:1px solid var(--line)`).
- [ ] **Step 4: Build + visual check** — Run: `cd frontend && npx tsc --noEmit && npm run build` Expected: clean. (Manual: `npm run dev`, toggle to Observe, confirm panels render from live/empty snapshot and chat survives toggle-back.)
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs-ui): Observability page + nav toggle"`

---

# WORKSTREAM 5 — Coherence eval

### Task 18: `coherence_eval.run_eval` (Phoenix batch)

**Files:** Create `backend/coherence_eval.py`; Test `backend/tests/test_coherence_eval.py`.

**Interfaces:**
- Produces: `run_eval(limit=200) -> {"evaluated": int, "off_domain": int}`; `MEDICAL_DOMAIN_CONTEXT`; `_RUNNING` guard.

- [ ] **Step 1: Write the failing test** (monkeypatch `phoenix.client.Client`, `LLM`, `evaluate_dataframe`, `to_annotation_dataframe`, `log_span_annotations_dataframe`)

```python
def test_run_eval_sends_only_labels_and_domain(monkeypatch):
    import backend.coherence_eval as ce, pandas as pd
    sent = {}
    df = pd.DataFrame({"context.span_id":["s1"], "attributes.cognition.feature_labels":['["dosing","python error"]']})
    class FakeSpans:
        def get_spans_dataframe(s,**k): return df
        def log_span_annotations_dataframe(s,**k): sent["logged"]=True
    class FakeClient:
        def __init__(s,**k): s.spans=FakeSpans()
    monkeypatch.setattr(ce, "Client", FakeClient)
    monkeypatch.setattr(ce, "LLM", lambda **k: object())
    monkeypatch.setattr(ce, "create_classifier", lambda **k: "clf")
    def fake_eval(dataframe, evaluators, **k):
        assert set(dataframe.columns) == {"context.span_id","feature_labels","domain_context"}  # HARD-RESTRICT
        assert "PROMPT" not in dataframe.to_string()
        d = dataframe.copy(); d["feature_coherence_score"]=[{"label":"off_domain","score":0.0,"explanation":"x"}]; return d
    monkeypatch.setattr(ce, "evaluate_dataframe", fake_eval)
    monkeypatch.setattr(ce, "to_annotation_dataframe", lambda dataframe: dataframe)
    out = ce.run_eval()
    assert out["evaluated"] == 1 and sent.get("logged")
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** the spec §4 `coherence_eval.py` (imports, `MEDICAL_DOMAIN_CONTEXT`, `_RUNNING` module flag with try/finally, hard-restrict df to `["context.span_id","feature_labels","domain_context"]`, `exit_on_error=False`, count `off_domain` from the `feature_coherence_score` dicts).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(obs): Phoenix feature-coherence batch eval"`

---

### Task 19: Wire eval button + full suite

**Files:** Modify `frontend/src/ObservabilityPage.tsx` (a "Run coherence eval" button → POST `/api/observability/eval`, disabled+spinner while running), `frontend/src/api.ts` (`runEval()`).

- [ ] **Step 1:** Add `runEval = () => fetch("/api/observability/eval",{method:"POST"}).then(r=>r.json())` to `api.ts`; button in `LatencyHealth`/eval panel with `useState` `running`.
- [ ] **Step 2: Full backend suite** — Run: `pytest backend/tests/ -q` Expected: PASS.
- [ ] **Step 3: Frontend build** — Run: `cd frontend && npx tsc --noEmit && npm run build` Expected: clean.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(obs): coherence-eval trigger button + suite green"`

---

## Self-Review notes (run before execution)

- **Spec coverage:** §2 Sentry → Tasks 5-6; §2 Phoenix span → 7-8; §2 taxonomy → 3; §4 observability.py → 2-3; sentry_api → 14; coherence_eval → 18; fanout rewrite → 4-8; config → 1; app endpoints → 15; analyze/pod/gpu timings → 9,12,13; frontend → 16-17,19; §6 invariants → tests in 2,3,5,6,7,8,9,10,18. ✓
- **Privacy invariants** (§6) each have a test (redaction 2/3, frame-vars 5, no-input/output 7, PodError 9, provider 10, eval-restrict 18, store guard 3). ✓
- **Type consistency:** `perf` shape (`t0_ns/turn_ms/stages/pod_stages`) identical across Tasks 7,13; stage names from the canonical list everywhere; `ObservabilitySnapshot` (16) mirrors §7.
- **Cross-lane risk:** Task 12 (gpu_service) is the only Lane-B touch; additive key → WS3/4 work even if WS2 lags.
