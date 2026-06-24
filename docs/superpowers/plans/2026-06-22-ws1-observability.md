# WS1 — Observability: drop Arize, keep & rename Sentry Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

Goal: Remove Arize/Phoenix from GlassBox entirely, keep Sentry but genericize its medical wording (and fix the `SENTRY_ORG` vs `SENTRY_ORG_SLUG` env mismatch), and rename the per-turn `cognition_event` to `introspection_event` (class `CognitionEvent` → `IntrospectionEvent`) across backend, frontend, and the fixture.

Architecture: The CPU orchestration backend (`backend/app.py`, never imports torch) builds one redacted event per chat turn and fans it to a sink registry in `backend/fanout.py`; today that registry has three sinks (Sentry, Phoenix, in-process Store). This workstream deletes the Phoenix sink and its two standalone eval modules, deletes the `/api/observability/eval` endpoint and `phoenix_ui_url`, neutralizes the Sentry fingerprint/message strings, and performs the mechanical `CognitionEvent` → `IntrospectionEvent` rename through `schema.py`, `events.py`, `analyze.py`, the frontend `types.ts`/hook, and the fixture. The event wire shape (`schema_version` stays `"1.0"`) is unchanged.

Tech Stack: Python 3.12, FastAPI, pydantic v2, sentry-sdk[fastapi], httpx, pytest; React + TypeScript + Vite frontend; `uv` for backend deps and test running.

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

## Sequencing note (read before starting)

This workstream is written against the **current flat `backend/config.py` globals** (`config.SENTRY_SEND_IO`, `config.PHOENIX_ENDPOINT`, `config.SENTRY_ORG_SLUG`, etc.), because WS1 can land before or after WS0's `AppConfig` refactor. WS1 touches four files that WS0 also rewrites — `backend/config.py`, `backend/schema.py`, `backend/app.py`, `.env.example`. Each task that edits a shared file flags it in its Interfaces block. If WS0 has **already** landed when you run a task, the config reads in that task move from `config.X` (flat global) to `cfg.observability.sentry.X` per the AppConfig Contract migration map — the field names in the contract are already neutral, so only the access path changes, never the logic. Do WS1's deletions and the rename exactly as written; only swap the config access path if WS0 is already in.

## File Structure

- `backend/coherence_eval.py` — **DELETE** (Phoenix batch feature-coherence eval).
- `backend/phoenix_eval_features.py` — **DELETE** (post-hoc Phoenix annotators).
- `scripts/run_phoenix.sh` — **DELETE** (Phoenix server launcher).
- `backend/tests/test_coherence_eval.py` — **DELETE** (tests for deleted module).
- `backend/tests/test_phoenix_eval_features.py` — **DELETE** (tests for deleted module).
- `backend/fanout.py` — strip `PhoenixSink`, `_phoenix_is_local`, `_stage_spans`, `_STAGE_KIND`, the OTel import, the OpenInference env block, the Phoenix branch of `init_sponsors`; neutralize Sentry fingerprint + message strings; rename `capture_cognition_alarm` → `capture_introspection_alarm`.
- `backend/observability.py` — drop the `feature_incoherence`/`phoenix-eval` taxonomy row.
- `backend/app.py` — delete `/api/observability/eval`; drop `snap["phoenix_ui_url"]`; neutralize the test-sentry hint text; update the `capture_cognition_alarm` import + calls to the renamed symbol.
- `backend/schema.py` — rename class `CognitionEvent` → `IntrospectionEvent`; update docstrings + fixture filename reference.
- `backend/events.py` — rename `build_cognition_event` → `build_introspection_event`; update imports + docstring.
- `backend/analyze.py` — update `CognitionEvent`/`build_cognition_event` references to the renamed symbols.
- `backend/batch_medqa_observability.py` — drop Phoenix prints + `config.PHOENIX_ENDPOINT` read; neutralize medical framing in docstring; remove the committed `glassbox-dev-secret` example token.
- `backend/smoke_fanout.py` — drop Phoenix references; point at the renamed fixture; neutralize the medical message text.
- `backend/config.py` — remove `PHOENIX_ENDPOINT`, `PHOENIX_UI_URL`, `EVAL_LLM_PROVIDER`, `EVAL_LLM_MODEL`; add `SENTRY_ORG`/`SENTRY_PROJECT` env fallbacks to the slug reads. (SHARED with WS0.)
- `backend/sentry_api.py` — accept `SENTRY_ORG`/`SENTRY_PROJECT` as fallbacks (covered by the config change; no logic change).
- `backend/requirements.txt` — NOT edited by WS1; owned and **deleted by WS0** (Task 18). The `arize-phoenix`/`openinference-instrumentation` removal is moot once the file is gone (those deps live only in `requirements.txt`, never in `pyproject.toml`).
- `.env.example` — remove `PHOENIX_COLLECTOR_ENDPOINT`; rename `SENTRY_ORG`/`SENTRY_PROJECT` → `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG`; remove the `glassbox-dev-secret` default; set `POD_URL=http://localhost:8001` (matches the contract `PodConfig.url` default + WS3/WS4 docs). (SHARED with WS0/WS3.)
- `backend/tests/test_fanout.py` — drop the two Phoenix tests; rename the alarm-symbol references; assert neutral fingerprint/message.
- `backend/tests/test_sentry_alarm.py` — rename `capture_cognition_alarm` references; assert neutral message text.
- `backend/tests/test_observability_endpoint.py` — drop `phoenix_ui_url` assertions.
- `backend/tests/test_events.py` — rename `build_cognition_event` references.
- `backend/tests/test_analyze.py` — rename `CognitionEvent` references.
- `backend/tests/test_api.py` — rename `CognitionEvent` references.
- `fixtures/cognition_event.sample.json` → `fixtures/introspection_event.sample.json` — `git mv` + neutralize the medical payload text.
- `frontend/src/types.ts` — rename interface `CognitionEvent` → `IntrospectionEvent`; drop `phoenix_ui_url` from `ObservabilitySnapshot`.
- `frontend/src/useCognitionStream.ts` → `frontend/src/useIntrospectionStream.ts` — rename hook + file.
- `frontend/src/App.tsx` — update the hook import + `CognitionEvent` type references.
- `frontend/src/mock.ts` — rename `CognitionEvent` type reference; drop `phoenix_ui_url` from the demo snapshot.
- `frontend/src/api.ts` — delete `runEval()`.
- `frontend/src/ObservabilityPage.tsx` — delete `PhoenixPanel`, the `runEval` import + eval state/handler + the "Run coherence eval" button; drop the `phoenix_ui_url` render.
- `frontend/src/instrument.ts` — neutralize the "medical tool" / Phoenix comments (copy only).
- `README.md` — purge Phoenix mentions + `cognition_event.sample.json`/`CognitionEvent` references + medical positioning (docs cleanup, enumerated edits in Task 11 Step 3d, verified by grep). `PLAN.md`/`LANES.md` are NOT touched by WS1 — they are deleted by WS4 (see Task 11 SHARED FILE note); purging Phoenix from them here would be thrown away.

---

### Task 1: Delete the Phoenix eval modules, their tests, and the launcher script

Files:
- Delete: `backend/coherence_eval.py`
- Delete: `backend/phoenix_eval_features.py`
- Delete: `backend/tests/test_coherence_eval.py`
- Delete: `backend/tests/test_phoenix_eval_features.py`
- Delete: `scripts/run_phoenix.sh`
- Test: the verification command is a grep + a pytest collection that proves the files and their tests are gone and nothing else imports them.

Interfaces:
- Consumes: nothing (these are leaf modules; only `backend/app.py`'s lazy `from . import coherence_eval` inside `/api/observability/eval` references `coherence_eval`, and that endpoint is removed in Task 3 — do Task 1 then Task 3 in order, or expect a stale import only inside that one endpoint until Task 3 lands).
- Produces: nothing. Removes symbols `coherence_eval.run_eval`, `coherence_eval.MEDICAL_DOMAIN_CONTEXT`, `phoenix_eval_features.evaluate_span` from the codebase.

Steps:

- [ ] Step 1: Write the failing guard test that asserts the deleted modules are no longer importable. Create `backend/tests/test_phoenix_removed.py`:
```python
"""WS1 guard: the Arize/Phoenix eval modules are deleted and not importable."""
import importlib

import pytest


@pytest.mark.parametrize("modname", [
    "backend.coherence_eval",
    "backend.phoenix_eval_features",
])
def test_phoenix_eval_modules_are_gone(modname):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)
```

- [ ] Step 2: Run the guard test and watch it FAIL (the modules still exist, so the import succeeds and `pytest.raises` fails):
```
uv run pytest backend/tests/test_phoenix_removed.py -q
```
Expected output (failure):
```
F                                                                        [100%]
...
DID NOT RAISE <class 'ModuleNotFoundError'>
1 failed, 1 passed in ...s
```
(One param fails because at least one module still imports; the assertion `DID NOT RAISE ModuleNotFoundError` is the signal.)

- [ ] Step 3: Delete the four Phoenix-eval files and the launcher script:
```
git rm backend/coherence_eval.py backend/phoenix_eval_features.py backend/tests/test_coherence_eval.py backend/tests/test_phoenix_eval_features.py scripts/run_phoenix.sh
```

- [ ] Step 4: Run the guard test and watch it PASS:
```
uv run pytest backend/tests/test_phoenix_removed.py -q
```
Expected output:
```
..                                                                       [100%]
2 passed in ...s
```

- [ ] Step 5: Commit:
```
git add backend/tests/test_phoenix_removed.py
git commit -m "feat(obs): delete Arize/Phoenix eval modules, tests, and launcher script"
```

---

### Task 2: Strip `PhoenixSink` and all Phoenix wiring from `backend/fanout.py`; neutralize Sentry fingerprint/message; rename the alarm entrypoint

Files:
- Modify: `backend/fanout.py` (lines 1-325 — whole-file edit; Step 3 replaces the ENTIRE file, so these per-block citations are informational only: header docstring lines 1-15, OpenInference env block lines 25-28, OTel import block lines 219-223, `PhoenixSink` block lines 238-262, `_phoenix_is_local` lines 275-277, the Phoenix branch of `init_sponsors` lines 302-314, the medical fingerprint/message lines 190/198, and the `capture_cognition_alarm` name lines 166/212).
- Test: `backend/tests/test_fanout.py`

Interfaces:
- Consumes: `config.SENTRY_DSN`, `config.SENTRY_ENVIRONMENT`, `config.SENTRY_RELEASE`, `config.SENTRY_SEND_IO`, `config.DISABLED_TRACKERS` (current flat globals — if WS0 has landed these become `cfg.sentry_dsn`, `cfg.observability.sentry.environment`, `cfg.observability.sentry.release`, `cfg.observability.sentry.send_io`, `cfg.probes.disabled`).
- Produces: `capture_introspection_alarm(event: dict, *, flush: bool = False) -> bool` (RENAMED from `capture_cognition_alarm`); `init_sponsors() -> None` (Phoenix branch removed); `SentrySink`, `StoreSink`, `fanout`, `report_error`, `sentry_enabled`, `_flag_reason`, `_scrub_pii`, `_level`, `register_sink` (UNCHANGED names). `PhoenixSink`, `_phoenix_is_local`, `_stage_spans`, `_STAGE_KIND`, `_tracer`, `set_span_in_context` are REMOVED.
- NOTE: `backend/app.py` (Task 3) imports `capture_cognition_alarm` and `_flag_reason` from this module — Task 3 updates that import to the renamed symbol. Run Task 2 then Task 3 in order.

Steps:

- [ ] Step 1: Replace the two Phoenix tests in `backend/tests/test_fanout.py` with a neutral-Sentry assertion, and neutralize the one retained test that hardcodes a medical literal. First, in `backend/tests/test_fanout.py`, DELETE `test_phoenix_not_registered_when_remote` (lines 54-57) and `test_phoenix_sink_redacts_and_builds_waterfall` (lines 60-83). Then, in the RETAINED `test_scrub_pii_walks_whole_event_not_just_three_sections`, neutralize the medical sample string (it is test-internal data, but it carries the medical phrasing the new alarm no longer emits): REPLACE the `"message"` value on line 130 `        "message": "Confident-wrong medical answer",  # key 'message' is NOT a PII key -> preserved` with `        "message": "Confident-wrong answer",  # key 'message' is NOT a PII key -> preserved`, and REPLACE the assertion on line 137 `    assert out["message"] == "Confident-wrong medical answer"` with `    assert out["message"] == "Confident-wrong answer"`. Then APPEND this new test to the end of the file:
```python
def test_capture_alarm_neutral_fingerprint_and_message(monkeypatch):
    """WS1: the Sentry alarm fingerprint and message carry no medical wording."""
    import backend.fanout as fo
    captured = {}
    fake = type("S", (), {})()
    fake.capture_message = lambda msg, level=None: captured.setdefault("msg", msg)
    fake.flush = lambda timeout=3: None

    class Scope:
        fingerprint = None
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def set_tag(s, *a): pass
        def set_context(s, k, v): captured.setdefault("ctx", v)
    fake.new_scope = lambda: Scope()
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", fake)
    monkeypatch.setattr(fo, "_sentry_on", True)

    ev = {
        "flag": True, "severity": "warning", "model": "m", "message_id": "m1",
        "uncertainty": 0.9, "uncertainty_proj": 1.0,
        "trackers": {"over_confidence": {"score": 0.9, "flag": True, "reliable": True}},
        "features": [{"label": "dosing"}],
        "io": {"user_msg": "SECRET", "response": "SECRET"},
    }
    captured.clear()
    captured["fp"] = None

    class CapturingScope(Scope):
        def __enter__(s):
            return s
        def __exit__(s, *a):
            captured["fp"] = s.fingerprint
            return False
    fake.new_scope = lambda: CapturingScope()

    assert fo.capture_introspection_alarm(ev, flush=True) is True
    assert "medical" not in captured["msg"].lower()
    assert captured["msg"] == "Confident-wrong answer — over_confidence"
    assert captured["fp"] == ["glassbox", "introspection", "over_confidence"]
    assert "SECRET" not in repr(captured["ctx"])
```

- [ ] Step 2: Run `test_fanout.py` and watch it FAIL (the renamed function `capture_introspection_alarm` does not exist yet, and the deleted Phoenix tests were referenced by nothing else):
```
uv run pytest backend/tests/test_fanout.py -q
```
Expected output (failure):
```
...
AttributeError: module 'backend.fanout' has no attribute 'capture_introspection_alarm'
1 failed, ... passed in ...s
```

- [ ] Step 3: Rewrite `backend/fanout.py` to remove all Phoenix code, neutralize the Sentry strings, and rename the alarm function. Replace the ENTIRE file with:
```python
"""The SINGLE sponsor seam (contract #4). Receives a plain JSON-serializable dict
(the introspection_event, tensors already stripped via IntrospectionEvent.model_dump()).
Adding/removing a sponsor = editing ONLY this file. NOTHING here imports torch or
touches the GPU.

OWNER: Lane A.

One redacted event → many interchangeable Sinks:
  - Sentry = the INCIDENT view. A flagged turn becomes a grouped Issue (fingerprinted by
             which probe fired), with a PII-free introspection payload. Prompt/response are
             NEVER sent by default; SENTRY_SEND_IO is an opt-in (default off), regex-scrubbed.
  - Store  = the in-process redacted snapshot behind /api/observability (+ future MCP).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from . import config
from . import observability

_sentry_on = False


# ---------------------------------------------------------------------------
# Sink protocol + registry
# ---------------------------------------------------------------------------

@runtime_checkable
class Sink(Protocol):
    name: str
    def emit(self, event: dict, perf: dict | None = None) -> None: ...


_SINKS: list[Sink] = []


def register_sink(s: Sink) -> None:
    _SINKS.append(s)


def fanout(event: dict, perf: dict | None = None) -> None:
    """Fan one finished introspection_event (+ optional perf) to every sink. CPU only, post-stream."""
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


# ---------------------------------------------------------------------------
# PII handling: structured regex scrub + level clamp + before_send
# ---------------------------------------------------------------------------

_VALID_LEVELS = {"fatal", "critical", "error", "warning", "info", "debug"}
_PII_KEYS = {"question", "answer", "user_msg", "response", "prompt", "messages", "io"}

# Structured-PII patterns scrubbed from any free-text before it leaves the box. Catches
# emails/phones/SSNs — NOT unstructured identifiers like names. To keep raw I/O out entirely keep
# SENTRY_SEND_IO off (the default) so prompt/response never leave at all.
_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[phone]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
]


def _scrub(s: str) -> str:
    for pat, repl in _PII:
        s = pat.sub(repl, s)
    return s


def _level(severity: str) -> str:
    return severity if severity in _VALID_LEVELS else "warning"


def _scrub_pii(event: dict, hint: dict):
    """Sentry before_send (supersedes main's _scrub_event). ALWAYS: zero every exception-frame
    local (they carry the prompt/io) + regex-scrub structured PII from every string anywhere.
    When SENTRY_SEND_IO is off (default), ALSO hard-redact any forbidden-key value so prompt/
    response can never leak even if some future code path attaches them."""
    for v in event.get("exception", {}).get("values", []):
        for fr in v.get("stacktrace", {}).get("frames", []):
            fr["vars"] = {}
    redact_keys = set() if config.SENTRY_SEND_IO else _PII_KEYS

    def walk(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in redact_keys:
                    o[k] = "[scrubbed]"
                elif isinstance(o[k], str):
                    o[k] = _scrub(o[k])
                else:
                    walk(o[k])
        elif isinstance(o, list):
            for i in range(len(o)):
                if isinstance(o[i], str):
                    o[i] = _scrub(o[i])
                else:
                    walk(o[i])

    walk(event)
    return event


# ---------------------------------------------------------------------------
# SentrySink — the alarm (quiet; fires only on flagged turns)
# ---------------------------------------------------------------------------

def _bucket(u):
    return "high" if (u or 0) >= 0.66 else "med" if (u or 0) >= 0.33 else "low"


def _observable_trackers(trackers: dict | None) -> dict:
    return {tid: tr for tid, tr in (trackers or {}).items() if tid not in config.DISABLED_TRACKERS}


_FLAG_PRIORITY = ("harmful", "harmful_prompt", "over_confidence")


def _flag_reason(event: dict) -> str:
    """Sentry grouping key = which reliable probe fired, so distinct failure modes become
    distinct Issues (a harmful answer and an over-confident one are triaged differently)."""
    trackers = _observable_trackers(event.get("trackers"))
    for tid in _FLAG_PRIORITY:
        if trackers.get(tid, {}).get("flag"):
            return tid
    for tid, t in trackers.items():
        if t.get("flag"):
            return tid
    return "confident_wrong"


def sentry_enabled() -> bool:
    return _sentry_on


def capture_introspection_alarm(event: dict, *, flush: bool = False) -> bool:
    """Emit a PII-free Sentry issue when ``event['flag']`` is true.

    Called automatically by ``fanout()`` for every chat turn. Use directly for tests or
    custom hooks. Returns True when a message was queued for Sentry."""
    if not _sentry_on:
        return False
    if not event.get("flag"):
        return False
    import sentry_sdk

    reason = _flag_reason(event)
    introspection = {
        "uncertainty": event.get("uncertainty"),
        "uncertainty_proj": event.get("uncertainty_proj"),
        "trackers": _observable_trackers(event.get("trackers")),
        "top_features": [f["label"] for f in (event.get("features") or [])[:15]],
    }
    if config.SENTRY_SEND_IO:
        io = event.get("io") or {}
        introspection["question"] = _scrub(io.get("user_msg", "") or "")
        introspection["answer"] = _scrub(io.get("response", "") or "")
    try:
        with sentry_sdk.new_scope() as scope:
            scope.fingerprint = ["glassbox", "introspection", reason]
            scope.set_tag("model", str(event.get("model") or "unknown"))
            scope.set_tag("event_type", "confident_wrong")
            scope.set_tag("flag_reason", str(reason))
            scope.set_tag("message_id", str(event.get("message_id") or "unknown"))
            scope.set_tag("uncertainty_bucket", _bucket(event.get("uncertainty")))
            scope.set_context("introspection", introspection)
            sentry_sdk.capture_message(
                f"Confident-wrong answer — {reason}", level=_level(event.get("severity", "warning"))
            )
        # Flush so short requests and the default transport queue cannot drop the alarm.
        sentry_sdk.flush(timeout=3)
        print(f"[fanout] sentry alarm emitted: reason={reason} message_id={event.get('message_id')}")
        return True
    except Exception as e:  # noqa: BLE001 — log but never break fanout / the chat stream
        print(f"[fanout] sentry alarm failed ({reason}, {event.get('message_id')}): {e}")
        return False


class SentrySink:
    name = "sentry"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        capture_introspection_alarm(event, flush=True)


# ---------------------------------------------------------------------------
# StoreSink + init_sponsors
# ---------------------------------------------------------------------------

class StoreSink:
    name = "store"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        observability.STORE.record(observability.to_redacted_view(event), perf)


def init_sponsors() -> None:
    """Call once at FastAPI startup. Safe to call with missing config — each sponsor
    is independently optional so the app still runs locally without a DSN."""
    global _sentry_on

    # --- Sentry (incident view) ---
    if config.SENTRY_DSN:
        import sentry_sdk
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            environment=config.SENTRY_ENVIRONMENT,
            release=config.SENTRY_RELEASE,        # None → Sentry auto-detects git SHA
            traces_sample_rate=0.0,
            send_default_pii=False,               # no IPs / headers / raw I/O by default
            include_local_variables=False,        # CRITICAL: frame locals carry the prompt + io.*
            include_source_context=False,
            max_request_body_size="never",
            enable_logs=True,                     # stdlib logging → Sentry (sentry-sdk >= 2.35)
            before_send=_scrub_pii,               # whole-event scrub (supersedes main's _scrub_event)
        )
        _sentry_on = True

    if _sentry_on:
        register_sink(SentrySink())
    register_sink(StoreSink())   # store is always on (in-process, redacted)


async def _claude_judge(event: dict) -> None:
    """Anthropic honesty-judge: adjudicate whether a flagged answer is hallucinated; patch the event.
    DEFERRED — privacy-killed in cloud (needs the answer); future LOCAL-model slot only."""
    # TODO(local model): score hallucination from {feature labels, probe scores} only.
    ...
```

- [ ] Step 4: Run `test_fanout.py` and watch it PASS:
```
uv run pytest backend/tests/test_fanout.py -q
```
Expected output:
```
.........                                                                [100%]
9 passed in ...s
```
(7 retained tests + 1 renamed `test_sentry_sink_quiet...` already present + the new neutral-fingerprint test; the two Phoenix tests are gone.)

- [ ] Step 5: Commit:
```
git add backend/fanout.py backend/tests/test_fanout.py
git commit -m "refactor(obs): remove PhoenixSink + Phoenix wiring from fanout; neutralize Sentry fingerprint/message; rename alarm to capture_introspection_alarm"
```

---

### Task 3: Remove `/api/observability/eval` and `phoenix_ui_url` from `backend/app.py`; rewire to the renamed alarm; neutralize the hint text

Files:
- Modify: `backend/app.py` (line 18 import, lines 29-30 `lifespan` docstring Phoenix mention, lines 49-71 the observability + eval endpoints, lines 105/110 test-sentry body, line 127 replay body)
- Test: `backend/tests/test_observability_endpoint.py`, `backend/tests/test_sentry_alarm.py`
- SHARED FILE: `backend/app.py` is also edited by WS0 (lifespan `app.state.config` wiring). WS1's edits here are: delete the eval endpoint, drop `phoenix_ui_url`, swap the alarm symbol, neutralize copy. Keep them surgical so WS0's lifespan rewrite merges cleanly.

Interfaces:
- Consumes: `fanout.capture_introspection_alarm` (renamed in Task 2), `fanout._flag_reason`, `fanout.sentry_enabled`, `config.SENTRY_DSN`, `config.SENTRY_AUTH_TOKEN` (flat globals; under WS0 these become `cfg.sentry_dsn` / `cfg.sentry_auth_token`).
- Produces: endpoints `GET /api/observability` (now without `phoenix_ui_url`), `POST /api/observability/test-sentry`, `POST /api/observability/replay-sentry` (UNCHANGED routes). `POST /api/observability/eval` is REMOVED.

Steps:

- [ ] Step 1: Update `backend/tests/test_observability_endpoint.py` to assert `phoenix_ui_url` is GONE. First, REPLACE the module header docstring (lines 1-5) — it names the deleted `coherence_eval`, which the Task 12 Step 2 "no stale references in tests" grep matches:
```python
"""Tests for GET /api/observability (Task 15).

Verifies the merged-shape contract from §7 of the design spec.
Does NOT test POST /api/observability/eval (coherence_eval doesn't exist yet).
"""
```
WITH:
```python
"""Tests for GET /api/observability.

Verifies the merged-shape contract: store snapshot + health + Sentry.
The /api/observability/eval route was removed in WS1.
"""
```
Then, in `test_observability_endpoint_shape`, REPLACE the docstring (line 31) `    """GET /api/observability returns the §7 merged shape with health, sentry, phoenix_ui_url."""` with `    """GET /api/observability returns the §7 merged shape with health and sentry."""` (so no `phoenix` substring survives — the Task 11 Step 4 grep recurses `backend/tests`), then REMOVE the literal line `    assert "phoenix_ui_url" in j` (the third "merged keys present" assertion, line 45) and the trailing block `    # phoenix_ui_url is a string` + `    assert isinstance(j["phoenix_ui_url"], str)` (lines 63-64). Then ADD this new test at the end of the file:
```python
def test_observability_endpoint_has_no_phoenix(monkeypatch):
    """WS1: phoenix_ui_url is gone and there is no /api/observability/eval route."""
    monkeypatch.setattr(observability.STORE, "snapshot", lambda: _empty_snapshot())
    async def _no_issues(limit=15):
        return []
    monkeypatch.setattr(sentry_api, "list_recent_issues", _no_issues)

    r = client.get("/api/observability")
    assert r.status_code == 200
    assert "phoenix_ui_url" not in r.json()

    # The eval endpoint is deleted → FastAPI returns 405 (route absent) for the POST.
    r2 = client.post("/api/observability/eval")
    assert r2.status_code in (404, 405)
```
Also neutralize the medical literal in the RETAINED `test_observability_endpoint_sentry_issues_forwarded` (it is a mocked Sentry-issue title, test-internal data, but it still carries the medical phrasing the code path no longer emits): REPLACE the `"title"` value on line 96 `        return [{"shortId": "G-1", "title": "Confident-wrong medical answer", "level": "warning",` with `        return [{"shortId": "G-1", "title": "Confident-wrong answer", "level": "warning",`.

- [ ] Step 2: Run both affected test files and watch them FAIL (`phoenix_ui_url` is still returned, and `/api/observability/eval` still routes):
```
uv run pytest backend/tests/test_observability_endpoint.py backend/tests/test_sentry_alarm.py -q
```
Expected output (failure):
```
...
assert 'phoenix_ui_url' not in {...}
1 failed, ... passed in ...s
```
(`test_sentry_alarm.py` still passes here because it patches `backend.app.capture_cognition_alarm` by string — that monkeypatch silently no-ops once the import is renamed; the real failure surfaces in Step 4 after the rename. The endpoint test is the failing signal in this step.)

- [ ] Step 3a: Update the import on line 18 of `backend/app.py`:
```python
from .fanout import fanout, init_sponsors, capture_introspection_alarm, sentry_enabled, _flag_reason
```

- [ ] Step 3a-2: Neutralize the `lifespan` docstring body (lines 27-31; the Phoenix mentions are on lines 29-30) so it no longer names Phoenix (the Task 11 Step 4 grep recurses `backend --include='*.py'` and would match it). REPLACE:
```python
    runtime.start_loading() kicks off the background model/SAE load (per-request readiness;
    requests use the synthetic fallback until it's ready). init_sponsors() is the SINGLE
    sponsor seam (contract #4): Sentry (incident view) + Phoenix (analytics view), each
    independently optional so a missing DSN or Phoenix sidecar logs a warning and the app
    still serves.
```
WITH:
```python
    runtime.start_loading() kicks off the background model/SAE load (per-request readiness;
    requests use the synthetic fallback until it's ready). init_sponsors() is the SINGLE
    sponsor seam (contract #4): the Sentry incident view (independently optional — a missing
    DSN logs a warning and the app still serves) plus the always-on in-process store.
```

- [ ] Step 3b: Replace the `observability_endpoint` function (lines 49-62) to drop the Phoenix line — REPLACE:
```python
@app.get("/api/observability")
async def observability_endpoint():
    """Return the in-process store snapshot merged with health, Sentry, and Phoenix UI URL.
    Never contains prompt or response text (store holds only redacted views)."""
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload()
    snap["sentry"] = {
        "emit_configured": bool(config.SENTRY_DSN),
        "configured": bool(config.SENTRY_AUTH_TOKEN),
        "deep_link": sentry_api.deep_link(),
        "issues": await sentry_api.list_recent_issues(),
    }
    snap["phoenix_ui_url"] = config.PHOENIX_UI_URL
    return snap
```
WITH:
```python
@app.get("/api/observability")
async def observability_endpoint():
    """Return the in-process store snapshot merged with health and Sentry.
    Never contains prompt or response text (store holds only redacted views)."""
    snap = observability.STORE.snapshot()
    snap["health"] = runtime.health_payload()
    snap["sentry"] = {
        "emit_configured": bool(config.SENTRY_DSN),
        "configured": bool(config.SENTRY_AUTH_TOKEN),
        "deep_link": sentry_api.deep_link(),
        "issues": await sentry_api.list_recent_issues(),
    }
    return snap
```

- [ ] Step 3c: DELETE the entire `observability_eval` endpoint (lines 65-71):
```python
@app.post("/api/observability/eval")
async def observability_eval():
    """Run a Phoenix batch coherence eval (labels-only, no prompt/response).
    coherence_eval is imported lazily so this task ships before that module exists."""
    from starlette.concurrency import run_in_threadpool
    from . import coherence_eval  # noqa: PLC0415 — intentionally lazy
    return await run_in_threadpool(coherence_eval.run_eval)
```
(Remove the whole block — there is no replacement.)

- [ ] Step 3d: In `observability_test_sentry`, update the alarm call on line 105 and neutralize the hint on line 110. REPLACE the line `sent = capture_cognition_alarm(event, flush=True)` with `sent = capture_introspection_alarm(event, flush=True)`, and REPLACE the hint string:
```python
        "hint": "Check Sentry Issues for “Confident-wrong medical answer”. Chat turns alarm the same way when a probe flags.",
```
WITH:
```python
        "hint": "Check Sentry Issues for “Confident-wrong answer”. Chat turns alarm the same way when a probe flags.",
```
Also, in the synthetic test `event` dict (lines 89-104), REPLACE the medical I/O strings on line 103:
```python
        "io": {"user_msg": "[synthetic test — not a real patient]", "response": "[synthetic test]"},
```
WITH:
```python
        "io": {"user_msg": "[synthetic test]", "response": "[synthetic test]"},
```

- [ ] Step 3e: In `observability_replay_sentry`, update line 127. REPLACE `sent = capture_cognition_alarm({**view, "io": {}}, flush=True)` with `sent = capture_introspection_alarm({**view, "io": {}}, flush=True)`.

- [ ] Step 3f: Update `backend/tests/test_sentry_alarm.py` for the renamed symbol. REPLACE both occurrences of the string `"backend.app.capture_cognition_alarm"` (lines 74 and 91) with `"backend.app.capture_introspection_alarm"`. (The `fo.capture_cognition_alarm(...)` calls on lines 30 and 32 are updated in Task 6, since they live in the fanout-symbol test and depend on the rename there — leave them for now; this step only fixes the two `backend.app.*` patch targets.)

- [ ] Step 4: Run both test files and watch them PASS:
```
uv run pytest backend/tests/test_observability_endpoint.py backend/tests/test_sentry_alarm.py -q
```
Expected output:
```
.......                                                                  [100%]
... passed in ...s
```

- [ ] Step 5: Commit:
```
git add backend/app.py backend/tests/test_observability_endpoint.py backend/tests/test_sentry_alarm.py
git commit -m "feat(obs): drop /api/observability/eval and phoenix_ui_url; rewire to capture_introspection_alarm; neutralize copy"
```

---

### Task 4: Drop the Phoenix taxonomy row from `backend/observability.py`

Files:
- Modify: `backend/observability.py` (lines 109-112, `concern_taxonomy`)
- Test: `backend/tests/test_observability_taxonomy.py` (new)

Interfaces:
- Consumes: `config.DISABLED_TRACKERS` (flat global; under WS0 → `cfg.probes.disabled`).
- Produces: `ObservabilityStore.concern_taxonomy() -> list[dict]` returning exactly two rows (`confident_wrong`, `instrument_unhealthy`); the `feature_incoherence` / `phoenix-eval` row is REMOVED.

Steps:

- [ ] Step 1: Write the failing test. Create `backend/tests/test_observability_taxonomy.py`:
```python
"""WS1: the concern taxonomy no longer advertises a Phoenix-eval source."""
from backend.observability import ObservabilityStore


def test_taxonomy_has_no_phoenix_source():
    tax = ObservabilityStore().concern_taxonomy()
    ids = {row["id"] for row in tax}
    sources = {row["source"] for row in tax}
    assert "feature_incoherence" not in ids
    assert "phoenix-eval" not in sources
    assert ids == {"confident_wrong", "instrument_unhealthy"}
```

- [ ] Step 2: Run it and watch it FAIL (the third row still exists):
```
uv run pytest backend/tests/test_observability_taxonomy.py -q
```
Expected output (failure):
```
F                                                                        [100%]
...
assert {'confident_wrong', 'feature_incoherence', 'instrument_unhealthy'} == {'confident_wrong', 'instrument_unhealthy'}
1 failed in ...s
```

- [ ] Step 3: Edit `concern_taxonomy` in `backend/observability.py`. REPLACE:
```python
    def concern_taxonomy(self) -> list[dict]:
        return [{"id": "confident_wrong", "description": "low-uncertainty flagged answer", "source": "tracker"},
                {"id": "instrument_unhealthy", "description": "pod/SAE/label failure", "source": "system"},
                {"id": "feature_incoherence", "description": "off-domain feature activation", "source": "phoenix-eval"}]
```
WITH:
```python
    def concern_taxonomy(self) -> list[dict]:
        return [{"id": "confident_wrong", "description": "low-uncertainty flagged answer", "source": "tracker"},
                {"id": "instrument_unhealthy", "description": "pod/SAE/label failure", "source": "system"}]
```

- [ ] Step 4: Run it and watch it PASS:
```
uv run pytest backend/tests/test_observability_taxonomy.py -q
```
Expected output:
```
.                                                                        [100%]
1 passed in ...s
```

- [ ] Step 5: Commit:
```
git add backend/observability.py backend/tests/test_observability_taxonomy.py
git commit -m "feat(obs): drop feature_incoherence/phoenix-eval row from concern taxonomy"
```

---

### Task 5: Remove `PHOENIX_*` / `EVAL_LLM_*` from `backend/config.py` and add `SENTRY_ORG`/`SENTRY_PROJECT` env fallbacks

Files:
- Modify: `backend/config.py` (lines 154, 157, 158-159, 163-164)
- Test: `backend/tests/test_config_sentry_env.py` (new)
- SHARED FILE: `backend/config.py` is fully rewritten by WS0 into `AppConfig`. WS1's contribution here is: (a) remove the four Phoenix/eval globals, and (b) fix the `SENTRY_ORG`/`SENTRY_PROJECT` vs `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG` mismatch by accepting both env names. If WS0 has landed, the org/project slugs already live at `cfg.observability.sentry.org_slug` / `.project_slug`; in that world this task's behavior is provided by `.env.example` using the correct `SENTRY_ORG_SLUG`/`SENTRY_PROJECT_SLUG` names (Task 9) plus the loader, and this task collapses to just confirming the four Phoenix globals are absent. As written below, it targets the CURRENT flat config.

Interfaces:
- Consumes: env vars `SENTRY_ORG_SLUG`, `SENTRY_ORG`, `SENTRY_PROJECT_SLUG`, `SENTRY_PROJECT`.
- Produces: `config.SENTRY_ORG_SLUG` (now falls back to `SENTRY_ORG` when `SENTRY_ORG_SLUG` unset), `config.SENTRY_PROJECT_SLUG` (falls back to `SENTRY_PROJECT`). REMOVES `config.PHOENIX_ENDPOINT`, `config.PHOENIX_UI_URL`, `config.EVAL_LLM_PROVIDER`, `config.EVAL_LLM_MODEL`.
- NOTE: `backend/sentry_api.py` reads `config.SENTRY_ORG_SLUG`/`config.SENTRY_PROJECT_SLUG` unchanged — the fallback is resolved inside `config.py`, so `sentry_api.py` needs no edit.

Steps:

- [ ] Step 1: Write the failing test. Create `backend/tests/test_config_sentry_env.py`:
```python
"""WS1: config drops PHOENIX_*/EVAL_LLM_* and accepts SENTRY_ORG/SENTRY_PROJECT as fallbacks."""
import importlib

import backend.config as config


def test_phoenix_and_eval_globals_removed():
    for name in ("PHOENIX_ENDPOINT", "PHOENIX_UI_URL", "EVAL_LLM_PROVIDER", "EVAL_LLM_MODEL"):
        assert not hasattr(config, name), f"{name} should be deleted from config"


def test_sentry_org_slug_falls_back_to_sentry_org(monkeypatch):
    monkeypatch.delenv("SENTRY_ORG_SLUG", raising=False)
    monkeypatch.delenv("SENTRY_PROJECT_SLUG", raising=False)
    monkeypatch.setenv("SENTRY_ORG", "my-org")
    monkeypatch.setenv("SENTRY_PROJECT", "my-proj")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.SENTRY_ORG_SLUG == "my-org"
        assert reloaded.SENTRY_PROJECT_SLUG == "my-proj"
    finally:
        monkeypatch.delenv("SENTRY_ORG", raising=False)
        monkeypatch.delenv("SENTRY_PROJECT", raising=False)
        importlib.reload(config)


def test_sentry_org_slug_preferred_over_sentry_org(monkeypatch):
    monkeypatch.setenv("SENTRY_ORG_SLUG", "explicit-slug")
    monkeypatch.setenv("SENTRY_ORG", "legacy-org")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.SENTRY_ORG_SLUG == "explicit-slug"
    finally:
        monkeypatch.delenv("SENTRY_ORG_SLUG", raising=False)
        monkeypatch.delenv("SENTRY_ORG", raising=False)
        importlib.reload(config)
```

- [ ] Step 2: Run it and watch it FAIL (the Phoenix globals still exist and there is no fallback):
```
uv run pytest backend/tests/test_config_sentry_env.py -q
```
Expected output (failure):
```
FF.                                                                      [100%]
...
assert not hasattr(config, 'PHOENIX_ENDPOINT')
2 failed, 1 passed in ...s
```

- [ ] Step 3a: In `backend/config.py`, DELETE the Phoenix endpoint line (line 154):
```python
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
```
(Remove the line entirely.)

- [ ] Step 3b: In the "Observability surfaces" block (lines 156-164), REPLACE:
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
WITH:
```python
# --- Observability surfaces (Sentry read path) ---
# Accept the canonical *_SLUG names; fall back to the shorter SENTRY_ORG/SENTRY_PROJECT
# so a .env using either spelling works (was a code-vs-.env mismatch before WS1).
SENTRY_ORG_SLUG     = os.getenv("SENTRY_ORG_SLUG") or os.getenv("SENTRY_ORG", "")
SENTRY_PROJECT_SLUG = os.getenv("SENTRY_PROJECT_SLUG") or os.getenv("SENTRY_PROJECT", "")
SENTRY_AUTH_TOKEN   = os.getenv("SENTRY_AUTH_TOKEN", "")                    # internal-integration, event:read+project:read
SENTRY_API_BASE     = os.getenv("SENTRY_API_BASE", "https://sentry.io").rstrip("/")
SENTRY_ORG_URL      = os.getenv("SENTRY_ORG_URL", "https://sentry.io")     # deep-link host
```

- [ ] Step 4: Run it and watch it PASS:
```
uv run pytest backend/tests/test_config_sentry_env.py -q
```
Expected output:
```
...                                                                      [100%]
3 passed in ...s
```

- [ ] Step 5: Commit:
```
git add backend/config.py backend/tests/test_config_sentry_env.py
git commit -m "feat(config): remove PHOENIX_*/EVAL_LLM_* globals; accept SENTRY_ORG/SENTRY_PROJECT as slug fallbacks"
```

---

### Task 6: Rename `CognitionEvent` → `IntrospectionEvent` and `build_cognition_event` → `build_introspection_event` in the backend

Files:
- Modify: `backend/schema.py` (lines 1-3 docstring, line 52 class name, line 78 docstring)
- Modify: `backend/events.py` (lines 1, 9, 12, 22, 41)
- Modify: `backend/analyze.py` (lines 1, 14, 15, 147, 184)
- Modify: `backend/app.py` (lines 138, 163 — the `chat`/`analyze` endpoint docstrings that name `CognitionEvent`; the import/calls were already updated in Task 3)
- Modify: `backend/tests/test_events.py` (lines 1, 5, 26, 42, 64)
- Modify: `backend/tests/test_analyze.py` (lines 2, 11, 20, 97)
- Modify: `backend/tests/test_api.py` (lines 7, 37, 48)
- Modify: `backend/tests/test_sentry_alarm.py` (line 4 test-function name, lines 30, 32 — the `fo.capture_cognition_alarm(...)` calls, line 44 assertion)
- SHARED FILE: `backend/schema.py` is the 3-way contract (mirrored by `frontend/src/types.ts` + the fixture). The frontend mirror is renamed in Task 8 and the fixture in Task 7. `schema_version` stays `"1.0"` — the wire shape is unchanged, only the Python class symbol changes.

Interfaces:
- Consumes: nothing new.
- Produces: `schema.IntrospectionEvent` (was `CognitionEvent`), `events.build_introspection_event(...) -> IntrospectionEvent` (was `build_cognition_event`). `schema.IO`, `schema.Tracker`, `schema.Feature`, `schema.Adjudication`, `schema.TokenLine`, `schema.SCHEMA_VERSION` UNCHANGED.

Steps:

- [ ] Step 1: Update the tests first so they reference the new symbols (TDD: tests define the target API). Apply these exact replacements:
  - `backend/tests/test_events.py`: replace `from backend.events import build_cognition_event` → `from backend.events import build_introspection_event`, and all four `build_cognition_event(` calls → `build_introspection_event(`.
  - `backend/tests/test_analyze.py`: replace `from backend.schema import CognitionEvent` → `from backend.schema import IntrospectionEvent`, and the three `CognitionEvent` usages (lines 11, 20, 97) → `IntrospectionEvent`.
  - `backend/tests/test_api.py`: replace the import `from backend.schema import CognitionEvent` (line 7) → `from backend.schema import IntrospectionEvent`, and the two `CognitionEvent(` call sites (lines 37, 48) → `IntrospectionEvent(`.
  - `backend/tests/test_sentry_alarm.py`: rename the TEST FUNCTION `def test_capture_cognition_alarm_requires_flag(monkeypatch):` (line 4) → `def test_capture_introspection_alarm_requires_flag(monkeypatch):` (so no `capture_cognition_alarm` substring survives in the green suite — Task 12 Step 2 greps for it). Replace the two `fo.capture_cognition_alarm(` calls (lines 30, 32) → `fo.capture_introspection_alarm(`. Also update the assertion `assert captured["msg"].startswith("Confident-wrong medical answer")` (line 44) → `assert captured["msg"].startswith("Confident-wrong answer")`.

- [ ] Step 2: Run the affected tests and watch them FAIL (the backend still exports the old names):
```
uv run pytest backend/tests/test_events.py backend/tests/test_analyze.py backend/tests/test_api.py backend/tests/test_sentry_alarm.py -q
```
Expected output (failure):
```
...
ImportError: cannot import name 'build_introspection_event' from 'backend.events'
... failed, ... passed in ...s
```

- [ ] Step 3a: Rename in `backend/schema.py`. REPLACE the module docstring (lines 1-5):
```python
"""SHARED SOURCE OF TRUTH for the cognition_event contract.

Mirrored byte-for-byte by frontend/src/types.ts and fixtures/cognition_event.sample.json.
No lane changes a field here without 3-way agreement (see LANES.md, contract #1).
"""
```
WITH:
```python
"""SHARED SOURCE OF TRUTH for the introspection_event contract.

Mirrored byte-for-byte by frontend/src/types.ts and fixtures/introspection_event.sample.json.
No lane changes a field here without 3-way agreement (contract #1).
"""
```
Then REPLACE `class CognitionEvent(BaseModel):` (line 52) with `class IntrospectionEvent(BaseModel):`, and in the `TokenLine` docstring (line 78) REPLACE `Final line of /api/chat is the CognitionEvent.` with `Final line of /api/chat is the IntrospectionEvent.`.

- [ ] Step 3b: Rename in `backend/events.py`. REPLACE the file's docstring line 1 `"""Assemble the ONE cognition_event per message. Pure CPU. No torch, no tensors out.` with `"""Assemble the ONE introspection_event per message. Pure CPU. No torch, no tensors out.`. REPLACE `from .schema import IO, CognitionEvent` (line 9) with `from .schema import IO, IntrospectionEvent`. REPLACE `def build_cognition_event(` (line 12) with `def build_introspection_event(`. REPLACE the return annotation `) -> CognitionEvent:` (line 22) with `) -> IntrospectionEvent:`. REPLACE `return CognitionEvent(` (line 41) with `return IntrospectionEvent(`.

- [ ] Step 3c: Rename in `backend/analyze.py`. REPLACE the docstring line 1 `"""Shared turn analysis: generate (or synthesize) a response and assemble its CognitionEvent.` with `"""Shared turn analysis: generate (or synthesize) a response and assemble its IntrospectionEvent.`. REPLACE `from .events import build_cognition_event` (line 14) with `from .events import build_introspection_event`. REPLACE `from .schema import CognitionEvent` (line 15) with `from .schema import IntrospectionEvent`. REPLACE the return annotation `) -> tuple[str, CognitionEvent, dict]:` (line 147) with `) -> tuple[str, IntrospectionEvent, dict]:`. REPLACE `event = build_cognition_event(` (line 184) with `event = build_introspection_event(`.

- [ ] Step 3d: Rename the leftover docstring references in `backend/app.py` (the import + calls were already swapped in Task 3; these two are prose-only and would otherwise trip the Task 11 Step 4 / Task 12 `CognitionEvent` grep). REPLACE the `chat` endpoint docstring (line 138) `    """Generate a turn, stream status + token replay, then one CognitionEvent line."""` with `    """Generate a turn, stream status + token replay, then one IntrospectionEvent line."""`. REPLACE the `analyze` endpoint docstring (line 163) `    """Post-hoc / non-streaming variant: returns the CognitionEvent as JSON."""` with `    """Post-hoc / non-streaming variant: returns the IntrospectionEvent as JSON."""`.

- [ ] Step 4: Run the affected tests and watch them PASS:
```
uv run pytest backend/tests/test_events.py backend/tests/test_analyze.py backend/tests/test_api.py backend/tests/test_sentry_alarm.py backend/tests/test_fanout.py -q
```
Expected output:
```
............................                                             [100%]
... passed in ...s
```

- [ ] Step 5: Commit:
```
git add backend/schema.py backend/events.py backend/analyze.py backend/app.py backend/tests/test_events.py backend/tests/test_analyze.py backend/tests/test_api.py backend/tests/test_sentry_alarm.py
git commit -m "refactor: rename CognitionEvent->IntrospectionEvent and build_cognition_event->build_introspection_event"
```

---

### Task 7: Rename the fixture file and neutralize its medical payload

Files:
- Rename: `fixtures/cognition_event.sample.json` → `fixtures/introspection_event.sample.json`
- Modify: the renamed fixture's `io` text + the deprecated tracker keys to neutral, non-medical content (keeping the same SHAPE so it still validates against `IntrospectionEvent`).
- Modify: `backend/smoke_fanout.py` (line 30 fixture path + lines 1-9 docstring + line 9 message text + the Phoenix step) to point at the new filename and drop Phoenix.
- Test: `backend/tests/test_fixture_validates.py` (new)

Interfaces:
- Consumes: `backend.schema.IntrospectionEvent` (Task 6).
- Produces: `fixtures/introspection_event.sample.json` (validates against `IntrospectionEvent`, `schema_version` still `"1.0"`).
- NOTE: `backend/smoke_fanout.py` calls `fanout.init_sponsors()` and `fanout.fanout()` (Phoenix already removed in Task 2) — its only remaining Phoenix references are in copy + the fixture path.

Steps:

- [ ] Step 1: Write the failing validation test. Create `backend/tests/test_fixture_validates.py`:
```python
"""WS1: the renamed introspection fixture exists, validates, and carries no medical text."""
import json
import pathlib

from backend.schema import IntrospectionEvent

_FIX = pathlib.Path(__file__).parents[2] / "fixtures" / "introspection_event.sample.json"
_OLD = pathlib.Path(__file__).parents[2] / "fixtures" / "cognition_event.sample.json"


def test_old_fixture_name_is_gone():
    assert not _OLD.exists(), "old cognition_event.sample.json must be renamed"


def test_introspection_fixture_validates():
    raw = json.loads(_FIX.read_text())
    ev = IntrospectionEvent(**raw)
    assert ev.schema_version == "1.0"
    assert ev.flag is True


def test_introspection_fixture_has_no_medical_text():
    blob = _FIX.read_text().lower()
    for term in ("metformin", "pregnancy", "trimester", "patient", "clinical", "ibuprofen"):
        assert term not in blob, f"fixture still contains medical term {term!r}"
```

- [ ] Step 2: Run it and watch it FAIL (the renamed file does not exist yet):
```
uv run pytest backend/tests/test_fixture_validates.py -q
```
Expected output (failure):
```
...
FileNotFoundError: .../fixtures/introspection_event.sample.json
... failed in ...s
```

- [ ] Step 3a: Rename the fixture with git:
```
git mv fixtures/cognition_event.sample.json fixtures/introspection_event.sample.json
```

- [ ] Step 3b: Replace the ENTIRE contents of `fixtures/introspection_event.sample.json` with this neutralized, same-shape payload:
```json
{
  "schema_version": "1.0",
  "type": "event",
  "message_id": "11111111-2222-3333-4444-555555555555",
  "ts": 1718841600.0,
  "model": "unsloth/gemma-3-4b-it",
  "layer": 17,
  "io": {
    "user_msg": "Is it safe to mix bleach and vinegar for cleaning?",
    "response": "Yes, mixing bleach and vinegar is completely safe and makes an effective cleaner with no precautions needed."
  },
  "uncertainty": 0.83,
  "uncertainty_proj": 1.27,
  "uncertainty_proj_pre": 0.91,
  "flag": true,
  "severity": "warning",
  "trackers": {
    "uncertainty":     {"score": 0.83, "proj": 1.27, "flag": true,  "reliable": true, "proj_pre": 0.91, "alert_direction": "high", "user_defined": false, "status": "ready"},
    "harmful":         {"score": 0.12, "proj": -0.40, "flag": false, "reliable": true, "alert_direction": "high", "user_defined": false, "status": "ready"},
    "hallucination":   {"score": 0.71, "proj": 0.90, "flag": true,  "reliable": true, "alert_direction": "high", "user_defined": false, "status": "ready"},
    "risk_awareness":  {"score": 0.24, "proj": -1.15, "flag": true, "reliable": true, "alert_direction": "low", "user_defined": false, "status": "ready"},
    "over_confidence": {"score": 0.55, "proj": 0.30, "flag": false, "reliable": true, "alert_direction": "high", "user_defined": true,  "status": "ready"}
  },
  "features": [
    {"index": 1622, "label": "household chemicals", "act": 7.4, "source": "17-gemmascope-2-res-16k", "caveat": "auto-interp label, may be unreliable", "tracked": null},
    {"index": 8801, "label": "hedging / expressions of caution", "act": 5.1, "source": "17-gemmascope-2-res-16k", "caveat": "auto-interp label, may be unreliable", "tracked": "uncertainty"},
    {"index": 4412, "label": "safety & risk", "act": 4.8, "source": "17-gemmascope-2-res-16k", "caveat": "auto-interp label, may be unreliable", "tracked": null}
  ],
  "adjudication": {
    "verdict": "likely_hallucinated",
    "rationale": "Mixing bleach and vinegar releases toxic chlorine gas; the answer asserts the opposite with unwarranted certainty.",
    "by": "claude"
  }
}
```

- [ ] Step 3c: Update `backend/smoke_fanout.py`. REPLACE the docstring (lines 1-11):
```python
"""See Sentry + Phoenix light up WITHOUT the GPU/model.

Fires the sample cognition_event (one flagged, one clean) through fanout().

  1. (optional) put SENTRY_DSN in .env
  2. (optional) start Phoenix:   bash scripts/run_phoenix.sh   # -> http://localhost:6006
  3. run:                        python -m backend.smoke_fanout   # from repo root

Then check: Sentry Issues for "Confident-wrong medical answer", and Phoenix at :6006
for a 'chat-turn' LLM span with cognition.* attributes.
"""
```
WITH:
```python
"""See Sentry light up WITHOUT the GPU/model.

Fires the sample introspection_event (one flagged, one clean) through fanout().

  1. (optional) put SENTRY_DSN in .env
  2. run:                        python -m backend.smoke_fanout   # from repo root

Then check Sentry Issues for "Confident-wrong answer".
"""
```
REPLACE the fixture path (line 30) `pathlib.Path(__file__).parents[1] / "fixtures" / "cognition_event.sample.json"` with `pathlib.Path(__file__).parents[1] / "fixtures" / "introspection_event.sample.json"`.
REPLACE the clean-event I/O (line 45) `"io": {"user_msg": "What is the capital of France?", "response": "Paris."},` — keep it (already neutral). REPLACE the final two lines (56-57):
```python
    time.sleep(3)  # let Phoenix's batched OTLP exporter flush before the process exits
    print("done — check Sentry Issues + Phoenix at http://localhost:6006")
```
WITH:
```python
    print("done — check Sentry Issues")
```

- [ ] Step 4: Run the fixture test and the broader event suite to confirm the rename + neutralization is sound:
```
uv run pytest backend/tests/test_fixture_validates.py backend/tests/test_events.py -q
```
Expected output:
```
......                                                                   [100%]
6 passed in ...s
```

- [ ] Step 5: Commit:
```
git add fixtures/introspection_event.sample.json backend/smoke_fanout.py backend/tests/test_fixture_validates.py
git commit -m "refactor: rename fixture to introspection_event.sample.json and neutralize medical payload; drop Phoenix from smoke script"
```

---

### Task 8: Rename the frontend event type + hook; drop `phoenix_ui_url` from `types.ts` and `mock.ts`

Files:
- Modify: `frontend/src/types.ts` (line 39 interface, line 63 comment, line 71 union, line 144 `phoenix_ui_url`)
- Rename: `frontend/src/useCognitionStream.ts` → `frontend/src/useIntrospectionStream.ts` (lines 1-3 comment, line 6 import, line 11 export name, line 13 state type)
- Modify: `frontend/src/App.tsx` (lines 1-2 comment, line 6 import, line 7 hook import, line 26 hook call, line 28 state type)
- Modify: `frontend/src/mock.ts` (line 2 import, line 19 comment, line 20 type, lines 23/27-30/42-54/57-60 the `DEMO_EVENT` medical payload, lines 96/100-103 the `DEMO_OBSERVABILITY_SNAPSHOT` medical labels, line 135 `phoenix_ui_url`)
- Test: the verification is a `tsc --noEmit` type check + a grep proving the old symbols are gone.

Interfaces:
- Consumes: nothing new.
- Produces: `types.IntrospectionEvent` (was `CognitionEvent`), `useIntrospectionStream` (was `useCognitionStream`), `ObservabilitySnapshot` WITHOUT the `phoenix_ui_url` field, and a `DEMO_EVENT` whose io/feature/adjudication content is non-medical (mirrors `fixtures/introspection_event.sample.json`).
- NOTE: `frontend/src/ObservabilityPage.tsx` reads `snapshot.phoenix_ui_url` (Task 10 removes that read). Run Task 8 then Task 10, OR run Task 10 first — either order is fine as long as both land before the type-check at the end of Task 10. Within Task 8, the `tsc` check will report the `ObservabilityPage` error until Task 10; for Task 8's own gate, scope the type-check to the renamed files.

Steps:

- [ ] Step 1: Establish the failing signal — grep proves the old symbols still exist:
```
grep -rn "CognitionEvent\|useCognitionStream\|phoenix_ui_url" frontend/src
```
Expected output (the symbols are present — this is the state we are removing):
```
frontend/src/types.ts:39:export interface CognitionEvent {
frontend/src/types.ts:71:export type StreamLine = TokenLine | CognitionEvent;
frontend/src/types.ts:144:  phoenix_ui_url: string;
frontend/src/useCognitionStream.ts:11:export function useCognitionStream(...
frontend/src/App.tsx:7:import { useCognitionStream } from "./useCognitionStream";
frontend/src/mock.ts:20:export const DEMO_EVENT: CognitionEvent = {
frontend/src/mock.ts:135:  phoenix_ui_url: "http://localhost:6006",
... (and others)
```

- [ ] Step 2a: Edit `frontend/src/types.ts`. REPLACE `export interface CognitionEvent {` (line 39) with `export interface IntrospectionEvent {`. REPLACE the comment on line 63 `// Streamed per-token line (live mode); final line of /api/chat is the CognitionEvent.` with `// Streamed per-token line (live mode); final line of /api/chat is the IntrospectionEvent.`. REPLACE `export type StreamLine = TokenLine | CognitionEvent;` (line 71) with `export type StreamLine = TokenLine | IntrospectionEvent;`. DELETE the line `  phoenix_ui_url: string;` (line 144) from the `ObservabilitySnapshot` interface.

- [ ] Step 2b: Rename the hook file:
```
git mv frontend/src/useCognitionStream.ts frontend/src/useIntrospectionStream.ts
```
Then in `frontend/src/useIntrospectionStream.ts`: REPLACE the header comment lines 1-3 references to `useCognitionStream`/Lane C as-is but update the leading line `// Consume POST /api/chat as NDJSON...` block's mention — specifically REPLACE `import type { CognitionEvent, StreamLine } from "./types";` (line 6) with `import type { IntrospectionEvent, StreamLine } from "./types";`. REPLACE `export function useCognitionStream(opts?: { mock?: boolean }) {` (line 11) with `export function useIntrospectionStream(opts?: { mock?: boolean }) {`. REPLACE `const [event, setEvent] = useState<CognitionEvent | null>(null);` (line 13) with `const [event, setEvent] = useState<IntrospectionEvent | null>(null);`.

- [ ] Step 2c: Edit `frontend/src/App.tsx`. REPLACE the comment line 2 `// Claude verdict) reflecting the LATEST message's CognitionEvent. Real /api/chat (no mock).` with `// Claude verdict) reflecting the LATEST message's IntrospectionEvent. Real /api/chat (no mock).`. REPLACE `import type { CognitionEvent } from "./types";` (line 6) with `import type { IntrospectionEvent } from "./types";`. REPLACE `import { useCognitionStream } from "./useCognitionStream";` (line 7) with `import { useIntrospectionStream } from "./useIntrospectionStream";`. REPLACE `const { answer, event, status, send } = useCognitionStream();` (line 26) with `const { answer, event, status, send } = useIntrospectionStream();`. REPLACE `const [latest, setLatest] = useState<CognitionEvent | null>(null);` (line 28) with `const [latest, setLatest] = useState<IntrospectionEvent | null>(null);`.

- [ ] Step 2d: Edit `frontend/src/mock.ts`. REPLACE `import type { CognitionEvent, Feature, ObservabilitySnapshot } from "./types";` (line 2) with `import type { IntrospectionEvent, Feature, ObservabilitySnapshot } from "./types";`. REPLACE `export const DEMO_EVENT: CognitionEvent = {` (line 20) with `export const DEMO_EVENT: IntrospectionEvent = {`. DELETE the line `  phoenix_ui_url: "http://localhost:6006",` (line 135) from `DEMO_OBSERVABILITY_SNAPSHOT`.

- [ ] Step 2d-2: Neutralize the `DEMO_EVENT` medical payload in `frontend/src/mock.ts` so the shipped demo data mirrors the new `fixtures/introspection_event.sample.json` (household bleach+vinegar safety example) rather than the medical ibuprofen/pregnancy case. The neutralized `Task 7` backend fixture and this UI demo must agree.
  - REPLACE the comment (line 19) `// Ibuprofen / 3rd trimester confident-wrong case — uses the live probe pair only.` with `// Bleach + vinegar safety confident-wrong case — uses the live probe pair only.`
  - REPLACE the `message_id` (line 23) `  message_id: "demo-ibuprofen",` with `  message_id: "demo-bleach-vinegar",`
  - REPLACE the `io` block (lines 27-31):
```typescript
  io: {
    user_msg: "Is ibuprofen safe to take in the third trimester of pregnancy?",
    response:
      "Yes — ibuprofen is generally considered safe in moderation during the third trimester for managing pain and inflammation. A typical dose is fine, though it's always good to check with your doctor.",
  },
```
  WITH:
```typescript
  io: {
    user_msg: "Is it safe to mix bleach and vinegar for cleaning?",
    response:
      "Yes — mixing bleach and vinegar is completely safe and makes an effective cleaner with no precautions needed.",
  },
```
  - REPLACE the `features` array (lines 41-56) — keep the same shape and the two deliberately-suspect features (`f(..., true)`) so the "unverified badge" demo still works, but relabel to the household-chemicals domain:
```typescript
  features: [
    f(4412, "household chemicals", 6.2),
    f(9281, "cleaning products", 5.8),
    f(1530, "chemical reaction / fumes", 5.1),
    f(7044, "reassurance · “completely safe”", 4.7),
    f(2218, "toxic gas terms", 4.2),
    f(11907, "ventilation & exposure", 3.8),
    f(333, "programming / code", 3.4, true),
    f(6650, "following instructions", 3.0),
    f(8123, "hedging language", 2.6),
    f(14002, "coffee / café", 2.1, true),
    f(512, "temporal periods", 1.8),
    f(10330, "safety & risk", 1.5),
    f(391, "affirmation / yes", 1.3),
    f(7788, "second-person address", 1.1),
  ],
```
  - REPLACE the `adjudication.rationale` (lines 59-60):
```typescript
    rationale:
      "NSAIDs like ibuprofen are contraindicated in the third trimester. The model stated the opposite with unwarranted certainty — both the harmfulness and over-confidence probes crossed threshold.",
```
  WITH:
```typescript
    rationale:
      "Mixing bleach and vinegar releases toxic chlorine gas. The model stated the opposite with unwarranted certainty — both the harmfulness and over-confidence probes crossed threshold.",
```
  - Neutralize the medical `feature_labels` / `top_features` in `DEMO_OBSERVABILITY_SNAPSHOT` (same file, lines ~96 and 100-103) so the Observe-tab demo matches the repositioned product. REPLACE the `feature_labels` array (line 96) `      feature_labels: ["anticoagulant dosing", "drug interaction risk", "clinical dosage"],` with `      feature_labels: ["household chemicals", "chemical reaction / fumes", "safety & risk"],`. REPLACE the four `top_features` rows (lines 100-103):
```typescript
    { label: "anticoagulant dosing", count: 7, mean_act: 1.82 },
    { label: "pregnancy & gestation", count: 6, mean_act: 1.74 },
    { label: "medication / drug safety", count: 5, mean_act: 1.61 },
    { label: "clinical dosage", count: 4, mean_act: 1.45 },
```
  WITH:
```typescript
    { label: "household chemicals", count: 7, mean_act: 1.82 },
    { label: "chemical reaction / fumes", count: 6, mean_act: 1.74 },
    { label: "cleaning products", count: 5, mean_act: 1.61 },
    { label: "ventilation & exposure", count: 4, mean_act: 1.45 },
```

- [ ] Step 3: Run a scoped TypeScript check on the renamed surface (full-project check is gated at the end of Task 10, after `ObservabilityPage` is fixed). From `frontend/`:
```
cd frontend && npx tsc --noEmit 2>&1 | grep -E "useCognitionStream|CognitionEvent|useIntrospectionStream|IntrospectionEvent" || echo "no stale symbol errors"
```
Expected output (the only remaining reference is `ObservabilityPage.tsx` reading `phoenix_ui_url`, addressed in Task 10 — this grep filters to the rename surface):
```
no stale symbol errors
```

- [ ] Step 4: Confirm the old symbol names are gone everywhere except the not-yet-touched `ObservabilityPage.tsx`:
```
grep -rn "useCognitionStream\|CognitionEvent" frontend/src
```
Expected output:
```
(no matches)
```
(`phoenix_ui_url` still appears in `ObservabilityPage.tsx` until Task 10 — that is expected.)

- [ ] Step 5: Commit:
```
git add frontend/src/types.ts frontend/src/useIntrospectionStream.ts frontend/src/App.tsx frontend/src/mock.ts
git commit -m "refactor(ui): rename CognitionEvent->IntrospectionEvent, useCognitionStream->useIntrospectionStream; drop phoenix_ui_url from snapshot types"
```

---

### Task 9: Update `.env.example` (drop Phoenix env; fix Sentry slug names)

> NOTE: `backend/requirements.txt` is owned and deleted by WS0 (Task 18, its dependency-reconciliation task); do not edit it here. The former `arize-phoenix`/`openinference-instrumentation` removal is moot once the file is gone — those deps live ONLY in `requirements.txt`, never in `pyproject.toml`.

Files:
- Modify: `.env.example` (lines 9-10 Phoenix block; lines 25-26 `SENTRY_ORG`/`SENTRY_PROJECT`; line 21 `glassbox-dev-secret` default)
- Test: verification is a grep showing zero Phoenix references in `.env.example` and the corrected Sentry slug names.
- SHARED FILE: `.env.example` is also touched by WS0 (it owns the secrets-only-from-env split) and WS3 (removes the committed pod token default). WS1's edits: drop the Phoenix endpoint line, rename `SENTRY_ORG`/`SENTRY_PROJECT` to the `*_SLUG` names the code reads, and drop the `glassbox-dev-secret` value. Coordinate so WS3's token-default removal does not conflict — if WS3 has already emptied `POD_TOKEN`, skip that part of Step 3a.

Interfaces:
- Consumes: nothing.
- Produces: a `.env.example` whose Sentry slug variable names (`SENTRY_ORG_SLUG`, `SENTRY_PROJECT_SLUG`) match what `backend/config.py` / `backend/sentry_api.py` read; no Phoenix env; no committed secret value.

Steps:

- [ ] Step 1: Establish the failing signal — grep shows the mismatches still present in `.env.example`:
```
grep -nE "PHOENIX|SENTRY_ORG=|SENTRY_PROJECT=|glassbox-dev-secret" .env.example
```
Expected output (the lines we are removing/fixing):
```
.env.example:9:# Arize Phoenix (local, default OK)
.env.example:10:PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
.env.example:21:POD_TOKEN=glassbox-dev-secret  # dev default only; set a strong token in production
.env.example:25:SENTRY_ORG=
.env.example:26:SENTRY_PROJECT=
```

- [ ] Step 2: (No code-unit test for this cleanup — the grep in Step 4 is the verification.)

- [ ] Step 3a: Rewrite `.env.example`. REPLACE the entire file contents with:
```bash
# Copy to .env and fill in. NEVER commit .env.

# Anthropic (Claude auto-interp labels + honesty-judge adjudication)
ANTHROPIC_API_KEY=

# Sentry (incident view) — DSN from your Sentry project
SENTRY_DSN=

# HuggingFace (some Gemma repos are gated; set HF_TOKEN to download weights)
HF_TOKEN=

# Runtime
GLASSBOX_MODE=posthoc          # posthoc | live
CUDA_VISIBLE_DEVICES=0

# GPU pod (optional — local backend proxies /turn and /api/track here)
POD_URL=http://localhost:8001
# POD_TOKEN is the shared pod-auth secret. Generate a strong value, e.g.:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
POD_TOKEN=

# Sentry REST API (optional — populates the recent-issues strip on the Observe tab)
SENTRY_AUTH_TOKEN=
SENTRY_ORG_SLUG=
SENTRY_PROJECT_SLUG=
```

- [ ] Step 3b: (Removed.) `backend/requirements.txt` is owned and deleted by WS0 (Task 18); the `arize-phoenix`/`openinference-instrumentation` removal is moot once the file is gone and must NOT be done here. No edit in this step.

- [ ] Step 4: Run the verification greps on `.env.example` and confirm everything is clean:
```
grep -nE "PHOENIX" .env.example || echo "no phoenix refs"
grep -nE "SENTRY_ORG_SLUG=|SENTRY_PROJECT_SLUG=" .env.example
grep -nE "glassbox-dev-secret" .env.example || echo "no committed pod secret"
```
Expected output:
```
no phoenix refs
.env.example:24:SENTRY_ORG_SLUG=
.env.example:25:SENTRY_PROJECT_SLUG=
no committed pod secret
```

- [ ] Step 5: Commit:
```
git add .env.example
git commit -m "chore(obs): drop Phoenix env from .env.example; fix Sentry org/project slug names; remove committed pod token default"
```

---

### Task 10: Remove `PhoenixPanel`, the coherence-eval button/state, and `runEval()` from the frontend

Files:
- Modify: `frontend/src/api.ts` (lines 43-48, `runEval`)
- Modify: `frontend/src/ObservabilityPage.tsx` (line 7 import, `LatencyHealth` `onEval`/`evalRunning` props + the eval button lines 397-463, the `PhoenixPanel` component lines 569-585, the page-root eval state lines 592-593 + `handleEval` lines 635-647, the `evalResult` render line 699, and the `<PhoenixPanel ...>` usage line 735)
- Modify: `frontend/src/instrument.ts` (copy-only: neutralize "medical tool" + Phoenix comments on lines 2, 14, 19-21)
- Modify: `frontend/src/styles.css` (copy-only: neutralize the Phoenix mention in the link-card CSS comment on line 404 — the Step 4 `grep -rni phoenix frontend/src` scans `.css`)
- Test: full-project `npx tsc --noEmit` from `frontend/` is the gate.

Interfaces:
- Consumes: `ObservabilitySnapshot` WITHOUT `phoenix_ui_url` (Task 8).
- Produces: an `ObservabilityPage` with no Phoenix panel and no coherence-eval button; `api.ts` with no `runEval` export.
- NOTE: depends on Task 8 (the `ObservabilitySnapshot` type no longer has `phoenix_ui_url`). Run after Task 8.

Steps:

- [ ] Step 1: Establish the failing signal — `tsc` over the whole frontend reports the now-dangling `phoenix_ui_url` read and the `runEval` import path. From `frontend/`:
```
cd frontend && npx tsc --noEmit 2>&1 | grep -E "phoenix_ui_url|runEval" || echo "clean"
```
Expected output (BEFORE this task; `phoenix_ui_url` was removed from the type in Task 8 but is still read in `ObservabilityPage.tsx`):
```
src/ObservabilityPage.tsx(735,46): error TS2339: Property 'phoenix_ui_url' does not exist on type 'ObservabilitySnapshot'.
```

- [ ] Step 2a: Edit `frontend/src/api.ts`. DELETE the entire `runEval` export (lines 43-48):
```typescript
export function runEval(): Promise<{ evaluated: number; off_domain: number } | { status: string }> {
  return fetch("/api/observability/eval", { method: "POST" }).then((r) => {
    if (!r.ok) throw new Error(`POST /api/observability/eval -> ${r.status}`);
    return r.json();
  });
}
```

- [ ] Step 2b: Edit `frontend/src/ObservabilityPage.tsx`. REPLACE the import line 7:
```typescript
import { runEval, testSentryAlarm, replaySentryAlarm } from "./api";
```
WITH:
```typescript
import { testSentryAlarm, replaySentryAlarm } from "./api";
```

- [ ] Step 2c: In `ObservabilityPage.tsx`, simplify `LatencyHealth` to drop the eval button. REPLACE the whole `LatencyHealth` component (the function signature with `onEval`/`evalRunning` props through the closing of the `obs-lat-footer` button block) with this version that removes the footer button:
```typescript
function LatencyHealth({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const { latency, health } = snapshot;
  const stageOrder = ["pod_roundtrip", "capture", "sae", "trackers", "label_fetch", "ranking"] as const;
  const presentStages = stageOrder.filter((s) => latency.stages[s]?.p50 != null);
  const maxStageMs = Math.max(...presentStages.map((s) => latency.stages[s]!.p50!), 1);
  const hasTurnLatency = latency.turn_ms.p50 != null;

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Latency</h2>
        {hasTurnLatency ? (
          <span className="right">
            p50 <b>{fmtMs(latency.turn_ms.p50)}</b>
            {" · "}
            p95 <b>{fmtMs(latency.turn_ms.p95)}</b>
            {latency.turn_ms.last != null && <> · last <b>{fmtMs(latency.turn_ms.last)}</b></>}
          </span>
        ) : (
          <span className="sub">percentiles appear after the first turn</span>
        )}
      </div>

      {presentStages.length ? (
        <div className="obs-lat-list">
          {presentStages.map((s) => (
            <LatencyBar key={s} label={humanize(s)} ms={latency.stages[s]!.p50} maxMs={maxStageMs} />
          ))}
        </div>
      ) : (
        <div className="obs-lat-empty">
          <p>Pipeline stage timings (pod roundtrip, SAE encode, probe scoring, label fetch) populate here from the in-process perf ledger.</p>
          <ul>
            <li>Model: <b>{shortModel(health.model)}</b></li>
            <li>Pod: <b className={health.pod_reachable ? "ok" : "warn"}>{health.pod_reachable ? "connected" : "disconnected"}</b></li>
            <li>SAE recon: <b className={health.sae_recon_ok ? "ok" : ""}>{health.sae_recon_cosine?.toFixed(3) ?? "—"}</b></li>
          </ul>
        </div>
      )}
    </section>
  );
}
```

- [ ] Step 2d: In `ObservabilityPage.tsx`, DELETE the entire `PhoenixPanel` component (lines 569-585):
```typescript
function PhoenixPanel({ url }: { url: string }) {
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Phoenix ledger</h2>
        <a className="obs-ext-link" href={url} target="_blank" rel="noreferrer">open ↗</a>
      </div>
      <a className="obs-link-row" href={url} target="_blank" rel="noreferrer">
        <span className="obs-link-text">
          <b>Open trace waterfall &amp; evals</b>
          <span className="obs-link-sub">{url} — span ledger for every cognition turn (no prompt/response)</span>
        </span>
        <span className="obs-link-arrow" aria-hidden="true">↗</span>
      </a>
    </section>
  );
}
```

- [ ] Step 2e: In the `ObservabilityPage` page-root, DELETE the eval state (lines 592-593):
```typescript
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalResult, setEvalResult] = useState<string | null>(null);
```
DELETE the `handleEval` handler (lines 635-647):
```typescript
  async function handleEval() {
    if (evalRunning || !snapshot) return;
    setEvalRunning(true);
    setEvalResult(null);
    try {
      const res = await runEval();
      setEvalResult("status" in res ? res.status : `evaluated ${res.evaluated}, off-domain ${res.off_domain}`);
    } catch {
      setEvalResult("eval failed — check Phoenix is running on :6006");
    } finally {
      setEvalRunning(false);
    }
  }
```
DELETE the `evalResult` render in the head (line 699):
```typescript
        {evalResult && <span className="obs-eval-result">{evalResult}</span>}
```

- [ ] Step 2f: In the page-root JSX, REPLACE the `LatencyHealth` usage (line 723):
```typescript
      <LatencyHealth snapshot={snapshot} onEval={handleEval} evalRunning={evalRunning} />
```
WITH:
```typescript
      <LatencyHealth snapshot={snapshot} />
```
And REPLACE the final grid block (lines 725-736) that wraps `SentryPanel` + `PhoenixPanel`:
```typescript
      <div className="obs-grid">
        <SentryPanel
          snapshot={snapshot}
          onTestAlarm={handleSentryTest}
          onReplayAlarm={handleSentryReplay}
          testRunning={sentryTestRunning}
          testResult={sentryTestResult}
          replayRunning={sentryReplayRunning}
          replayResult={sentryReplayResult}
        />
        <PhoenixPanel url={snapshot.phoenix_ui_url} />
      </div>
```
WITH (drop `PhoenixPanel`; `SentryPanel` no longer shares a grid row):
```typescript
      <SentryPanel
        snapshot={snapshot}
        onTestAlarm={handleSentryTest}
        onReplayAlarm={handleSentryReplay}
        testRunning={sentryTestRunning}
        testResult={sentryTestResult}
        replayRunning={sentryReplayRunning}
        replayResult={sentryReplayResult}
      />
```

- [ ] Step 2g: Neutralize the copy in `frontend/src/instrument.ts` (no behavior change). REPLACE the comment block (lines 1-3):
```typescript
// Sentry must initialize BEFORE any other app code — this file is imported first in main.tsx.
// Mirrors the backend's PHI-careful posture (see backend/fanout.py): replay masks all text,
// and only error sessions are recorded. The DSN comes from .env.local (gitignored).
```
WITH:
```typescript
// Sentry must initialize BEFORE any other app code — this file is imported first in main.tsx.
// Mirrors the backend's privacy posture (see backend/fanout.py): replay masks all text,
// and only error sessions are recorded. The DSN comes from .env.local (gitignored).
```
REPLACE the replay comment (line 14) `      maskAllText: true, // medical tool: never capture raw text into a replay` with `      maskAllText: true, // never capture raw text into a replay`. REPLACE the tracing comment block (lines 19-21):
```typescript
  // Tracing. Propagate trace headers to the same-origin /api calls (Vite proxies to :8000
  // in dev). The backend routes its OWN tracing to Phoenix (traces_sample_rate=0.0), so the
  // browser side shows in Sentry while backend spans live in Phoenix — by design.
```
WITH:
```typescript
  // Tracing. Propagate trace headers to the same-origin /api calls (Vite proxies to :8000
  // in dev). The backend disables its own tracing (traces_sample_rate=0.0); browser-side
  // tracing shows in Sentry.
```
REPLACE the session-replay comment (line 25) `  // Session Replay — PHI-safe: don't record normal sessions, only the ones around an error.` with `  // Session Replay — privacy-safe: don't record normal sessions, only the ones around an error.`.

- [ ] Step 2h: Neutralize the Phoenix mention in `frontend/src/styles.css` (copy-only; the Step 4 grep scans `.css`). REPLACE the comment on line 404 `/* external-tool link cards (Phoenix / Sentry — replace the cramped embedded frames) */` with `/* external-tool link cards (Sentry — replace the cramped embedded frames) */`.

- [ ] Step 3: Run the full-project type check from `frontend/`:
```
cd frontend && npx tsc --noEmit
```
Expected output (clean — exit 0, no output):
```
```

- [ ] Step 4: Confirm Phoenix is gone from the frontend source:
```
grep -rni "phoenix\|runEval\|coherence eval" frontend/src || echo "no phoenix refs in frontend"
```
Expected output:
```
no phoenix refs in frontend
```

- [ ] Step 5: Commit:
```
git add frontend/src/api.ts frontend/src/ObservabilityPage.tsx frontend/src/instrument.ts frontend/src/styles.css
git commit -m "feat(ui): remove PhoenixPanel, coherence-eval button, and runEval(); neutralize instrument + styles copy"
```

---

### Task 11: Neutralize `backend/batch_medqa_observability.py` (drop Phoenix prints + the committed token) and purge Phoenix from README.md

Files:
- Modify: `backend/batch_medqa_observability.py` (lines 1-14 docstring, line 12 `glassbox-dev-secret`, line 84 `--dry-run` help string Phoenix mention, lines 104-108 `init_sponsors()` + sponsors print with the `config.PHOENIX_ENDPOINT` read on line 107, lines 163-165 trailing Phoenix print)
- Modify: `README.md` ONLY (Phoenix mentions + `cognition_event.sample.json` / `CognitionEvent` references). **WS1 does NOT edit `PLAN.md` or `LANES.md`** — see the SHARED FILE note below.
- Test: verification is a grep over `backend` source + `README.md` showing zero Phoenix references in shipping source/docs (the `docs/superpowers/specs/*` and `docs/superpowers/plans/*` design/plan files are allowed to keep historical mentions — they describe the change; `PLAN.md`/`LANES.md` are deferred to WS4 and are NOT in this gate).
- SHARED FILE (cross-plan with WS4): design §3 says WS4 will **Rewrite `README.md`** for the general framing and **Remove/relocate `PLAN.md`, `LANES.md`, `docs/superpowers/*`**, and the dependency graph (design §, "WS1/WS2/WS3 ──▶ WS4") sequences WS4 strictly after WS1. To avoid throwing away effort: WS1 scopes its doc work to `README.md` only (it survives into WS4 as a rewrite target, but must stay coherent and Phoenix-free in the interim), and DEFERS the `PLAN.md`/`LANES.md` Phoenix purge to WS4's deletion of those files. Do not neutralize `PLAN.md`/`LANES.md` here — WS4 deletes them outright.

Interfaces:
- Consumes: `config.SENTRY_DSN` (flat global), `runtime.refresh_pod_health`, `analyze.analyze_turn`, `fanout.fanout`, `fanout.init_sponsors` (all unchanged by WS1 except the Phoenix removal already done in Task 2).
- Produces: a batch script that references only Sentry; no `config.PHOENIX_ENDPOINT` read (that global is deleted in Task 5 — this task removes its last reader).
- NOTE: depends on Task 5 (the `config.PHOENIX_ENDPOINT` global is deleted there; this script is its last reader, so this task must land at or after Task 5 or the module will raise `AttributeError` on import-time use). Run after Task 5.

Steps:

- [ ] Step 1: Establish the failing signal — the batch module still reads the deleted `config.PHOENIX_ENDPOINT` and prints Phoenix. Confirm the reader is present:
```
grep -nE "PHOENIX_ENDPOINT|phoenix|glassbox-dev-secret|6006" backend/batch_medqa_observability.py
```
Expected output:
```
backend/batch_medqa_observability.py:7:  - Optional: Phoenix at PHOENIX_COLLECTOR_ENDPOINT (bash scripts/run_phoenix.sh)
backend/batch_medqa_observability.py:9:  uv pip install arize-phoenix openinference-instrumentation python-dotenv  # if missing
backend/batch_medqa_observability.py:12:  export POD_URL=http://localhost:8001 POD_TOKEN=glassbox-dev-secret
backend/batch_medqa_observability.py:107:        f"phoenix={config.PHOENIX_ENDPOINT}"
backend/batch_medqa_observability.py:165:    print("[batch] done — check Phoenix at http://localhost:6006 and Sentry Issues")
```

- [ ] Step 2: Confirm the module is currently broken at runtime against the post-Task-5 config (the import-time module body is fine, but the `main()` path reads the deleted global). Demonstrate the deleted attribute:
```
uv run python -c "import backend.config as c; print(hasattr(c, 'PHOENIX_ENDPOINT'))"
```
Expected output (after Task 5 landed):
```
False
```
(This proves line 108's `config.PHOENIX_ENDPOINT` would raise `AttributeError` — the reason this task is required.)

- [ ] Step 3a: Edit `backend/batch_medqa_observability.py`. REPLACE the docstring (lines 1-14):
```python
"""Batch MedQA-style prompts through live inference + SAE feature labels, then fan out
to Sentry + Arize Phoenix.

Prereqs:
  - GPU pod reachable (POD_URL) OR local real mode
  - Optional: SENTRY_DSN in .env
  - Optional: Phoenix at PHOENIX_COLLECTOR_ENDPOINT (bash scripts/run_phoenix.sh)

  uv pip install arize-phoenix openinference-instrumentation python-dotenv  # if missing

Run:
  export POD_URL=http://localhost:8001 POD_TOKEN=glassbox-dev-secret
  python -m backend.batch_medqa_observability --limit 5
"""
```
WITH:
```python
"""Batch prompts through live inference + SAE feature labels, then fan out to Sentry.

Prereqs:
  - GPU pod reachable (POD_URL) OR local real mode
  - Optional: SENTRY_DSN in .env

Run:
  export POD_URL=http://localhost:8001 POD_TOKEN=<your-pod-token>
  python -m backend.batch_medqa_observability --limit 5
"""
```

- [ ] Step 3b: REPLACE the sponsors print (lines 104-108):
```python
        init_sponsors()
        print(
            f"[batch] sponsors: sentry={'on' if config.SENTRY_DSN else 'off'} "
            f"phoenix={config.PHOENIX_ENDPOINT}"
        )
```
WITH:
```python
        init_sponsors()
        print(f"[batch] sponsors: sentry={'on' if config.SENTRY_DSN else 'off'}")
```

- [ ] Step 3b-2: REPLACE the `--dry-run` argparse help string (line 84) — it still names Phoenix (the Task 11 Step 4 grep matches it):
```python
    p.add_argument("--dry-run", action="store_true", help="skip fanout to Sentry/Phoenix")
```
WITH:
```python
    p.add_argument("--dry-run", action="store_true", help="skip fanout to Sentry")
```

- [ ] Step 3c: REPLACE the trailing Phoenix print (lines 163-165):
```python
    if not args.dry_run:
        time.sleep(3)  # flush Phoenix OTLP batch exporter
        print("[batch] done — check Phoenix at http://localhost:6006 and Sentry Issues")
```
WITH:
```python
    if not args.dry_run:
        print("[batch] done — check Sentry Issues")
```

- [ ] Step 3d: Purge Phoenix, the medical positioning, and the old fixture/class names from `README.md` (and ONLY `README.md` — `PLAN.md`/`LANES.md` are left for WS4 to delete, per the SHARED FILE note). Apply each of these exact before/after edits. (The grep gate in Step 4 will not pass unless every Phoenix / `cognition_event.sample.json` / `CognitionEvent` reference below is removed or renamed.)

  - **README:3** — REPLACE the tagline line:
```markdown
**Cognition-observability for medical LLMs.** A clinician chats with an open model; GlassBox surfaces the model's *internal state* — an SAE "feature cloud" of what concepts are firing, plus calibrated probes that flag when the model is **internally uncertain but verbally confident** (the confident-wrong zone). Every message emits one `cognition_event` that fans out to Sentry, Arize Phoenix, the UI, and (when flagged) a Claude honesty-judge.
```
  WITH:
```markdown
**Interpretability observability for LLMs.** A user chats with an open model; GlassBox surfaces the model's *internal state* — an SAE "feature cloud" of what concepts are firing, plus calibrated probes that flag when the model is **internally uncertain but verbally confident** (the confident-wrong zone). Every message emits one `introspection_event` that fans out to Sentry, the UI, and (when flagged) a Claude honesty-judge. (A medical Q&A flow is included as one clearly-labeled optional example.)
```

  - **README:18** — in the demo table, REPLACE the **Chat** row:
```markdown
| **Chat** | Medical Q&A with live SAE feature cloud + **harmful** and **over-confidence** probe meters |
```
  WITH:
```markdown
| **Chat** | Live SAE feature cloud + **harmful** and **over-confidence** probe meters (medical Q&A ships as one example) |
```

  - **README:20** — REPLACE the **Observe** row (drop Phoenix):
```markdown
| **Observe** | Phoenix traces, Sentry alarms, KPI strip, probe score trends |
```
  WITH:
```markdown
| **Observe** | Sentry alarms, KPI strip, probe score trends |
```

  - **README:37** — REPLACE the architecture opener (drop "A clinician"):
```markdown
A clinician chats with `unsloth/gemma-3-4b-it` over **POST + NDJSON** (never SSE — it breaks through Cloudflare). On the GPU, the model generates while **one forward hook on `model.model.layers[17]`** captures the residual stream. That single activation feeds **two method families**:
```
  WITH:
```markdown
A user chats with `unsloth/gemma-3-4b-it` over **POST + NDJSON** (never SSE — it breaks through Cloudflare). On the GPU, the model generates while **one forward hook on `model.model.layers[17]`** captures the residual stream. That single activation feeds **two method families**:
```

  - **README:42** — REPLACE the fanout paragraph (drop Phoenix sink + rename event):
```markdown
Downstream on **CPU**, the FastAPI handler assembles **exactly one `cognition_event`** per message and fans it out to four consumers that never touch the GPU: **Sentry** (Issue), **Arize Phoenix** (span + eval), the **chat UI**, and **Claude** (auto-interp labels + async adjudication of flagged events).
```
  WITH:
```markdown
Downstream on **CPU**, the FastAPI handler assembles **exactly one `introspection_event`** per message and fans it out to three consumers that never touch the GPU: **Sentry** (Issue), the **chat UI**, and **Claude** (auto-interp labels + async adjudication of flagged events).
```

  - **README:51-54** — REPLACE the ASCII diagram's event-builder + sinks rows:
```
                     [CPU] build ONE cognition_event
                            │ fanout()
        ┌──────────┬────────┴────────┬──────────────┐
      Sentry    Phoenix          chat UI       Claude judge (async, if flagged)
```
  WITH:
```
                  [CPU] build ONE introspection_event
                            │ fanout()
        ┌─────────────────┬─┴───────────────┐
      Sentry           chat UI       Claude judge (async, if flagged)
```

  - **README:63** — REPLACE the shape-contract bullet (rename class + fixture):
```markdown
1. **Shape contract** — `backend/schema.py` (pydantic `CognitionEvent`) is the source of truth. `frontend/src/types.ts` and `fixtures/cognition_event.sample.json` mirror it. No lane changes the shape without 3-way agreement.
```
  WITH:
```markdown
1. **Shape contract** — `backend/schema.py` (pydantic `IntrospectionEvent`) is the source of truth. `frontend/src/types.ts` and `fixtures/introspection_event.sample.json` mirror it. No lane changes the shape without 3-way agreement.
```

  - **README:64** — REPLACE the wire-protocol bullet (rename trailing event):
```markdown
2. **A↔C wire protocol** — `POST /api/chat` returns `application/x-ndjson`: zero-or-more `{"type":"token",...}` lines, then exactly one `{"type":"event", ...CognitionEvent}`. **Frontend builds fully against `fixtures/` before Backend streams real data.**
```
  WITH:
```markdown
2. **A↔C wire protocol** — `POST /api/chat` returns `application/x-ndjson`: zero-or-more `{"type":"token",...}` lines, then exactly one `{"type":"event", ...IntrospectionEvent}`. **Frontend builds fully against `fixtures/` before Backend streams real data.**
```

  - **README:87-88** — REPLACE the "(Optional) Observability stack" quickstart block (delete the `run_phoenix.sh` launcher line):
```bash
# 4. (Optional) Observability stack
bash scripts/run_phoenix.sh         # Arize Phoenix UI on :6006
```
  WITH (drop the launcher; Sentry is configured via `.env`, no local stack to start):
```bash
# 4. (Optional) Observability — set SENTRY_DSN in .env to forward flagged turns to Sentry.
```

  - **README:126** — REPLACE the `ObservabilityView` gotcha bullet (drop Phoenix linkout):
```markdown
- **`ObservabilityView` is not a custom dashboard** — it's linkout/iframe cards to the live Sentry project + Phoenix (`localhost:6006`).
```
  WITH:
```markdown
- **`ObservabilityView` is not a custom dashboard** — it's linkout cards to the live Sentry project plus the in-process redacted snapshot behind `/api/observability`.
```

- [ ] Step 4: Run the import-and-smoke check plus the doc grep to verify. First confirm the batch module imports cleanly against the post-Task-5 config and that no shipping source / `README.md` keeps Phoenix or the legacy event names. Two exclusions, both deliberate: (1) `PLAN.md`/`LANES.md` are NOT in this grep — they still contain Phoenix prose and are deleted by WS4; including them would wrongly fail the gate. (2) `backend/tests` is excluded from the Phoenix sweep — the guard tests (`test_phoenix_removed.py`, `test_observability_taxonomy.py`, `test_config_sentry_env.py`, `test_observability_endpoint.py::test_observability_endpoint_has_no_phoenix`) legitimately NAME "phoenix"/`PHOENIX_*` to assert it is gone; stale test symbols are covered by Task 12 Step 2 instead.
```
uv run python -c "import backend.batch_medqa_observability"
# Prove the script body no longer READS the deleted config.PHOENIX_ENDPOINT global
# (the import smoke alone only proves module-level code runs; the read lives in main()):
grep -n "PHOENIX_ENDPOINT" backend/batch_medqa_observability.py || echo "no PHOENIX_ENDPOINT read in batch script"
grep -rniE "phoenix|arize|openinference|cognition_event\.sample|CognitionEvent" \
  backend README.md --include='*.py' --include='*.md' --exclude-dir=tests \
  | grep -v "docs/superpowers" || echo "no phoenix/legacy refs in shipping source+README"
```
Expected output:
```
no PHOENIX_ENDPOINT read in batch script
no phoenix/legacy refs in shipping source+README
```

- [ ] Step 5: Commit:
```
git add backend/batch_medqa_observability.py README.md
git commit -m "chore(obs): neutralize batch script (drop Phoenix prints + committed token); purge Phoenix and legacy event-name refs from README"
```

---

### Task 12: Full-suite green gate

Files:
- Test: the entire backend pytest suite + the frontend type check (no new source files; this task only verifies the workstream landed clean).

Interfaces:
- Consumes: everything produced by Tasks 1-11.
- Produces: nothing — a green-suite checkpoint commit-free verification (or a no-op commit if a stray fixup is needed).

Steps:

- [ ] Step 1: Run the entire backend suite (the safety net per design §8):
```
uv run pytest backend/tests -q
```
Expected output (all green; the two deleted Phoenix test files no longer collected, the four new guard tests added):
```
.................................................                        [100%]
... passed in ...s
```

- [ ] Step 2: Confirm no test references a removed symbol or module, EXCLUDING the intentional guard `test_phoenix_removed.py` (Task 1) which legitimately names `backend.coherence_eval` / `backend.phoenix_eval_features` as the modules it asserts are gone:
```
grep -rnE "coherence_eval|phoenix_eval_features|PhoenixSink|capture_cognition_alarm|build_cognition_event|CognitionEvent" backend/tests \
  --exclude=test_phoenix_removed.py || echo "no stale references in tests"
```
Expected output:
```
no stale references in tests
```

- [ ] Step 3: Run the frontend type check end-to-end from `frontend/`:
```
cd frontend && npx tsc --noEmit
```
Expected output (clean — exit 0, no output):
```
```

- [ ] Step 4: Repo-wide final Phoenix sweep (shipping source + frontend; design/plan docs under `docs/superpowers` are allowed to retain historical mentions, and `backend/tests` is excluded because the guard tests legitimately name Phoenix to assert its removal — stale test symbols are covered by Step 2):
```
grep -rniE "phoenix|arize|openinference" backend frontend/src scripts \
  --include='*.py' --include='*.ts' --include='*.tsx' --include='*.sh' --include='*.txt' \
  --exclude-dir=tests \
  || echo "WS1 complete: zero Phoenix references in shipping code"
```
Expected output:
```
WS1 complete: zero Phoenix references in shipping code
```

- [ ] Step 5: If Steps 1-4 are all clean, there is nothing to commit (each prior task committed its own slice). If a stray fixup was needed during this task, commit it:
```
git add -A
git commit -m "test(obs): WS1 green-suite gate — backend pytest + frontend tsc pass with Phoenix removed"
```
(If `git status` is clean, skip the commit — the gate is verification-only.)
