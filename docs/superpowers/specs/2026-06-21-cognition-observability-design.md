# Cognition Observability Platform — Design

**Date:** 2026-06-21
**Status:** Design (v2, hardened by adversarial review) — pending spec review → implementation plan
**Scope (v1):** Wire the two observability sinks (Sentry + Arize Phoenix) end-to-end behind a hard privacy boundary (no prompt/response ever leaves the box), surface both on a new frontend Observability page, and run a Phoenix batch eval over the model's *internal concepts*. Architect the seam as *one event → many sinks + a shared read API* so an MCP server (for Poke) is a later drop-in.

**Deferred (v2+):** Claude adjudicator (privacy-killed; the hallucination probe subsumes it — note its field stays forward-compat but is always `None` in v1), true OTel metric instruments (no Prometheus/Grafana to render them), the MCP server itself.

All external-API claims verified against the installed venv (`arize-phoenix 17.9.0`, `arize-phoenix-evals 3.1.0`, `arize-phoenix-client 2.9.0`, `arize-phoenix-otel 0.16.1`, `sentry-sdk 2.63.0`, `httpx 0.28.1`). NEW/EDIT/REWRITE labels are explicit per component.

---

## 1. Context & the reframe

`glassbox` is a cognition-observability tool for a medical chatbot (`gemma-3-4b-it`, layer 17, gemma-scope-2 SAEs at 16k width). Per turn it produces a tensor-free `CognitionEvent` (`schema.py`) carrying calibrated probe trackers (Family B, reliable), SAE feature labels + activations (Family A, exploratory), an uncertainty flag, and a severity. `fanout.py` is the single sponsor seam, called once per turn *after* the answer has streamed.

**The reframe:** normal observability tracks *what the system did* (latency, errors). Here the cognition **is** the product, so the unit worth surfacing is **divergence between behavior and internal state**. Signals worth tracking are (a) that divergence, (b) the probe/feature signals feeding it, (c) checks that the instrument itself is trustworthy.

**The privacy spine (hard requirement).** Medical data. **The user prompt and model response must NEVER leave the machine** — not to Sentry cloud, not to a cloud LLM judge. What may leave is the **de-identified abstraction**: SAE feature *labels* (global Neuronpedia concept descriptions fetched by feature *index* — not the patient's words; verified that autointerp labels derive from the global corpus, never the live turn), probe *scores*, flags, ids, timings. Cloud is acceptable for these abstracted signals only. Labels reveal *topics*, not identities — the deliberate de-identification line. Phoenix runs **local** (`localhost:6006`); a remote Phoenix collector is forbidden for patient data (§6).

This **corrects existing behavior**: `_to_phoenix` sets `input.value`/`output.value` (`fanout.py:120-121`) and `_to_sentry` attaches `question`/`answer` (`fanout.py:100-103`) — both leak raw text and must be removed.

---

## 2. What goes where (the content contract)

### Sentry = the alarm (only flagged turns; never every turn)

**Decision (resolves a contradiction in v1): `SentrySink.emit` early-returns when `event["flag"]` is False.** Unflagged turns produce **no** Sentry traffic. There is no "ok" Issue. (Phoenix + the store still receive every turn — Sentry is the alarm, Phoenix is the ledger.)

| Concern (taxonomy id) | Trigger | Send call | Fingerprint |
|---|---|---|---|
| **`confident_wrong`** (probe flag) | `event["flag"] is True` (any tracker crossed threshold) | `capture_message(msg, level=_level(severity))` | `["glassbox","medical-cognition","confident_wrong"]` |
| **`instrument_unhealthy`** | pod unreachable / SAE recon bad / label-fetch error | `report_error(stage, exc)` → `capture_exception` | `["glassbox","system", stage]` (`pod-down`, `sae-recon-bad`, `label-fetch`) |

- **Severity → Sentry level:** `severity` is `Literal["info","warning"]` (schema). `_level()` clamps to the valid `LogLevelStr` domain `{fatal,critical,error,warning,info,debug}`, defaulting unknown→`"warning"`. **`report_error` hardcodes `level="error"`** independent of the schema enum (it never reads `event.severity`). Deliberate inversion: a broken instrument (`error`) outranks a cognition `warning` — if the microscope is broken, every flag is suspect.
- **PII-free context** (built explicitly, not auto-attached): `uncertainty`, `uncertainty_proj`, per-tracker scores, feature **labels**, uncertainty bucket, model, `message_id`. **No `question`, no `answer`.**

### Phoenix = the ledger + microscope (every turn, queryable, evaluable)

Every turn → one **redacted** OpenInference trace: parent `chat-turn` (kind `llm`) + **child spans per pipeline stage** (the latency waterfall), carrying cognition attributes but **never any `input.value`/`output.value` or `*.input`/`*.output` attribute**. Then a **post-hoc batch eval** runs *feature-coherence* — "are the activated feature concepts off-domain for a medical assistant?" — from labels + a static domain context only, logging results back as span annotations.

### Shared concern taxonomy (one definition, three surfaces)

`confident_wrong` · `instrument_unhealthy` · `feature_incoherence` (from the eval). Same ids = Sentry fingerprints, Phoenix annotation names, future MCP query categories. Defined once in `observability.py` (`concern_taxonomy()`).

---

## 3. Architecture: one event → many sinks

```
/api/chat turn
  app.py:  turn_start_ns = time.time_ns()          # ABSOLUTE epoch ns — owns the clock origin
           answer, event, perf = analyze_turn(...)  # perf built in analyze.py, see §4
           perf["t0_ns"] = turn_start_ns
           stream tokens + event line to UI          # UNCHANGED, cosmetic
           fanout(event.model_dump(), perf)          # app.py:58, after stream (perf is NEW 2nd arg)
                 │
   fanout.py:  for sink in _SINKS: sink.emit(event, perf)   # per-sink try/except isolation
                 ├─ SentrySink   → if flag: redacted alarm | report_error: sanitized system exc
                 ├─ PhoenixSink  → redacted parent+child-span waterfall (epoch-ns timestamps)
                 └─ StoreSink    → STORE.record(to_redacted_view(event), perf)

GET /api/observability  (page polls ~2s)
   → STORE.snapshot()  +  {"health": runtime.health_payload()}  +  {"sentry": await sentry_api.list_recent_issues()}  +  {"phoenix_ui_url": ...}
POST /api/observability/eval  (or `python -m backend.coherence_eval`)
   → bounded batch: pull recent Phoenix spans (lacking a feature_coherence annotation) → classifier → log annotations

Phoenix iframe → browser loads PHOENIX_UI_URL (localhost:6006)
Sentry strip → REST issues + permalink deep-links (Sentry CANNOT be iframed: X-Frame-Options: deny, verified)
```

The `Sink` protocol + `_SINKS` registry generalize today's hardcoded dispatch. Adding the future `McpPushSink` = registering one more sink.

---

## 4. Components (explicit NEW / EDIT / REWRITE)

### NEW `backend/observability.py` (Lane A, no torch) — canonical store + redaction + read API

Redaction is **allow-list** (rebuild the view from named safe fields) — NOT a deny-list, because raw text is nested (`io.user_msg`, `io.response`) and a flat key-blocklist would miss it. `io` is simply never read.

```python
from collections import deque

def to_redacted_view(event: dict) -> dict:
    """The ONE redaction projector (store + REST + future MCP all go through it).
    ALLOW-LIST: io.* is never referenced → prompt/response cannot leak structurally."""
    t = lambda x: {k: x.get(k) for k in ("score","proj","proj_pre","flag","reliable","user_defined","status")}
    a = event.get("adjudication")
    return {
        "message_id": event["message_id"], "ts": event["ts"],
        "model": event["model"], "layer": event["layer"],
        "uncertainty": event.get("uncertainty"),
        "uncertainty_proj": event.get("uncertainty_proj"),
        "uncertainty_proj_pre": event.get("uncertainty_proj_pre"),
        "flag": event.get("flag", False), "severity": event.get("severity", "info"),
        "trackers": {tid: t(tr) for tid, tr in event.get("trackers", {}).items()},
        "features": [{"index": f["index"], "label": f["label"], "act": f.get("act"),
                      "source": f.get("source"), "tracked": f.get("tracked")}
                     for f in event.get("features", [])],
        # adjudication: forward-compat, ALWAYS None in v1. verdict+by ONLY — never `rationale` (free prose
        # that could quote the answer). See §9.
        "adjudication": ({"verdict": a["verdict"], "by": a.get("by", "claude")} if a else None),
    }

class ObservabilityStore:
    def __init__(self, maxlen: int = 200):
        self._buf: deque[dict] = deque(maxlen=maxlen)        # {ts, view, perf}
    def record(self, view: dict, perf: dict | None) -> None:
        assert "io" not in view, "store must hold only redacted views"   # idempotent guard, §6
        self._buf.append({"ts": view["ts"], "view": view, "perf": perf})
    # read API (REST + future MCP both call these; all returns redacted):
    def snapshot(self) -> dict: ...            # §7
    def get_recent_concerns(self, kind=None, limit=20) -> list[dict]: ...
    def query_turns(self, filter: dict) -> dict: ...
    def get_turn(self, message_id: str) -> dict | None: ...
    def concern_taxonomy(self) -> list[dict]: ...

STORE = ObservabilityStore()
```

`snapshot()` derivations are all **bounded by the `deque(maxlen=200)`** — one point per stored turn, never an independently-growing list: per-tracker `series` (turns where a tracker is absent → skipped/`null`); `uncertainty_series`; `flag_rate`; `confident_wrong` feed (**lists each flagged turn up to maxlen — no fingerprint dedup**, so repeats are visible; dedup is a Sentry concern, not the feed's); `top_features` leaderboard; `latency` (`p50/p95/last` per stage from `perf`); `totals`. **v1 reality:** until calibrated probes are registered, `trackers == {}` (analyze.py:128 / events.py), so `trackers`/series/`confident_wrong` are empty — panels must render empty states (§5).

### NEW `backend/sentry_api.py` (Lane A) — Sentry REST read path (issues strip)

`async def list_recent_issues(limit=15) -> list[dict]` — `GET {SENTRY_API_BASE}/api/0/projects/{org}/{project}/issues/`, `Authorization: Bearer {SENTRY_AUTH_TOKEN}`, params `statsPeriod=24h&query=is:unresolved&sort=date&limit=`. `httpx.AsyncClient(timeout=5.0)`. **Never raises** — `[]` on any failure. Projects `{id, shortId, title, culprit, level, count, userCount, lastSeen, permalink}`. Send-DSN and read-token are two different credentials. The strip treats issue `title`/`culprit` as **tainted** display-only (they originate cloud-side; per §6 the source exceptions are already sanitized) and never re-logs them.

### NEW `backend/coherence_eval.py` (Lane A) — Phoenix batch eval (verified 3.1.0 chain)

```python
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery            # NOT phoenix.trace.dsl (two distinct classes)
from phoenix.evals import create_classifier, evaluate_dataframe
from phoenix.evals.llm import LLM
from phoenix.evals.utils import to_annotation_dataframe

MEDICAL_DOMAIN_CONTEXT = "A clinical medical assistant: expected concepts are clinical, pharmacological, diagnostic, anatomical, procedural."

def run_eval(limit: int = 200) -> dict:
    client = Client(base_url=config.PHOENIX_ENDPOINT)
    df = client.spans.get_spans_dataframe(                  # server-side DSL string; integration-verify (§8)
        query=SpanQuery().where("span_kind == 'LLM'").select("attributes.cognition.feature_labels"),
        project_identifier="glassbox", limit=limit,
    )
    if df.empty: return {"evaluated": 0, "off_domain": 0}
    df = df[df.get("feature_coherence_label").isna()] if "feature_coherence_label" in df else df  # skip already-evaluated
    df["feature_labels"] = df["attributes.cognition.feature_labels"]      # already a JSON string of LABELS
    df["domain_context"] = MEDICAL_DOMAIN_CONTEXT
    df = df[["context.span_id", "feature_labels", "domain_context"]]      # HARD-RESTRICT before the LLM — §6
    assert not df.astype(str).apply(lambda c: c.str.contains("PROMPT_OR_RESPONSE_SENTINEL")).any().any()  # test-time guard
    llm = LLM(provider=config.EVAL_LLM_PROVIDER, model=config.EVAL_LLM_MODEL)   # key from env
    clf = create_classifier(
        name="feature_coherence",
        prompt_template=("You audit a MEDICAL assistant's internal concepts (NOT its words).\n"
                         "Domain: {domain_context}\nActive SAE feature concepts this turn:\n{feature_labels}\n"
                         "Are any OFF-DOMAIN / incongruous for a medical assistant? Answer on_domain or off_domain."),
        llm=llm, choices={"on_domain": 1.0, "off_domain": 0.0})   # single-brace {var}; label→score
    results = evaluate_dataframe(dataframe=df, evaluators=[clf], exit_on_error=False)  # per-row failures don't abort
    annotations = to_annotation_dataframe(dataframe=results)   # already carries annotation_name + annotator_kind cols
    client.spans.log_span_annotations_dataframe(dataframe=annotations)   # df is self-describing; no redundant kwargs
    ...
```

**Privacy-confirmed:** the classifier infers required inputs purely from `{placeholders}` → `['domain_context','feature_labels']`; no `input`/`output` is sent. The df is hard-restricted to those columns before the LLM call. **Execution model:** synchronous-but-bounded (`limit` caps spans; `exit_on_error=False`); a module-level `_RUNNING` lock rejects overlapping runs; `POST /api/observability/eval` runs it in a threadpool and returns `{evaluated, off_domain}`; the frontend button is disabled+spinner while running.

### REWRITE `backend/fanout.py` — sinks + redaction + waterfall + error capture

(Not an edit — the current simple dispatch is replaced by a `Sink` protocol + registry.)

```python
from typing import Protocol, runtime_checkable
@runtime_checkable
class Sink(Protocol):
    name: str
    def emit(self, event: dict, perf: dict | None = None) -> None: ...

_SINKS: list[Sink] = []
def register_sink(s: Sink) -> None: _SINKS.append(s)

def fanout(event: dict, perf: dict | None = None) -> None:   # NEW 2nd param
    for s in _SINKS:
        try: s.emit(event, perf)
        except Exception as e:  # one bad sink never breaks others / the stream
            print(f"[fanout] sink {s.name} failed: {e}")

def report_error(stage: str, exc: BaseException, ctx: dict | None = None) -> None:
    """System-error → Sentry capture_exception, sanitized. Relies on init's locals-off + sanitized exceptions."""
    if not _sentry_on: return
    import sentry_sdk
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["glassbox", "system", stage]
        scope.set_tag("subsystem", stage); scope.set_level("error")
        if ctx: scope.set_context("system", ctx)   # ctx must be PII-free (e.g. {"feature_index": 12})
        sentry_sdk.capture_exception(exc)
```

- **`init_sponsors()`** — **Sentry hardened against frame-locals leak (critical):**
  ```python
  sentry_sdk.init(dsn=config.SENTRY_DSN, traces_sample_rate=0.0, environment="hackathon",
      send_default_pii=False,
      include_local_variables=False,   # CRITICAL: frame locals carry the prompt (pod payload) + event io.*
      include_source_context=False,
      max_request_body_size="never",
      before_send=_scrub_pii)          # defense-in-depth: recursively drop frame .vars + any forbidden-key value
  ```
  **Phoenix redaction** (primary = env vars, documented + version-stable): require `OPENINFERENCE_HIDE_INPUTS=true` / `OPENINFERENCE_HIDE_OUTPUTS=true`; **plus** never set input/output in `PhoenixSink` (the real guarantee); optionally `OITracer(provider.get_tracer(__name__), config=TraceConfig(hide_inputs=True, hide_outputs=True))` as belt-and-suspenders (note: passing `config=` through `register(**kwargs)` works in 0.16.1 but rides an undocumented kwargs path — prefer the env vars or explicit `OITracer`). **Phoenix-local assertion:** if `urlparse(PHOENIX_ENDPOINT).hostname not in {"localhost","127.0.0.1"}`, do **not** register `PhoenixSink` (fail closed). Then `register_sink(...)` for each configured sink.
- **`SentrySink.emit`**: `if not event["flag"]: return` (quiet). Else `new_scope()` (not deprecated `push_scope()`), fingerprint `["glassbox","medical-cognition","confident_wrong"]`, PII-free context (labels + scores only — **no question/answer**), `capture_message(level=_level(event["severity"]))`.
- **`PhoenixSink.emit`**: build the waterfall with **absolute epoch-ns** timestamps (`fanout` runs after the work, so use the synthetic-timestamp form). **Never set `input.value`/`output.value` or any INPUT/OUTPUT attribute** (unit-tested). Span-kind strings must be **lowercase** (`"llm"`,`"tool"`,`"chain"` — uppercase raises `ValueError`); do not pass `OpenInferenceSpanKindValues.X.value`.
  ```python
  from opentelemetry.trace import set_span_in_context
  t0 = perf["t0_ns"]; MS = 1_000_000
  parent = _tracer.start_span("chat-turn", start_time=t0, openinference_span_kind="llm")
  parent.set_attribute("cognition.flag", bool(event["flag"]))
  if event.get("uncertainty") is not None:
      parent.set_attribute("cognition.uncertainty", float(event["uncertainty"]))
  parent.set_attribute("cognition.feature_labels", json.dumps([f["label"] for f in event["features"]]))  # eval input
  for tid, tr in event.get("trackers", {}).items():
      parent.set_attribute(f"cognition.tracker.{tid}", float(tr["score"]))
  ctx = set_span_in_context(parent); off = t0
  for name, kind, ms in _stage_spans(perf):          # canonical stage list, §4-stages
      child = _tracer.start_span(name, context=ctx, start_time=off, openinference_span_kind=kind)
      child.set_attribute("cognition.stage", name); child.end(end_time=off + int(ms * MS)); off += int(ms * MS)
  parent.end(end_time=t0 + int(perf["turn_ms"] * MS))
  ```
- **`StoreSink.emit`**: `observability.STORE.record(observability.to_redacted_view(event), perf)`.
- **`_claude_judge`**: remains a stub; one-line note it is a future *local-model* slot (privacy-killed in cloud).

**Canonical stage list** (single source of truth; matches real code, fixes the v1 line/stage errors):

| stage | span kind | source | measured around |
|---|---|---|---|
| `pod_roundtrip` | tool | orchestration | `pod_client.turn()` (analyze.py:125) |
| `capture` | chain | pod `timings` (subset of pod_roundtrip) | `_capture`→`engine.generate_and_capture` (gpu_service.py:235) — **generation+attribution are one grad-enabled pass; not separable** |
| `sae` | chain | pod `timings` | `_pooled_activations`+`_sae_candidates` (gpu_service.py:240-242) |
| `trackers` | chain | pod `timings` | `_score_trackers` (gpu_service.py:243) |
| `label_fetch` | tool | orchestration | label fetch within `_rank_features` (analyze.py:126) |
| `ranking` | chain | orchestration | rest of `_rank_features` |

`turn_ms` = total. These exact names are reused in §7 `latency.stages` and the PhoenixSink `_stage_spans`.

### EDIT `backend/config.py` — new env vars (existing `os.getenv` pattern; `SENTRY_DSN`:117, `PHOENIX_ENDPOINT`:118; **`AUTOINTERP_MODEL` already exists at :74** — `EVAL_LLM_MODEL` is separate)

```python
PHOENIX_UI_URL     = os.getenv("PHOENIX_UI_URL", PHOENIX_ENDPOINT)
SENTRY_ORG_SLUG    = os.getenv("SENTRY_ORG_SLUG", "")
SENTRY_PROJECT_SLUG= os.getenv("SENTRY_PROJECT_SLUG", "")
SENTRY_AUTH_TOKEN  = os.getenv("SENTRY_AUTH_TOKEN", "")                 # internal-integration, event:read+project:read
SENTRY_API_BASE    = os.getenv("SENTRY_API_BASE", "https://sentry.io").rstrip("/")
SENTRY_ORG_URL     = os.getenv("SENTRY_ORG_URL", "https://sentry.io")  # deep-link host
EVAL_LLM_PROVIDER  = os.getenv("EVAL_LLM_PROVIDER", "anthropic")       # phoenix.evals provider
EVAL_LLM_MODEL     = os.getenv("EVAL_LLM_MODEL", "claude-haiku-4-5-20251001")
```
All 8 are NEW (none currently defined). Deep-link template: `f"{SENTRY_ORG_URL}/organizations/{SENTRY_ORG_SLUG}/issues/"`; per-issue links always use the API-returned `permalink`.

### EDIT `backend/app.py` — timing origin + endpoints + fanout 2nd arg

- **App owns the clock origin.** Capture `turn_start_ns = time.time_ns()` (absolute epoch ns) immediately before `analyze_turn` (app.py:47); after it returns `(answer, event, perf)`, set `perf["t0_ns"] = turn_start_ns`. Pass `perf` to `fanout(payload, perf)` (app.py:58). (`perf_counter()` is monotonic/relative and **must not** be used for span timestamps; per-stage `perf_counter` deltas are durations only, converted to absolute via `t0_ns + cumulative_offset` in PhoenixSink.)
- **NEW `GET /api/observability`** → `STORE.snapshot()` merged with `{"health": runtime.health_payload(), "sentry": await sentry_api.list_recent_issues(), "phoenix_ui_url": config.PHOENIX_UI_URL}` (see §7 for merge precedence).
- **NEW `POST /api/observability/eval`** → `run_in_threadpool(coherence_eval.run_eval)`; returns `{evaluated, off_domain}`; the `_RUNNING` guard returns `{status:"already_running"}` on overlap.

### EDIT `backend/analyze.py` — return `perf` out-of-band (schema frozen). ⚠ Updates the return contract — **all callers change**

`analyze_turn(...) -> (str, CognitionEvent, dict)` (was a 2-tuple). Measure `pod_roundtrip` around `pod_client.turn()` (:125) and `label_fetch`/`ranking` around `_rank_features()` (:126) with `perf_counter` deltas (ms); merge the pod-reported `timings` (`capture`/`sae`/`trackers`). `perf = {"turn_ms": float, "stages": {...orchestration ms...}, "pod_stages": {...pod ms...}}` (`t0_ns` is set by app.py). **`perf` is never written into `CognitionEvent`.** **Callers to update for the 3-tuple:** `app.py:47`, `app.py:69`, `backend/validation/batch_medqa_observability.py`, `backend/tests/test_analyze.py`.

### EDIT `backend/pod_client.py` — thread pod timings + **sanitize `PodError` (privacy)**

`turn()` (:107-112) exposes the additive `timings` key from the pod JSON on its return dict. **`PodError` must carry only a status code + a static stage label — never `r.text`** (the pod body echoes `answer`; today `_post` at :40 embeds up to 200 chars → that would ship the model response to Sentry via `report_error`). Log `r.text` locally (`print`) only. `report_error` then captures a sanitized exception (`f"pod {stage} failed: HTTP {status}"`).

### EDIT `backend/gpu_service.py` — best-effort stage timings (the one Lane-B touch ⚠)

`/turn` (:230-249): add `time.perf_counter()` pairs around the **real** measurable boundaries — `_capture` (:235, generation+attribution combined), `_pooled_activations`+`_sae_candidates` (:240-242), `_score_trackers` (:243) — and return `"timings": {"capture": ms, "sae": ms, "trackers": ms}`. **Additive key**; orchestration degrades to orchestration-only stages if absent. Only change outside Lane A — flagged for coordination; independently mergeable behind the additive-key degradation.

### EDIT `backend/labels.py` — capture genuine exceptions only

At the Neuronpedia GET `except` branch (~:84) and the `_autointerp` Claude `except` (~:195), call `fanout.report_error("label-fetch", e, {"feature_index": index})` **only in real `except` branches with a caught exception** (lazy import to avoid the fanout↔labels cycle). Non-200/empty-explanation fallbacks have no exception → skip reporting (or synthesize a PII-free ctx). Note: `_autointerp`'s Claude call is a **cloud** call that sends **only Neuronpedia global-corpus excerpts**, never the live prompt/response (invariant, §6).

### Frontend (Lane C) — upgrade the planned Observability tab (LANES.md:11,24). All NEW unless noted

| File | NEW/EDIT | Change |
|---|---|---|
| `App.tsx` | EDIT | Add `view: "chat" \| "observe"` + a minimal top-nav toggle (no router dep). **The chat view stays MOUNTED** (hidden via CSS `display:none`) when on Observe — toggling must NOT unmount `ChatPanel`/`useCognitionStream`, abort an in-flight stream, or lose history. |
| `ObservabilityPage.tsx` | NEW | Sections: **TrackerStrip** (probe sparklines, green→red), **ConfidentWrongFeed** (redacted flagged turns — labels+scores, **no Q/A**), **FeatureLeaderboard** (top labels), **LatencyHealth** (per-stage bars + recon cosine + pod badge), **PhoenixEmbed** (`<iframe src=phoenix_ui_url>`), **SentryStrip** (REST issues → `permalink` deep-links; **not** an iframe). |
| `useObservability.ts` | NEW | Polls `GET /api/observability` every 2s; returns `{snapshot, status}`. |
| `api.ts` | EDIT | `getObservability(): Promise<ObservabilitySnapshot>` (relative fetch). |
| `types.ts` | EDIT | `ObservabilitySnapshot` + sub-types. No `timings` on `CognitionEvent` (out-of-band). |
| `styles.css` | EDIT | Reuse tokens (`--panel`, `--accent` red, `--ok` green, `--pad`, `--r`, Hanken, tabular-nums). Hand-rolled SVG sparklines/bars — **no charting dep**. |
| `mock.ts` | EDIT | `DEMO_OBSERVABILITY_SNAPSHOT` fixture. |

No new frontend dependencies (no router, no recharts).

---

## 5. Error handling / degradation

| Condition | Behaviour |
|---|---|
| Sentry DSN absent | `SentrySink` not registered; alarms no-op |
| Sentry token/slugs absent | issues strip → `[]`; page shows "Sentry not configured" + (if `SENTRY_ORG_URL`) a single deep-link |
| Sentry REST slow/rate-limited | 5s timeout → `[]`; snapshot never blocks |
| Phoenix not running / non-local | `PhoenixSink` not registered (fail-closed if remote); iframe shows Phoenix's own error; our panels unaffected |
| Pod fallback/synthetic | `perf` has orchestration stages only; pod stages null; health badge = "fallback" |
| One sink raises | caught per-sink in `fanout`; others + the stream unaffected |
| Empty store / no probes (v1) | `trackers`/series/`confident_wrong` empty → panels render empty states |
| Eval key absent / per-row failure | `exit_on_error=False` records failures in execution-details; `run_eval` returns `{evaluated:0}` on no key; never crashes |
| Eval already running | `_RUNNING` guard → `{status:"already_running"}` |

---

## 6. Privacy invariants (enforced by tests, §8)

1. **No sink payload, store entry, or `/api/observability` response contains `io.user_msg`/`io.response`.** Enforced by: allow-list `to_redacted_view` (io never read); explicit Sentry context (no Q/A) **+ `include_local_variables=False` + `include_source_context=False` + `before_send` stripping frame `.vars`**; `PhoenixSink` never setting input/output attrs + `OPENINFERENCE_HIDE_*` env vars.
2. **`PodError` and any captured exception carry no response/prompt text** — sanitized to status + static stage. `report_error` ctx is PII-free.
3. **Redaction is allow-list and defined once** (`to_redacted_view`); `STORE.record` asserts no `io` present (idempotent guard) so the store can never physically hold raw data regardless of caller. `adjudication.rationale` is never copied (verdict+by only).
4. **Feature labels + probe scores are the only cognition content crossing a network boundary.** Labels derive only from the global Neuronpedia corpus (by index / global activations), never the live turn. `_autointerp` cloud calls send only global-corpus excerpts.
5. **No raw-text upload path is active for patient data.** `NeuronpediaProvider.features_for` POSTs raw turn text to neuronpedia.org (latent leak whenever device resolves to CPU); the live path must assert/guard `LocalSAEProvider` for patient turns. The text-POST provider is forbidden in medical mode.
6. **Phoenix is local.** `PhoenixSink` registers only when `PHOENIX_ENDPOINT` host ∈ {localhost,127.0.0.1} (fail-closed). Remote collectors forbidden for patient data.

---

## 7. `GET /api/observability` response shape (the page contract)

Merge precedence: top-level keys are **snapshot-derived**; `health` is exactly `runtime.health_payload()` verbatim (flat — its own `mode`/`model`/`layer`/`pod_reachable`/`sae_recon_cosine`); `sentry` and `phoenix_ui_url` are appended. No key is duplicated across snapshot and `health`.

```jsonc
{
  "ts": 1750000000.0,
  "totals": { "turns": 24, "flags": 5 },
  "flag_rate": 0.21,
  "uncertainty_series": [0.1, 0.4, 0.71],            // ≤ maxlen, one point/turn
  "trackers": { "uncertainty": { "current": 0.71, "flag_count": 3, "series": [0.1,0.4,0.71] } },  // {} in v1 until probes
  "confident_wrong": [ { "message_id":"m_12","ts":1750.0,"uncertainty":0.2,
                         "trackers":{"hallucination":{"score":0.8,"flag":true}},
                         "feature_labels":["anticoagulant dosing"] } ],   // NO question/answer
  "top_features": [ { "label":"anticoagulant dosing","count":7,"mean_act":1.8 } ],
  "latency": { "turn_ms": {"p50":820,"p95":1400,"last":910},
               "stages": { "pod_roundtrip":{"p50":640}, "capture":{"p50":410}, "sae":{"p50":120},
                           "trackers":{"p50":40}, "label_fetch":{"p50":90}, "ranking":{"p50":20} } },
  "health": { "mode":"real","model_loaded":true,"sae_loaded":true,"model":"...","layer":17,
              "d_sae":16384,"trackers":[],"sae_recon_cosine":0.91,"sae_recon_ok":true,
              "pod_reachable":true,"pod_url_configured":true },     // == runtime.health_payload() verbatim
  "sentry": { "configured":true, "deep_link":"https://sentry.io/organizations/<org>/issues/",
              "issues":[ {"shortId":"GLASSBOX-1","title":"Confident-wrong medical answer",
                          "level":"warning","count":14,"lastSeen":"...","permalink":"..."} ] },
  "phoenix_ui_url": "http://localhost:6006"
}
```

---

## 8. Testing (v1) — all backend tests run WITHOUT torch

- **`observability.py`** — snapshot aggregates (series bounded by maxlen, flag_rate, top_features, latency percentiles, confident_wrong); empty-store/no-probes empty states. **Redaction (allow-list):** feed an event with planted prompt/response + planted `adjudication.rationale`; assert none appear in `to_redacted_view`, `snapshot`, `get_turn`, `query_turns`. **Store guard:** `record` of a raw event raises / strips so the buffer is clean.
- **`fanout.py`** — `_SINKS` dispatch + isolation (raising sink doesn't block others); `SentrySink` early-returns on `flag=False`; flagged path builds context with **no** Q/A; `report_error` fingerprint `["glassbox","system",stage]`, level `error`. **`before_send` strips planted prompt text from an exception frame's `vars`.** `_level` clamps out-of-domain severity. `PhoenixSink` builds parent+child spans with **no `input.value`/`output.value`**, lowercase kinds, epoch-ns timestamps that order correctly.
- **`pod_client.py`** — `PodError` string contains status+stage and **never** the response body; pod `timings` threaded through.
- **`sentry_api.py`** — field projection; `[]` on timeout/non-2xx/unconfigured.
- **`coherence_eval.py`** — monkeypatch `Client`+`LLM`; assert the df sent to `evaluate_dataframe` has exactly `{context.span_id, feature_labels, domain_context}` and **no prompt/response substring**; `exit_on_error=False`; `_RUNNING` guard; already-annotated spans skipped.
- **provider guard** — assert no raw turn text is POSTed to the Neuronpedia activation endpoint on the live path (LocalSAEProvider asserted).
- **`app.py`** — `/api/observability` returns the §7 shape & merge (monkeypatch `STORE.snapshot`+`sentry_api`+`health_payload`); `analyze_turn` 3-tuple threaded into `fanout` with `perf["t0_ns"]` set.
- **Frontend** — render `ObservabilityPage` from `DEMO_OBSERVABILITY_SNAPSHOT`; assert no Q/A text in the confident-wrong feed; toggling to Observe and back preserves chat state (ChatPanel stays mounted).
- **Usefulness probe (end-to-end):** scripted turn through real `fanout` with mocked sinks → (a) a probe flag yields exactly one redacted Sentry alarm, an unflagged turn yields none; (b) the Phoenix span tree has the stage waterfall, no input/output; (c) a deliberately off-domain feature-label set makes `run_eval` return `off_domain`. Report which signals are genuinely informative vs. noise.
- **Integration-only (live Phoenix):** the `SpanQuery` DSL string (`span_kind == 'LLM'`, `select('attributes.cognition.feature_labels')`) is server-evaluated — verify against a running Phoenix, not a unit test.

---

## 9. Out of scope (explicit)

| Item | Why / when |
|---|---|
| Claude adjudicator (`_claude_judge`) | privacy-killed (needs the answer); hallucination probe subsumes it. `adjudication` field is forward-compat **but always `None` in v1** — excluded from the test matrix beyond the rationale-redaction test. |
| True OTel metric instruments | Phoenix renders traces/evals, not OTel metrics; no Prometheus/Grafana. |
| MCP server for Poke | architecture made MCP-ready (`Sink` protocol + store read API + `to_redacted_view`); server is v2. |
| Self-hosted Sentry / full air-gap | cloud OK for abstracted signals (decided). |
| Iframing Sentry | impossible (`X-Frame-Options: deny`); REST + deep-links instead. |
| Independent generation-vs-attribution timing | they share one grad-enabled pass (`generate_and_capture`); measured together as `capture`. |

## 10. Implementation priority — five independently-mergeable workstreams

1. **Privacy + sinks core** *(ship first; fixes the live leak; no FE, no timings)* — `observability.py` (`to_redacted_view` allow-list, `STORE`), REWRITE `fanout.py` (`Sink`/`_SINKS`, hardened Sentry `init`, `report_error`, redacted `SentrySink`/`PhoenixSink`/`StoreSink`, Phoenix-local fail-closed), `PodError` sanitization, provider guard. **Acceptance:** all §6 invariants tested green; no Q/A in any sink; existing turns still stream.
2. **Timings** *(Lane-B `gpu_service` + `pod_client` + `analyze` perf + Phoenix waterfall; gated behind additive-key degradation)* — **Acceptance:** waterfall renders with epoch-ns child spans using the canonical stage names; absent pod timings degrade cleanly.
3. **Read API + `sentry_api` + `GET /api/observability`** — **Acceptance:** §7 shape returned; graceful `[]`/empty states.
4. **Frontend Observability page** — **Acceptance:** panels render from snapshot + live poll; Phoenix iframe + Sentry strip; chat state survives toggle.
5. **Coherence eval** — `coherence_eval.py` + `POST /api/observability/eval` + button. **Acceptance:** labels-only df, off-domain detection, no-duplicate re-runs, never crashes.
6. *(v2)* MCP server as a new sink + read-API wrapper.
