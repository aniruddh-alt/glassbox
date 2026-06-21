"""The SINGLE sponsor seam (contract #4). Receives a plain JSON-serializable dict
(the cognition_event, tensors already stripped via CognitionEvent.model_dump()).
Adding/removing a sponsor = editing ONLY this file. NOTHING here imports torch or
touches the GPU.

OWNER: Lane A.

Two observability surfaces, fed by ONE event:
  - Sentry  = the INCIDENT view. A confident-wrong answer becomes a grouped Issue
              (fingerprint collapses repeats), with the cognition payload attached.
  - Phoenix = the ANALYTICS view. Every message becomes one OpenInference LLM span
              you can filter/sort/eval in the dashboard at localhost:6006.
"""

from __future__ import annotations

import os
import json
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

# Belt-and-suspenders: hide inputs/outputs from any openinference auto-instrumentation
# BEFORE phoenix is imported anywhere in this process.
os.environ.setdefault("OPENINFERENCE_HIDE_INPUTS", "true")
os.environ.setdefault("OPENINFERENCE_HIDE_OUTPUTS", "true")

from . import config
from . import observability

_sentry_on = False
_tracer = None  # Phoenix OTEL tracer, set in init_sponsors()


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


# ---------------------------------------------------------------------------
# Task 5: PII scrub + level clamp
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task 6: SentrySink
# ---------------------------------------------------------------------------

def _bucket(u):
    return "high" if (u or 0) >= 0.66 else "med" if (u or 0) >= 0.33 else "low"


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


# ---------------------------------------------------------------------------
# Task 7: PhoenixSink + _stage_spans
# ---------------------------------------------------------------------------

try:
    from opentelemetry.trace import set_span_in_context
except ImportError:  # pragma: no cover
    def set_span_in_context(span, context=None):  # type: ignore[misc]
        return None


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


# ---------------------------------------------------------------------------
# Task 8: StoreSink + _phoenix_is_local + complete init_sponsors
# ---------------------------------------------------------------------------

class StoreSink:
    name = "store"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        observability.STORE.record(observability.to_redacted_view(event), perf)


def _phoenix_is_local() -> bool:
    host = (urlparse(config.PHOENIX_ENDPOINT).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", ""}


def init_sponsors() -> None:
    """Call once at FastAPI startup. Safe to call with missing config — each sponsor
    is independently optional so the app still runs locally without DSN/Phoenix."""
    global _tracer, _sentry_on

    # --- Sentry (incident view) ---
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

    # --- Arize Phoenix (analytics view) — local-only for patient data ---
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


async def _claude_judge(event: dict) -> None:
    """Anthropic prize: adjudicate whether a flagged answer is hallucinated; patch the event.
    Reuse the trait artifact's eval_prompt. Push the verdict to the UI + Sentry/Phoenix
    out-of-band (it arrives a beat after the answer — that's fine, it's a review signal).
    DEFERRED — stub only."""
    # TODO(Lane A/B): anthropic.AsyncAnthropic().messages.create(...) -> Adjudication;
    #   set event["adjudication"], push to UI channel, optionally re-tag the Sentry Issue.
    ...
