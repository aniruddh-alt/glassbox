"""The SINGLE sponsor seam (contract #4). Receives a plain JSON-serializable dict
(the cognition_event, tensors already stripped via CognitionEvent.model_dump()).
Adding/removing a sponsor = editing ONLY this file. NOTHING here imports torch or
touches the GPU.

OWNER: Lane A.

One redacted event → many interchangeable Sinks:
  - Sentry  = the INCIDENT view. A flagged turn becomes a grouped Issue (fingerprinted by
              which probe fired), with a PII-free cognition payload. Prompt/response are
              NEVER sent by default; SENTRY_SEND_IO is an opt-in (default off), regex-scrubbed.
  - Phoenix = the ANALYTICS view. Every message becomes one OpenInference span (NO input/output)
              with a per-stage latency waterfall, filterable/evaluable at localhost:6006.
  - Store   = the in-process redacted snapshot behind /api/observability (+ future MCP).
"""

from __future__ import annotations

import os
import json
import re
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
# PII handling: structured regex scrub + level clamp + before_send
# ---------------------------------------------------------------------------

_VALID_LEVELS = {"fatal", "critical", "error", "warning", "info", "debug"}
_PII_KEYS = {"question", "answer", "user_msg", "response", "prompt", "messages", "io"}

# Structured-PII patterns scrubbed from any free-text before it leaves the box. Catches
# emails/phones/SSNs — NOT unstructured identifiers like names. For real PHI keep
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
    distinct Issues (a harmful answer and a hallucinated one are triaged differently)."""
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


def capture_cognition_alarm(event: dict, *, flush: bool = False) -> bool:
    """Emit a PII-free Sentry issue when ``event['flag']`` is true.

    Called automatically by ``fanout()`` for every chat turn. Use directly for tests or
    custom hooks. Returns True when a message was queued for Sentry."""
    if not _sentry_on or not event.get("flag"):
        return False
    import sentry_sdk

    reason = _flag_reason(event)
    cognition = {
        "uncertainty": event.get("uncertainty"),
        "uncertainty_proj": event.get("uncertainty_proj"),
        "trackers": _observable_trackers(event.get("trackers")),
        "top_features": [f["label"] for f in event.get("features", [])],
    }
    if config.SENTRY_SEND_IO:
        io = event.get("io") or {}
        cognition["question"] = _scrub(io.get("user_msg", "") or "")
        cognition["answer"] = _scrub(io.get("response", "") or "")
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["glassbox", "medical-cognition", reason]
        scope.set_tag("model", event.get("model"))
        scope.set_tag("event_type", "confident_wrong")
        scope.set_tag("flag_reason", reason)
        scope.set_tag("message_id", event.get("message_id"))
        scope.set_tag("uncertainty_bucket", _bucket(event.get("uncertainty")))
        scope.set_context("cognition", cognition)
        sentry_sdk.capture_message(
            "Confident-wrong medical answer", level=_level(event.get("severity", "warning"))
        )
    if flush:
        sentry_sdk.flush(timeout=3)
    return True


class SentrySink:
    name = "sentry"
    def emit(self, event: dict, perf: dict | None = None) -> None:
        capture_cognition_alarm(event)


# ---------------------------------------------------------------------------
# PhoenixSink — redacted span waterfall (epoch-ns; NEVER input/output)
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
# StoreSink + local-only Phoenix gate + init_sponsors
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
            dsn=config.SENTRY_DSN,
            environment=config.SENTRY_ENVIRONMENT,
            release=config.SENTRY_RELEASE,        # None → Sentry auto-detects git SHA
            traces_sample_rate=0.0,               # tracing goes to Phoenix (OTel), not Sentry perf
            send_default_pii=False,               # medical tool: no IPs / headers / PHI by default
            include_local_variables=False,        # CRITICAL: frame locals carry the prompt + io.*
            include_source_context=False,
            max_request_body_size="never",
            enable_logs=True,                     # stdlib logging → Sentry (sentry-sdk >= 2.35)
            before_send=_scrub_pii,               # whole-event scrub (supersedes main's _scrub_event)
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
        print("[fanout] PHOENIX_ENDPOINT is non-local; PhoenixSink disabled (PII fail-closed).")

    if _sentry_on:
        register_sink(SentrySink())
    register_sink(StoreSink())   # store is always on (in-process, redacted)


async def _claude_judge(event: dict) -> None:
    """Anthropic prize: adjudicate whether a flagged answer is hallucinated; patch the event.
    DEFERRED — privacy-killed in cloud (needs the answer); future LOCAL-model slot only."""
    # TODO(local model): score hallucination from {feature labels, probe scores} only.
    ...
