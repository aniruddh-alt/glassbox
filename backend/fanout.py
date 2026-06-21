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

import asyncio
import json
import re

from . import config

_tracer = None  # Phoenix OTEL tracer, set in init_sponsors()
_sentry_on = False

# Structured-PII patterns scrubbed from any cognition free-text before it leaves the box.
_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[phone]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
]


def _scrub(s: str) -> str:
    for pat, repl in _PII:
        s = pat.sub(repl, s)
    return s


def _scrub_event(event, _hint):
    """Sentry before_send hook (defense-in-depth PHI redaction). Catches emails/phones/
    SSNs in the cognition free-text — NOT unstructured identifiers like names. For real
    PHI, gate the raw text off entirely (SENTRY_SEND_IO=0) or de-identify upstream."""
    cog = (event.get("contexts") or {}).get("cognition")
    if isinstance(cog, dict):
        for k in ("question", "answer"):
            if isinstance(cog.get(k), str):
                cog[k] = _scrub(cog[k])
    return event


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
            release=config.SENTRY_RELEASE,  # None → auto-detect git SHA
            traces_sample_rate=0.0,  # tracing goes to Phoenix (OTel), not Sentry perf
            send_default_pii=False,  # medical tool: no IPs / headers / PHI by default
            enable_logs=True,  # stdlib logging → Sentry (sentry-sdk >= 2.35)
            before_send=_scrub_event,  # defense-in-depth PHI redaction
        )
        _sentry_on = True

    # --- Arize Phoenix (analytics view) ---
    try:
        from phoenix.otel import register

        # register() reads PHOENIX_COLLECTOR_ENDPOINT; if spans don't appear, point it at
        # the OTLP traces path explicitly: endpoint="http://localhost:6006/v1/traces".
        provider = register(
            project_name="glassbox",
            endpoint=f"{config.PHOENIX_ENDPOINT.rstrip('/')}/v1/traces",
            batch=True,
            auto_instrument=False,  # we instrument manually, one span per message
        )
        _tracer = provider.get_tracer(__name__)
    except Exception as e:  # noqa: BLE001 - never let a missing sidecar break the app
        print(f"[fanout] Phoenix not initialized ({e}); continuing without it.")


def fanout(event: dict) -> None:
    """Fan one finished cognition_event out to every sponsor consumer (CPU only).
    Call AFTER the event line has been streamed to the UI, so nothing here blocks tokens."""
    _to_sentry(event)
    _to_phoenix(event)
    if event.get("flag"):
        # Anthropic prize: adjudicate async so it NEVER blocks the stream.
        try:
            asyncio.get_running_loop()
            asyncio.create_task(_claude_judge(event))
        except RuntimeError:
            pass  # no event loop (e.g. called from a sync context) — skip live adjudication
        # _notify_fetch_uagent(event)   # stretch


def _bucket(u: float | None) -> str:
    if u is None:
        return "unknown"
    return "high" if u >= 0.66 else "med" if u >= 0.33 else "low"


def _flag_reason(event: dict) -> str:
    """Sentry grouping key = which reliable probe fired. Distinct failure modes become
    distinct Issues (a harmful answer and a hallucinated one are triaged differently)."""
    trackers = event.get("trackers", {})
    for tid in ("harmful", "hallucination"):  # priority order — most severe first
        t = trackers.get(tid)
        if t and t.get("flag"):
            return tid
    return "confident_wrong" if event.get("flag") else "ok"


def _to_sentry(event: dict) -> None:
    """One Sentry Issue per failure mode. fingerprint groups repeats of the SAME failure
    mode into a single trackable Issue; the cognition payload rides along as context."""
    if not _sentry_on:
        return
    import sentry_sdk

    reason = _flag_reason(event)
    cognition = {
        "uncertainty": event["uncertainty"],
        "uncertainty_proj": event["uncertainty_proj"],
        "trackers": event["trackers"],
        "top_features": [f["label"] for f in event["features"]],
    }
    if config.SENTRY_SEND_IO:  # PHI gate (scrubbed again in before_send)
        cognition["question"] = event["io"]["user_msg"]
        cognition["answer"] = event["io"]["response"]

    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["glassbox", "medical-cognition", reason]
        scope.set_tag("model", event["model"])
        scope.set_tag("event_type", "confident_wrong" if event["flag"] else "ok")
        scope.set_tag("flag_reason", reason)
        scope.set_tag("uncertainty_bucket", _bucket(event["uncertainty"]))
        scope.set_context("cognition", cognition)
        # Only flagged answers should surface as alerts; "ok" turns into a quiet info Issue.
        sentry_sdk.capture_message(
            "Confident-wrong medical answer" if event["flag"] else "cognition ok",
            level=event["severity"],  # "warning" | "info"
        )


def _to_phoenix(event: dict) -> None:
    """Every message → one OpenInference LLM span with the cognition signals as attributes.
    Span attributes must be primitives (str/bool/int/float) — dump nested data to JSON."""
    if _tracer is None:
        return
    with _tracer.start_as_current_span(
        "chat-turn", openinference_span_kind="llm"
    ) as span:
        span.set_attribute("input.value", event["io"]["user_msg"])
        span.set_attribute("output.value", event["io"]["response"])
        if event["uncertainty"] is not None:
            span.set_attribute("cognition.uncertainty", float(event["uncertainty"]))
        span.set_attribute("cognition.flag", bool(event["flag"]))
        span.set_attribute("cognition.severity", event["severity"])
        span.set_attribute(
            "cognition.top_features",
            json.dumps([f["label"] for f in event["features"]]),
        )
        for tid, t in event["trackers"].items():
            span.set_attribute(f"cognition.tracker.{tid}", float(t["score"]))


async def _claude_judge(event: dict) -> None:
    """Anthropic prize: adjudicate whether a flagged answer is hallucinated; patch the event.
    Reuse the trait artifact's eval_prompt. Push the verdict to the UI + Sentry/Phoenix
    out-of-band (it arrives a beat after the answer — that's fine, it's a review signal)."""
    # TODO(Lane A/B): anthropic.AsyncAnthropic().messages.create(...) -> Adjudication;
    #   set event["adjudication"], push to UI channel, optionally re-tag the Sentry Issue.
    ...
