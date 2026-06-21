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

from . import config

_tracer = None  # Phoenix OTEL tracer, set in init_sponsors()
_sentry_on = False


def init_sponsors() -> None:
    """Call once at FastAPI startup. Safe to call with missing config — each sponsor
    is independently optional so the app still runs locally without DSN/Phoenix."""
    global _tracer, _sentry_on

    # --- Sentry (incident view) ---
    if config.SENTRY_DSN:
        import sentry_sdk

        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            traces_sample_rate=0.0,  # we emit our own spans to Phoenix, not Sentry perf
            environment="hackathon",
            send_default_pii=False,
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


def _bucket(u: float) -> str:
    return "high" if u >= 0.66 else "med" if u >= 0.33 else "low"


def _to_sentry(event: dict) -> None:
    """One Sentry Issue per failure mode. fingerprint groups all confident-wrong events
    into a single trackable Issue; the cognition payload rides along as context."""
    if not _sentry_on:
        return
    import sentry_sdk

    event_type = "confident_wrong" if event["flag"] else "ok"
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["glassbox", "medical-cognition", event_type]
        scope.set_tag("model", event["model"])
        scope.set_tag("event_type", event_type)
        scope.set_tag("uncertainty_bucket", _bucket(event["uncertainty"]))
        scope.set_context(
            "cognition",
            {
                "uncertainty": event["uncertainty"],
                "uncertainty_proj": event["uncertainty_proj"],
                "trackers": event["trackers"],
                "top_features": [f["label"] for f in event["features"]],
                "question": event["io"]["user_msg"],
                "answer": event["io"]["response"],
            },
        )
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
