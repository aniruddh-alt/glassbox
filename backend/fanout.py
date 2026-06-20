"""The SINGLE sponsor seam (contract #4). Receives a plain JSON-serializable dict
(tensors already stripped). Adding/removing a sponsor = editing ONLY this file.
NOTHING here imports torch or touches the GPU.

OWNER: Lane A.
"""
from __future__ import annotations

import asyncio

from . import config


def init_sponsors() -> None:
    """Call once at startup."""
    # TODO(Lane A):
    # import sentry_sdk; sentry_sdk.init(dsn=config.SENTRY_DSN)
    # from phoenix.otel import register; global _tracer
    # _tracer = register(endpoint=config.PHOENIX_ENDPOINT, project_name="glassbox").get_tracer(__name__)
    ...


def fanout(event: dict) -> None:
    """Fan one finished cognition_event out to every sponsor consumer (CPU only)."""
    _to_sentry(event)
    _to_phoenix(event)
    if event.get("flag"):
        # async so it NEVER blocks the NDJSON stream; patches adjudication out-of-band.
        asyncio.create_task(_claude_judge(event))
        # _notify_fetch_uagent(event)   # stretch


def _to_sentry(event: dict) -> None:
    # TODO(Lane A): with sentry_sdk.new_scope() as s:
    #   s.fingerprint = ["glassbox", "uncertainty", event["severity"]]   # collapse into one Issue
    #   s.set_context("cognition", event)
    #   sentry_sdk.capture_message("confident-wrong" if event["flag"] else "ok",
    #                              level=event["severity"])
    ...


def _to_phoenix(event: dict) -> None:
    # TODO(Lane A): with _tracer.start_as_current_span("chat-turn", openinference_span_kind="llm") as sp:
    #   sp.set_attribute("input.value", event["io"]["user_msg"])
    #   sp.set_attribute("output.value", event["io"]["response"])
    #   sp.set_attribute("cognition.uncertainty", event["uncertainty"])
    #   sp.set_attribute("cognition.flag", event["flag"])
    ...


async def _claude_judge(event: dict) -> None:
    """Anthropic prize: adjudicate whether a flagged answer is hallucinated; patch the event."""
    # TODO(Lane A/B): anthropic call reusing the trait eval_prompt → Adjudication;
    #   patch into event["adjudication"], push to UI, update Sentry/Phoenix out-of-band.
    ...
