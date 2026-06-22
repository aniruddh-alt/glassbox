"""Tests for GET /api/observability (Task 15).

Verifies the merged-shape contract from §7 of the design spec.
Does NOT test POST /api/observability/eval (coherence_eval doesn't exist yet).
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend import observability, sentry_api

client = TestClient(app)


def _empty_snapshot():
    return {
        "ts": None,
        "totals": {"turns": 0, "flags": 0},
        "flag_rate": 0.0,
        "uncertainty_series": [],
        "trackers": {},
        "top_features": [],
        "latency": {"turn_ms": {"p50": None, "p95": None, "last": None}, "stages": {}},
        "confident_wrong": [],
    }


def test_observability_endpoint_shape(monkeypatch):
    """GET /api/observability returns the §7 merged shape with health, sentry, phoenix_ui_url."""
    monkeypatch.setattr(observability.STORE, "snapshot", lambda: _empty_snapshot())
    # list_recent_issues is async; monkeypatch with a coroutine function
    async def _no_issues(sentry, token, limit=15):
        return []
    monkeypatch.setattr(sentry_api, "list_recent_issues", _no_issues)

    r = client.get("/api/observability")
    assert r.status_code == 200
    j = r.json()

    # merged keys present
    assert "health" in j
    assert "sentry" in j
    assert "phoenix_ui_url" in j

    # snapshot keys present
    assert "totals" in j
    assert "trackers" in j
    assert "confident_wrong" in j

    # sentry sub-shape
    sentry = j["sentry"]
    assert "configured" in sentry
    assert "deep_link" in sentry
    assert "issues" in sentry
    assert isinstance(sentry["issues"], list)

    # health sub-shape — must match runtime.health_payload() fields
    health = j["health"]
    assert {"mode", "model", "layer", "trackers", "pod_reachable"} <= set(health)

    # phoenix_ui_url is a string
    assert isinstance(j["phoenix_ui_url"], str)


def test_observability_endpoint_no_pii(monkeypatch):
    """The /api/observability response must contain no prompt or response text."""
    snap = _empty_snapshot()
    snap["confident_wrong"] = [
        {
            "message_id": "m1",
            "ts": 1.0,
            "uncertainty": 0.9,
            "trackers": {},
            "feature_labels": ["anticoagulant dosing"],
        }
    ]
    monkeypatch.setattr(observability.STORE, "snapshot", lambda: snap)
    async def _no_issues(sentry, token, limit=15):
        return []
    monkeypatch.setattr(sentry_api, "list_recent_issues", _no_issues)

    r = client.get("/api/observability")
    assert r.status_code == 200
    text = r.text
    # none of the PII marker strings should appear
    assert "user_msg" not in text
    assert "io" not in text or '"io"' not in text  # "io" may appear in other words


def test_observability_endpoint_sentry_issues_forwarded(monkeypatch):
    """Issues returned by sentry_api are forwarded in the sentry.issues list."""
    monkeypatch.setattr(observability.STORE, "snapshot", lambda: _empty_snapshot())
    async def _with_issues(sentry, token, limit=15):
        return [{"shortId": "G-1", "title": "Confident-wrong medical answer", "level": "warning",
                 "count": 3, "lastSeen": "2026-06-21", "permalink": "https://sentry.io/issues/1"}]
    monkeypatch.setattr(sentry_api, "list_recent_issues", _with_issues)

    r = client.get("/api/observability")
    assert r.status_code == 200
    j = r.json()
    assert j["sentry"]["issues"][0]["shortId"] == "G-1"
