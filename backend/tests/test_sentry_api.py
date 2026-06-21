"""Tests for backend/sentry_api.py — field projection + graceful failure."""
import asyncio

import backend.sentry_api as sa


def test_list_recent_issues_projects_fields(monkeypatch):
    """Returns projected dicts with only the allowed fields (drops extras)."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "id": "1",
                    "shortId": "G-1",
                    "title": "t",
                    "culprit": "c",
                    "level": "warning",
                    "count": 5,
                    "userCount": 2,
                    "lastSeen": "x",
                    "permalink": "p",
                    "extra": "drop",
                }
            ]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(sa.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "tok")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "org")
    monkeypatch.setattr(sa.config, "SENTRY_PROJECT_SLUG", "proj")

    out = asyncio.run(sa.list_recent_issues())
    assert len(out) == 1
    assert out[0]["shortId"] == "G-1"
    assert "extra" not in out[0]
    # all projected fields present
    for field in ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink"):
        assert field in out[0]


def test_list_recent_issues_empty_when_unconfigured(monkeypatch):
    """Returns [] immediately when token is missing (no HTTP request)."""
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "")
    assert asyncio.run(sa.list_recent_issues()) == []


def test_list_recent_issues_empty_when_org_slug_missing(monkeypatch):
    """Returns [] immediately when org slug is missing."""
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "tok")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "")
    monkeypatch.setattr(sa.config, "SENTRY_PROJECT_SLUG", "proj")
    assert asyncio.run(sa.list_recent_issues()) == []


def test_list_recent_issues_empty_on_timeout(monkeypatch):
    """Returns [] on any httpx exception (e.g. timeout)."""

    class TimeoutClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            raise sa.httpx.TimeoutException("timeout")

    monkeypatch.setattr(sa.httpx, "AsyncClient", TimeoutClient)
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "tok")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "org")
    monkeypatch.setattr(sa.config, "SENTRY_PROJECT_SLUG", "proj")

    assert asyncio.run(sa.list_recent_issues()) == []


def test_list_recent_issues_empty_on_non_2xx(monkeypatch):
    """Returns [] on non-2xx HTTP status."""

    class BadResponse:
        def raise_for_status(self):
            raise sa.httpx.HTTPStatusError("403", request=None, response=None)

        def json(self):
            return []

    class BadClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return BadResponse()

    monkeypatch.setattr(sa.httpx, "AsyncClient", BadClient)
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "tok")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "org")
    monkeypatch.setattr(sa.config, "SENTRY_PROJECT_SLUG", "proj")

    assert asyncio.run(sa.list_recent_issues()) == []


def test_deep_link_when_configured(monkeypatch):
    """deep_link() returns the issues URL when token + org slug are set."""
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "tok")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_SLUG", "myorg")
    monkeypatch.setattr(sa.config, "SENTRY_ORG_URL", "https://sentry.io")
    link = sa.deep_link()
    assert link == "https://sentry.io/organizations/myorg/issues/"


def test_deep_link_none_when_unconfigured(monkeypatch):
    """deep_link() returns None when token is absent."""
    monkeypatch.setattr(sa.config, "SENTRY_AUTH_TOKEN", "")
    assert sa.deep_link() is None
