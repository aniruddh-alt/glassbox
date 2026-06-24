"""Tests for backend/sentry_api.py — field projection + graceful failure."""
import asyncio

import backend.sentry_api as sa
from backend.config import SentryConfig


def _sentry(org="org", proj="proj", base="https://sentry.io", org_url="https://sentry.io"):
    return SentryConfig(org_slug=org, project_slug=proj, api_base=base, org_url=org_url)


def test_list_recent_issues_projects_fields(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{
                "id": "1", "shortId": "G-1", "title": "t", "culprit": "c", "level": "warning",
                "count": 5, "userCount": 2, "lastSeen": "x", "permalink": "p", "extra": "drop",
            }]

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
    out = asyncio.run(sa.list_recent_issues(_sentry(), "tok"))
    assert len(out) == 1
    assert out[0]["shortId"] == "G-1"
    assert "extra" not in out[0]
    for field in ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink"):
        assert field in out[0]


def test_list_recent_issues_empty_when_unconfigured():
    assert asyncio.run(sa.list_recent_issues(_sentry(), "")) == []


def test_list_recent_issues_empty_when_org_slug_missing():
    assert asyncio.run(sa.list_recent_issues(_sentry(org=""), "tok")) == []


def test_list_recent_issues_empty_on_timeout(monkeypatch):
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
    assert asyncio.run(sa.list_recent_issues(_sentry(), "tok")) == []


def test_list_recent_issues_empty_on_non_2xx(monkeypatch):
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
    assert asyncio.run(sa.list_recent_issues(_sentry(), "tok")) == []


def test_deep_link_when_configured():
    link = sa.deep_link(_sentry(org="myorg", org_url="https://sentry.io"), "tok")
    assert link == "https://sentry.io/organizations/myorg/issues/"


def test_deep_link_none_when_unconfigured():
    assert sa.deep_link(_sentry(), "") is None
