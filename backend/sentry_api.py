"""Sentry REST read path — recent-issues strip.
Lane A — never imports torch. Never raises (returns [] on any failure).
"""
from __future__ import annotations

import httpx

from . import config

_ISSUE_FIELDS = ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink")


def deep_link() -> str | None:
    """Return the Sentry issues deep-link URL when configured, else None."""
    if config.SENTRY_ORG_SLUG and config.SENTRY_AUTH_TOKEN:
        return f"{config.SENTRY_ORG_URL}/organizations/{config.SENTRY_ORG_SLUG}/issues/"
    return None


async def list_recent_issues(limit: int = 15) -> list[dict]:
    """GET recent unresolved issues from Sentry.

    Returns [] when unconfigured (no token/slugs) or on any httpx/JSON error.
    Projects only the fields in _ISSUE_FIELDS (drops extras).
    """
    if not config.SENTRY_AUTH_TOKEN or not config.SENTRY_ORG_SLUG or not config.SENTRY_PROJECT_SLUG:
        return []
    url = f"{config.SENTRY_API_BASE}/api/0/projects/{config.SENTRY_ORG_SLUG}/{config.SENTRY_PROJECT_SLUG}/issues/"
    params = {
        "statsPeriod": "24h",
        "query": "is:unresolved",
        "sort": "date",
        "limit": limit,
    }
    headers = {"Authorization": f"Bearer {config.SENTRY_AUTH_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            issues = r.json()
            return [{k: issue.get(k) for k in _ISSUE_FIELDS} for issue in issues]
    except Exception:  # noqa: BLE001 — never raises to caller
        return []
