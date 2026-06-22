"""Sentry REST read path — recent-issues strip.
Lane A — never imports torch. Never raises (returns [] on any failure).
"""
from __future__ import annotations

import httpx

_ISSUE_FIELDS = ("id", "shortId", "title", "culprit", "level", "count", "userCount", "lastSeen", "permalink")


def deep_link(sentry, token: str) -> str | None:
    """Return the Sentry issues deep-link URL when configured, else None."""
    if sentry.org_slug and token:
        return f"{sentry.org_url}/organizations/{sentry.org_slug}/issues/"
    return None


async def list_recent_issues(sentry, token: str, limit: int = 15) -> list[dict]:
    """GET recent unresolved issues from Sentry. Returns [] when unconfigured or on any error."""
    if not token or not sentry.org_slug or not sentry.project_slug:
        return []
    url = f"{sentry.api_base}/api/0/projects/{sentry.org_slug}/{sentry.project_slug}/issues/"
    params = {"statsPeriod": "24h", "query": "is:unresolved", "sort": "date", "limit": limit}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            issues = r.json()
            return [{k: issue.get(k) for k in _ISSUE_FIELDS} for issue in issues]
    except Exception:  # noqa: BLE001 — never raises to caller
        return []
