# backend/observability.py
"""Canonical cognition-observability store + the SINGLE redaction projector.
Lane A — never imports torch. Prompt/response (io.*) are NEVER read here."""
from __future__ import annotations
from collections import deque

_TRACKER_KEYS = ("score", "proj", "proj_pre", "flag", "reliable", "user_defined", "status")

def to_redacted_view(event: dict) -> dict:
    """ALLOW-LIST projection → de-identified view. io.* and adjudication.rationale are never copied,
    so prompt/response cannot leak structurally."""
    a = event.get("adjudication")
    return {
        "message_id": event["message_id"], "ts": event["ts"],
        "model": event.get("model"), "layer": event.get("layer"),
        "uncertainty": event.get("uncertainty"),
        "uncertainty_proj": event.get("uncertainty_proj"),
        "uncertainty_proj_pre": event.get("uncertainty_proj_pre"),
        "flag": event.get("flag", False), "severity": event.get("severity", "info"),
        "trackers": {tid: {k: tr.get(k) for k in _TRACKER_KEYS}
                     for tid, tr in (event.get("trackers") or {}).items()},
        "features": [{"index": f["index"], "label": f["label"], "act": f.get("act"),
                      "source": f.get("source"), "tracked": f.get("tracked")}
                     for f in (event.get("features") or [])],
        "adjudication": ({"verdict": a["verdict"], "by": a.get("by", "claude")} if a else None),
    }


def _pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]

class ObservabilityStore:
    def __init__(self, maxlen: int = 200):
        self._buf: deque[dict] = deque(maxlen=maxlen)

    def record(self, view: dict, perf: dict | None) -> None:
        assert "io" not in view, "store must hold only redacted views (io present)"
        self._buf.append({"ts": view["ts"], "view": view, "perf": perf or {}})

    def snapshot(self) -> dict:
        items = list(self._buf)
        views = [it["view"] for it in items]
        unc = [v["uncertainty"] for v in views if v.get("uncertainty") is not None]
        flags = [v for v in views if v.get("flag")]
        # per-tracker series (bounded by deque; absent → skipped)
        tracker_ids = {tid for v in views for tid in (v.get("trackers") or {})}
        trackers = {}
        for tid in tracker_ids:
            series = [v["trackers"][tid]["score"] for v in views if tid in (v.get("trackers") or {})]
            trackers[tid] = {"current": series[-1] if series else None,
                             "flag_count": sum(1 for v in views if (v.get("trackers") or {}).get(tid, {}).get("flag")),
                             "series": series}
        # feature leaderboard
        feat = {}
        for v in views:
            for f in v.get("features") or []:
                d = feat.setdefault(f["label"], {"label": f["label"], "count": 0, "_acts": []})
                d["count"] += 1
                if f.get("act") is not None: d["_acts"].append(f["act"])
        top = sorted(feat.values(), key=lambda d: -d["count"])
        for d in top:
            d["mean_act"] = round(sum(d["_acts"]) / len(d["_acts"]), 4) if d["_acts"] else None
            d.pop("_acts")
        # latency percentiles per stage
        def stage_vals():
            out = {}
            for it in items:
                for name, ms in (it["perf"].get("stages", {}) | it["perf"].get("pod_stages", {})).items():
                    out.setdefault(name, []).append(ms)
            return out
        turn_ms = [it["perf"].get("turn_ms") for it in items if it["perf"].get("turn_ms") is not None]
        stages = stage_vals()
        latency = {"turn_ms": {"p50": _pct(turn_ms, 50), "p95": _pct(turn_ms, 95),
                               "last": turn_ms[-1] if turn_ms else None},
                   "stages": {n: {"p50": _pct(vs, 50)} for n, vs in stages.items()}}
        cw = [{"message_id": v["message_id"], "ts": v["ts"], "uncertainty": v.get("uncertainty"),
               "trackers": v.get("trackers", {}),
               "feature_labels": [f["label"] for f in v.get("features") or []]} for v in flags]
        return {"ts": views[-1]["ts"] if views else None,
                "totals": {"turns": len(views), "flags": len(flags)},
                "flag_rate": (len(flags) / len(views)) if views else 0.0,
                "uncertainty_series": unc, "trackers": trackers,
                "top_features": top, "latency": latency, "confident_wrong": cw}

    def get_recent_concerns(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        cw = self.snapshot()["confident_wrong"][-limit:]
        return cw if kind in (None, "confident_wrong") else []

    def get_turn(self, message_id: str) -> dict | None:
        return next((it["view"] for it in reversed(self._buf) if it["view"]["message_id"] == message_id), None)

    def query_turns(self, filter: dict) -> dict:
        out = [it["view"] for it in self._buf]
        if filter.get("flagged"): out = [v for v in out if v.get("flag")]
        if filter.get("min_uncertainty") is not None:
            out = [v for v in out if (v.get("uncertainty") or 0) >= filter["min_uncertainty"]]
        lim = filter.get("limit", 50)
        return {"turns": out[-lim:], "next_cursor": None}

    def concern_taxonomy(self) -> list[dict]:
        return [{"id": "confident_wrong", "description": "low-uncertainty flagged answer", "source": "tracker"},
                {"id": "instrument_unhealthy", "description": "pod/SAE/label failure", "source": "system"},
                {"id": "feature_incoherence", "description": "off-domain feature activation", "source": "phoenix-eval"}]

STORE = ObservabilityStore()
