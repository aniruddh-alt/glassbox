from backend import fanout


def test_fanout_null_uncertainty_does_not_crash(monkeypatch):
    monkeypatch.setattr(fanout, "_sentry_on", False)
    monkeypatch.setattr(fanout, "_tracer", None)
    ev = {
        "flag": False,
        "severity": "info",
        "uncertainty": None,
        "model": "m",
        "trackers": {},
        "features": [{"label": "pregnancy"}],
        "io": {"user_msg": "q", "response": "a"},
    }
    fanout.fanout(ev)
