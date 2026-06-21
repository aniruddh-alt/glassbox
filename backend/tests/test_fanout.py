import backend.fanout as fo


def test_fanout_null_uncertainty_does_not_crash(monkeypatch):
    monkeypatch.setattr(fo, "_SINKS", [])
    ev = {
        "flag": False,
        "severity": "info",
        "uncertainty": None,
        "model": "m",
        "trackers": {},
        "features": [{"label": "pregnancy"}],
        "io": {"user_msg": "q", "response": "a"},
    }
    fo.fanout(ev)


def test_fanout_isolates_failing_sink(monkeypatch):
    calls = []
    class Good: name = "good";  emit = lambda self, e, p=None: calls.append("good")
    class Bad:  name = "bad";   emit = lambda self, e, p=None: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(fo, "_SINKS", [Bad(), Good()])
    fo.fanout({"flag": False}, {"turn_ms": 1})   # must not raise; Good still runs
    assert calls == ["good"]
