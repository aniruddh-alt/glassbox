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


def test_scrub_pii_strips_exception_frame_vars():
    from backend.fanout import _scrub_pii
    ev = {"exception": {"values": [{"stacktrace": {"frames": [
            {"vars": {"messages": "PATIENT_PROMPT", "x": 1}}]}}]},
          "contexts": {}, "extra": {"answer": "RESPONSE_TEXT"}}
    out = _scrub_pii(ev, {})
    assert out["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"] == {}
    assert "RESPONSE_TEXT" not in repr(out)


def test_level_clamps():
    from backend.fanout import _level
    assert _level("warning") == "warning" and _level("info") == "info"
    assert _level("bogus") == "warning"


def test_fanout_isolates_failing_sink(monkeypatch):
    calls = []
    class Good: name = "good";  emit = lambda self, e, p=None: calls.append("good")
    class Bad:  name = "bad";   emit = lambda self, e, p=None: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(fo, "_SINKS", [Bad(), Good()])
    fo.fanout({"flag": False}, {"turn_ms": 1})   # must not raise; Good still runs
    assert calls == ["good"]
