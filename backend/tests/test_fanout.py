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


def test_sentry_sink_quiet_on_unflagged_and_redacted_on_flag(monkeypatch):
    import backend.fanout as fo
    captured = {}
    fake = type("S", (), {})()
    fake.capture_message = lambda msg, level=None: captured.setdefault("msgs", []).append((msg, level))
    class Scope:
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def set_tag(s, *a): pass
        def set_context(s, k, v): captured["ctx"] = v
        fingerprint = None
    fake.new_scope = lambda: Scope()
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", fake)
    monkeypatch.setattr(fo, "_sentry_on", True)
    sink = fo.SentrySink()
    ev = {"flag": False}; sink.emit(ev); assert "msgs" not in captured        # quiet
    ev = {"flag": True, "severity": "warning", "model": "g", "uncertainty": 0.2,
          "uncertainty_proj": 1.0, "trackers": {}, "io": {"user_msg": "SECRET", "response": "SECRET"},
          "features": [{"label": "dosing"}]}
    sink.emit(ev)
    assert captured["msgs"][0][1] == "warning"
    assert "SECRET" not in repr(captured["ctx"]) and captured["ctx"]["top_features"] == ["dosing"]
