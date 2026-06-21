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


def test_store_sink_records_redacted(monkeypatch):
    import backend.fanout as fo
    from backend import observability
    store = observability.ObservabilityStore()
    monkeypatch.setattr(observability, "STORE", store)
    fo.StoreSink().emit({"message_id": "m", "ts": 1.0, "io": {"user_msg": "SECRET"},
                         "features": [], "trackers": {}}, {"turn_ms": 5})
    assert store.snapshot()["totals"]["turns"] == 1
    assert "SECRET" not in repr(store.snapshot())


def test_phoenix_not_registered_when_remote(monkeypatch):
    import backend.fanout as fo
    monkeypatch.setattr(fo.config, "PHOENIX_ENDPOINT", "https://cloud.phoenix.example.com")
    assert fo._phoenix_is_local() is False


def test_phoenix_sink_redacts_and_builds_waterfall(monkeypatch):
    import backend.fanout as fo
    spans = []
    class FakeSpan:
        def __init__(s, name): s.name = name; s.attrs = {}; s.ended = None
        def set_attribute(s, k, v): s.attrs[k] = v
        def end(s, end_time=None): s.ended = end_time
    class FakeTracer:
        def start_span(s, name, context=None, start_time=None, openinference_span_kind=None):
            assert openinference_span_kind == openinference_span_kind.lower()  # lowercase contract
            sp = FakeSpan(name); sp.start = start_time; sp.kind = openinference_span_kind; spans.append(sp); return sp
    monkeypatch.setattr(fo, "_tracer", FakeTracer())
    monkeypatch.setattr(fo, "set_span_in_context", lambda sp: None, raising=False)
    ev = {"flag": True, "model": "g", "uncertainty": 0.2, "trackers": {"unc": {"score": 0.2}},
          "features": [{"label": "dosing"}], "io": {"user_msg": "SECRET", "response": "SECRET"}}
    perf = {"t0_ns": 1_000_000_000, "turn_ms": 100,
            "stages": {"pod_roundtrip": 60, "label_fetch": 10, "ranking": 5},
            "pod_stages": {"capture": 40, "sae": 15, "trackers": 3}}
    fo.PhoenixSink().emit(ev, perf)
    all_attrs = {k: v for sp in spans for k, v in sp.attrs.items()}
    assert "input.value" not in all_attrs and "output.value" not in all_attrs
    assert "SECRET" not in repr(all_attrs)
    assert spans[0].name == "chat-turn" and any(sp.name == "pod_roundtrip" for sp in spans)
    assert all_attrs["cognition.feature_labels"] == '["dosing"]'


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


def test_scrub_pii_walks_whole_event_not_just_three_sections():
    """Regression (final-review Important #1): _scrub_pii must scrub forbidden keys ANYWHERE
    in the event (breadcrumbs/threads/logentry), not only contexts/extra/request."""
    from backend.fanout import _scrub_pii
    ev = {
        "message": "Confident-wrong medical answer",  # key 'message' is NOT a PII key -> preserved
        "breadcrumbs": {"values": [{"data": {"messages": "PATIENT_SECRET_Q"}}]},
        "threads": {"values": [{"stacktrace": {"frames": [{"vars": {"response": "ANSWER_SECRET"}}]}}]},
    }
    out = _scrub_pii(ev, {})
    blob = repr(out)
    assert "PATIENT_SECRET_Q" not in blob and "ANSWER_SECRET" not in blob
    assert out["message"] == "Confident-wrong medical answer"
