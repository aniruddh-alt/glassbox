from backend import analyze, runtime
from backend.schema import CognitionEvent


def test_analyze_turn_fallback_builds_valid_event():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
    answer, event, perf = analyze.analyze_turn(
        [{"role": "user", "content": "Is metformin safe during pregnancy?"}]
    )
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(event, CognitionEvent)
    assert event.uncertainty is None
    assert event.flag is False
    assert event.severity == "info"
    assert event.io.user_msg == "Is metformin safe during pregnancy?"
    assert event.io.response == answer
    assert len(event.features) > 0
    assert event.features[0].label
    # round-trips through a fresh validation
    CognitionEvent(**event.model_dump())


def test_real_turn_uses_pod_client_and_ranks(monkeypatch):
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(
        analyze.labels,
        "get_feature_stats",
        lambda i, **k: {"label": f"label-{i}", "max_act": 1.0, "density": 0.001},
    )

    def fake_turn(messages, pod, pod_token, *, max_new):
        return {
            "answer": "pod answer",
            "candidates": [
                {"index": 2, "act": 1.0, "attr": 0.9, "source": "s"},
                {"index": 1, "act": 5.0, "attr": 0.2, "source": "s"},
            ],
            "trackers": {
                "uncertainty": {
                    "score": 0.3,
                    "proj": 0.1,
                    "proj_pre": None,
                    "flag": False,
                    "reliable": True,
                    "status": "ready",
                    "user_defined": False,
                }
            },
            "reliable": True,
        }

    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", fake_turn)
    answer, event, perf = analyze.analyze_turn([{"role": "user", "content": "hi"}])
    assert answer == "pod answer"
    assert [f.index for f in event.features] == [2, 1]
    assert event.features[0].label == "label-2"


def test_real_turn_degrades_on_pod_failure(monkeypatch):
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", lambda *a, **k: (_ for _ in ()).throw(pc.PodError(0, "turn")))
    monkeypatch.setattr(analyze.runtime, "refresh_pod_health", lambda: None)
    answer, event, perf = analyze.analyze_turn([{"role": "user", "content": "Is ibuprofen safe?"}])
    assert isinstance(answer, str) and answer.strip()
    assert len(event.features) > 0


def test_analyze_turn_returns_perf(monkeypatch):
    """analyze_turn must return a 3-tuple (answer, event, perf) with timing fields."""
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(
        analyze.labels,
        "get_feature_stats",
        lambda i, **k: {"label": f"label-{i}", "max_act": 1.0, "density": 0.001},
    )

    def fake_turn_with_timings(messages, pod, pod_token, *, max_new):
        return {
            "answer": "timed answer",
            "candidates": [{"index": 1, "act": 1.0, "attr": 0.5, "source": "s"}],
            "trackers": {},
            "reliable": True,
            "timings": {"capture": 10.0, "sae": 5.0, "trackers": 2.0},
        }

    import backend.pod_client as pc

    monkeypatch.setattr(pc, "turn", fake_turn_with_timings)
    result = analyze.analyze_turn([{"role": "user", "content": "Is aspirin safe?"}])
    assert len(result) == 3, "analyze_turn must return 3-tuple (answer, event, perf)"
    answer, event, perf = result
    assert answer == "timed answer"
    assert isinstance(event, CognitionEvent)
    # perf shape
    assert "turn_ms" in perf and isinstance(perf["turn_ms"], float)
    assert "stages" in perf
    assert "pod_roundtrip" in perf["stages"]
    assert "pod_stages" in perf
    assert perf["pod_stages"].get("capture") == 10.0
    assert perf["pod_stages"].get("sae") == 5.0
    assert perf["pod_stages"].get("trackers") == 2.0


def test_pod_failure_reports_instrument_unhealthy(monkeypatch):
    """Regression (final-review Important #2): a pod failure must wire the instrument_unhealthy
    concern to Sentry via fanout.report_error('pod-down', exc)."""
    from backend import runtime
    import backend.pod_client as pc
    import backend.fanout as fo
    calls = []
    monkeypatch.setattr(fo, "report_error", lambda stage, exc, ctx=None: calls.append(stage))
    monkeypatch.setattr(analyze.runtime, "refresh_pod_health", lambda: None)
    runtime.STATE.update(mode="real", model_loaded=True, sae_loaded=True)
    monkeypatch.setattr(pc, "turn", lambda *a, **k: (_ for _ in ()).throw(pc.PodError(0, "turn")))
    analyze.analyze_turn([{"role": "user", "content": "x"}])
    assert "pod-down" in calls
