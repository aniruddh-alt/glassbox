# backend/tests/test_observability.py
from backend.observability import to_redacted_view, ObservabilityStore

SECRET_Q, SECRET_A = "PATIENT_PROMPT_42yo_warfarin", "RESPONSE_take_5mg"

def _event(**over):
    e = {"message_id": "m1", "ts": 1.0, "model": "gemma", "layer": 17,
         "io": {"user_msg": SECRET_Q, "response": SECRET_A},
         "uncertainty": 0.2, "uncertainty_proj": 1.1, "uncertainty_proj_pre": 0.9,
         "flag": True, "severity": "warning",
         "trackers": {"hallucination": {"score": 0.8, "proj": 2.0, "proj_pre": 1.0,
                                        "flag": True, "reliable": True, "user_defined": False, "status": "ready"}},
         "features": [{"index": 12, "label": "anticoagulant dosing", "act": 1.8,
                       "source": "17-gemmascope-2-res-16k", "caveat": "x", "tracked": None}],
         "adjudication": {"verdict": "likely_hallucinated", "rationale": SECRET_A, "by": "claude"}}
    e.update(over); return e

def test_redacted_view_omits_prompt_response_and_rationale():
    v = to_redacted_view(_event())
    blob = repr(v)
    assert SECRET_Q not in blob and SECRET_A not in blob
    assert "io" not in v
    assert v["adjudication"] == {"verdict": "likely_hallucinated", "by": "claude"}  # no rationale
    assert v["features"][0]["label"] == "anticoagulant dosing"
    assert "hallucination" not in v["trackers"]

def test_redacted_view_handles_no_adjudication_and_empty_trackers():
    v = to_redacted_view(_event(adjudication=None, trackers={}, features=[]))
    assert v["adjudication"] is None and v["trackers"] == {} and v["features"] == []

def test_store_rejects_raw_event_and_snapshot_aggregates():
    s = ObservabilityStore(maxlen=5)
    raw = {"io": {"user_msg": "x"}, "message_id": "m", "ts": 1.0}
    try:
        s.record(raw, None); assert False, "should reject un-redacted view"
    except AssertionError as e:
        assert "redacted" in str(e)
    # well-formed redacted views
    for i in range(3):
        v = to_redacted_view({"message_id": f"m{i}", "ts": float(i), "model": "g", "layer": 17,
                              "uncertainty": 0.1 * i, "flag": i == 2, "severity": "warning",
                              "trackers": {"over_confidence": {"score": 0.1 * i, "flag": i == 2}},
                              "features": [{"index": 1, "label": "dosing", "act": 1.0}]})
        s.record(v, {"turn_ms": 100 + i, "stages": {"pod_roundtrip": 50}})
    snap = s.snapshot()
    assert snap["totals"]["turns"] == 3
    assert len(snap["uncertainty_series"]) == 3
    assert abs(snap["flag_rate"] - 1/3) < 1e-9
    assert snap["trackers"]["over_confidence"]["series"] == [0.0, 0.1, 0.2]
    assert snap["top_features"][0]["label"] == "dosing" and snap["top_features"][0]["count"] == 3
    assert snap["latency"]["turn_ms"]["last"] == 102
    assert len(snap["confident_wrong"]) == 1 and snap["confident_wrong"][0]["message_id"] == "m2"

def test_store_empty_snapshot_is_well_formed():
    snap = ObservabilityStore().snapshot()
    assert snap["totals"]["turns"] == 0 and snap["trackers"] == {} and snap["confident_wrong"] == []
