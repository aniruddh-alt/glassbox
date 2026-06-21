# backend/tests/test_observability.py
from backend.observability import to_redacted_view

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
    assert v["trackers"]["hallucination"]["score"] == 0.8

def test_redacted_view_handles_no_adjudication_and_empty_trackers():
    v = to_redacted_view(_event(adjudication=None, trackers={}, features=[]))
    assert v["adjudication"] is None and v["trackers"] == {} and v["features"] == []
