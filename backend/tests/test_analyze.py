from backend import analyze, runtime
from backend.schema import CognitionEvent


def test_analyze_turn_fallback_builds_valid_event():
    runtime.STATE.update(mode="fallback", model_loaded=False, sae_loaded=False)
    answer, event = analyze.analyze_turn(
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
