from backend.events import build_cognition_event


def test_empty_trackers_yield_null_uncertainty():
    ev = build_cognition_event(
        message_id="m1", ts=1.0, user_msg="q", response="a",
        trackers={}, features=[], model="unsloth/gemma-3-4b-it", layer=17,
    )
    assert ev.uncertainty is None
    assert ev.uncertainty_proj is None
    assert ev.uncertainty_proj_pre is None
    assert ev.flag is False
    assert ev.severity == "info"
    assert ev.trackers == {}
    assert ev.model == "unsloth/gemma-3-4b-it"
    assert ev.layer == 17


def test_uncertainty_tracker_populates_meter():
    trackers = {
        "uncertainty": {
            "score": 0.83, "proj": 1.27, "proj_pre": 0.91, "flag": True,
            "reliable": True, "status": "ready", "user_defined": False,
        }
    }
    ev = build_cognition_event(
        message_id="m2", ts=1.0, user_msg="q", response="a",
        trackers=trackers, features=[], model="m", layer=17,
    )
    assert ev.uncertainty == 0.83
    assert ev.uncertainty_proj == 1.27
    assert ev.flag is True
    assert ev.severity == "warning"
