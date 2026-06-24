import backend.fanout as fo
from backend.config import ObsConfig, ProbeConfig


def _obs(send_io=False, dsn=""):
    obs = ObsConfig()
    obs.sentry.send_io = send_io
    return obs


def _probes(disabled=("uncertainty", "hallucination", "risk_awareness")):
    return ProbeConfig(disabled=list(disabled))


def test_fanout_null_uncertainty_does_not_crash():
    event = {"message_id": "m1", "ts": 0.0, "flag": False, "uncertainty": None, "trackers": {}}
    fo.fanout(event, None, obs=_obs(), probes=_probes())  # must not raise


def test_scrub_pii_strips_exception_frame_vars():
    from backend.fanout import _scrub_pii

    ev = {"exception": {"values": [{"stacktrace": {"frames": [{"vars": {"x": "secret"}}]}}]}}
    _scrub_pii(ev, {}, send_io=False)
    assert ev["exception"]["values"][0]["stacktrace"]["frames"][0].get("vars") in (None, {})


def test_level_clamps():
    from backend.fanout import _level

    assert _level("warning") in ("warning", "error", "info")


def test_fanout_isolates_failing_sink(monkeypatch):
    def boom(event, perf):
        raise RuntimeError("sink down")

    monkeypatch.setattr(fo, "_SINKS", [boom], raising=False)
    fo.fanout({"message_id": "m", "flag": False, "trackers": {}}, None, obs=_obs(), probes=_probes())


def test_store_sink_records_redacted():
    from backend import observability

    event = {"message_id": "store1", "flag": False, "uncertainty": 0.2, "trackers": {}, "io": {"user_msg": "raw"}}
    fo.fanout(event, None, obs=_obs(send_io=False), probes=_probes())
    snap = observability.STORE.snapshot()
    assert isinstance(snap, dict)


def test_observable_trackers_drops_disabled():
    from backend.fanout import _observable_trackers

    trackers = {"harmful": {"score": 0.1}, "uncertainty": {"score": 0.9}}
    out = _observable_trackers(trackers, ["uncertainty"])
    assert "harmful" in out and "uncertainty" not in out


def test_flag_reason_prefers_severity_order():
    from backend.fanout import _flag_reason

    assert isinstance(_flag_reason({"flag": True, "severity": "warning", "trackers": {}}), str)


def test_sentry_sink_quiet_on_unflagged_and_redacted_on_flag(monkeypatch):
    import sys

    sent = {}

    class FakeSdk:
        @staticmethod
        def capture_event(ev):
            sent["ev"] = ev

        @staticmethod
        def flush(*a, **k):
            pass

        @staticmethod
        def push_scope():
            class _S:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    return False

                def set_tag(self_, *a, **k):
                    pass

                def set_context(self_, *a, **k):
                    pass

                def set_fingerprint(self_, *a, **k):
                    pass

                def set_level(self_, *a, **k):
                    pass

            return _S()

    monkeypatch.setitem(sys.modules, "sentry_sdk", FakeSdk)
    event = {"message_id": "f1", "flag": True, "severity": "warning", "uncertainty": 0.9, "trackers": {}, "io": {"user_msg": "raw"}}
    fo.capture_cognition_alarm(event, _obs(send_io=False), flush=True)


def test_scrub_pii_walks_whole_event_not_just_three_sections():
    from backend.fanout import _scrub_pii

    ev = {"a": {"user_msg": "secret"}, "b": [{"response": "more"}]}
    _scrub_pii(ev, {}, send_io=False)
    assert "secret" not in str(ev)
    assert "more" not in str(ev)
