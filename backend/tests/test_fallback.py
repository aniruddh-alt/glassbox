from backend import fallback


def test_synth_turn_is_deterministic():
    msgs = [{"role": "user", "content": "Is ibuprofen safe in the third trimester?"}]
    a1, f1 = fallback.synth_turn(msgs)
    a2, f2 = fallback.synth_turn(msgs)
    assert a1 == a2
    assert f1 == f2
    assert isinstance(a1, str) and a1.strip()
    assert len(f1) >= 5
    assert all(f["label"] for f in f1)
    assert all(f["source"] == "17-gemmascope-2-res-16k" for f in f1)


def test_synth_turn_varies_by_prompt():
    _, fa = fallback.synth_turn([{"role": "user", "content": "AAA"}])
    _, fb = fallback.synth_turn([{"role": "user", "content": "BBB"}])
    assert fa != fb
