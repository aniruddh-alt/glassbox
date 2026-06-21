import torch

from backend.science import concept_synth as cs

SPEC = {
    "pos_prompt": "You are extremely sycophantic.",
    "neg_prompt": "You are neutral and direct.",
    "questions": ["Is my treatment plan good?", "Should I worry?"],
}


def _fake_generate(messages, max_new=64):
    # acts shape [seq, d_in]; resp_start splits prompt vs response.
    return {
        "answer": "ok " + messages[0]["content"][:8],
        "acts": torch.ones(10, 2560),
        "out_ids": torch.zeros(10, dtype=torch.long),
        "resp_start": 6,
        "tok": None,
    }


def test_generate_contrastive_emits_two_rows_per_question():
    rows = cs.generate_contrastive(SPEC, generate_fn=_fake_generate)
    assert len(rows) == 4  # 2 questions x 2 sides
    assert sum(r["intended_label"] for r in rows) == 2
    assert all(r["act_resp"].shape[0] == 2560 for r in rows)
    assert all(r["act_last"].shape[0] == 2560 for r in rows)


def test_generate_contrastive_disables_attribution():
    captured = []

    def spy(messages, max_new=64, attribution=None):
        captured.append(attribution)
        return _fake_generate(messages, max_new)

    import backend.engine as engine

    original = engine.generate_and_capture
    engine.generate_and_capture = spy
    try:
        cs.generate_contrastive(SPEC)
    finally:
        engine.generate_and_capture = original
    assert captured and all(a is False for a in captured)


def test_generate_prepends_system_into_user_turn():
    captured = []

    def spy(messages, max_new=64):
        captured.append(messages)
        return _fake_generate(messages, max_new)

    cs.generate_contrastive(SPEC, generate_fn=spy)
    # gemma has no system role — instruction must be inside the user content
    assert all(m[0]["role"] == "user" for m in captured)
    assert SPEC["pos_prompt"] in captured[0][0]["content"]
