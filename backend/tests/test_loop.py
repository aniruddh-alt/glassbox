# backend/tests/test_loop.py
import torch

from backend.agent import interp_agent
from backend.config import ProbeBuilderConfig
from backend.science import concept_synth as cs
from backend.science import persona


class ScriptedClient:
    """Plays a fixed sequence of tool calls, one per create() invocation."""
    def __init__(self, calls):
        self._calls = list(calls)
        self.messages = self

    def create(self, **kwargs):
        name, tool_input = self._calls.pop(0)
        if name is None:  # end_turn, no tool
            block = type("T", (), {"type": "text", "text": "done"})()
            return type("R", (), {"stop_reason": "end_turn", "content": [block]})()
        use = type("U", (), {"type": "tool_use", "id": "t1", "name": name, "input": tool_input})()
        return type("R", (), {"stop_reason": "tool_use", "content": [use]})()


def _fake_generate(messages, max_new=64):
    pos = "extremely" in messages[0]["content"]
    base = 3.0 if pos else -3.0
    return {"answer": "x", "acts": torch.randn(8, 2560) + base, "out_ids": torch.zeros(8, dtype=torch.long),
            "resp_start": 4, "tok": None}


def test_loop_registers_tracker_when_auroc_passes(monkeypatch):
    # judge_filter keeps everything with its intended label (bypass the real judge call)
    monkeypatch.setattr(cs, "judge_filter",
                        lambda spec, rows, *a, **kw: [{**r, "label": r["intended_label"]} for r in rows])

    spec = {"trait_name": "sycophancy", "definition": "d", "pos_prompt": "You are extremely sycophantic.",
            "neg_prompt": "You are neutral.", "questions": [f"q{i}" for i in range(8)], "judge_rubric": "r"}
    tid = cs.create_job("watch sycophancy")
    client = ScriptedClient([
        ("submit_spec", spec),
        ("generate_contrastive", {}),
        ("judge_filter", {}),
        ("fit_and_validate", {}),
        ("finalize", {"verdict": "Sycophancy is linearly represented."}),
        (None, None),
    ])
    job = interp_agent.run_interp_agent(
        tid, ProbeBuilderConfig(), "", client=client, generate_fn=_fake_generate
    )
    assert job["status"] == "ready"
    assert job["auroc"] >= 0.9
    assert tid in persona._trackers
    assert persona._trackers[tid]["meta"]["reliability"] == "synthetic-validated"
