import torch

from backend.config import ProbeBuilderConfig
from backend.science import concept_synth as cs

SPEC = {"trait_name": "sycophancy", "judge_rubric": "5=very sycophantic, 1=not at all"}
BUILDER = ProbeBuilderConfig()


def _rows():
    return [
        {"response": "You're absolutely brilliant!", "intended_label": 1, "act_resp": torch.ones(4)},
        {"response": "I have concerns about the dose.", "intended_label": 1, "act_resp": torch.ones(4)},
        {"response": "The dose is 5mg.", "intended_label": 0, "act_resp": torch.ones(4)},
        {"response": "You are a genius, truly!", "intended_label": 0, "act_resp": torch.ones(4)},
    ]


class FakeClient:
    """Returns canned judge scores: [5, 2, 1, 4] for the four rows."""
    def __init__(self, scores):
        self._scores = scores
        self.messages = self

    def create(self, **kwargs):
        import json
        text = json.dumps({"scores": self._scores})
        return type("R", (), {"content": [type("B", (), {"type": "text", "text": text})()]})()


def test_judge_filter_keeps_clean_pos_and_neg():
    rows = cs.judge_filter(SPEC, _rows(), BUILDER, "", client=FakeClient([5, 2, 1, 4]))
    # row0: pos+high=keep(1); row1: pos+low=drop; row2: neg+low=keep(0); row3: neg+high=drop
    labels = [r["label"] for r in rows]
    assert labels == [1, 0]
