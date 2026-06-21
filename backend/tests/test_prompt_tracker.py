"""A tracker with scores_on='prompt' scores the last-prompt-token activation (harmful intent),
not the response — and vice versa. Guards the prompt/response branch in score_all_trackers."""
import pytest
import torch

from backend.science import persona

_DIR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # unit direction along dim 0


def _vec(x: float) -> torch.Tensor:
    return torch.tensor([x, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)


def test_prompt_tracker_scores_prompt_ignores_response():
    persona.clear_trackers()
    persona.register_tracker("intent", direction=_DIR, threshold=0.5, scores_on="prompt")
    # harmful prompt (proj high), safe response (proj low) -> flags on the PROMPT
    hi = persona.score_all_trackers(_vec(5.0), _vec(-5.0))["intent"]
    # benign prompt (proj low), harmful-looking response (proj high) -> must NOT flag
    lo = persona.score_all_trackers(_vec(-5.0), _vec(5.0))["intent"]
    persona.clear_trackers()
    assert hi["scores_on"] == "prompt"
    assert hi["flag"] is True and hi["score"] > 0.5
    assert lo["flag"] is False and lo["score"] < 0.5


def test_response_tracker_scores_response_ignores_prompt():
    persona.clear_trackers()
    persona.register_tracker("resp", direction=_DIR, threshold=0.5)  # default scores_on="response"
    out = persona.score_all_trackers(_vec(5.0), _vec(-5.0))["resp"]  # harmful prompt, safe response
    persona.clear_trackers()
    assert out["scores_on"] == "response"
    assert out["flag"] is False and out["score"] < 0.5


def test_register_rejects_unknown_scores_on():
    with pytest.raises(ValueError):
        persona.register_tracker("x", direction=_DIR, scores_on="bogus")
