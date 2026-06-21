import torch

from backend.science import concept_synth as cs


def _rows(mean_pos, mean_neg, n=8, d=2560, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        rows.append({"act_resp": torch.randn(d, generator=g) + mean_pos, "label": 1})
        rows.append({"act_resp": torch.randn(d, generator=g) + mean_neg, "label": 0})
    return rows


def test_separable_set_scores_high_auroc():
    rows = _rows(mean_pos=3.0, mean_neg=-3.0)  # cleanly separated
    out = cs.fit_and_validate(rows)
    assert out["status"] == "ok"
    assert out["auroc"] >= 0.9
    assert out["direction"].shape[0] == 2560
    assert out["calibrator"] is not None


def test_single_class_is_insufficient():
    rows = [{"act_resp": torch.randn(2560), "label": 1} for _ in range(8)]
    out = cs.fit_and_validate(rows)
    assert out["status"] == "insufficient_data"
    assert out["direction"] is None


def test_tiny_set_is_insufficient():
    rows = [
        {"act_resp": torch.randn(2560), "label": 1},
        {"act_resp": torch.randn(2560), "label": 0},
    ]
    out = cs.fit_and_validate(rows)
    assert out["status"] == "insufficient_data"
