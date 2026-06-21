"""Attribution ranking: the kernel math (sae.attribution_topk), the analyze re-rank branch,
and the feature_provider candidate selection. The torch-dependent kernel uses a tiny fake SAE
so the formula act_{f,p} · (grad_p · W_dec[f]) is checked exactly, with no model load."""

import torch

from backend import analyze
from backend.science import feature_provider as fp
from backend.science import sae as sae_mod


class _FakeSAE:
    """Linear stand-in: encode(a) = a @ W_enc; exposes W_dec and parameters() for device lookup."""

    def __init__(self, W_enc, W_dec):
        self.W_enc = W_enc
        self.W_dec = W_dec

    def encode(self, a):
        return a @ self.W_enc

    def parameters(self):
        return iter([self.W_enc])


def _install_fake_sae(monkeypatch, W_enc, W_dec):
    monkeypatch.setattr(sae_mod, "_sae", _FakeSAE(W_enc, W_dec))


def test_attribution_topk_matches_hand_computed(monkeypatch):
    # d_in=2, d_sae=2, identity encode + identity decoder -> feats == acts, gd == grad
    eye = torch.eye(2)
    _install_fake_sae(monkeypatch, W_enc=eye, W_dec=eye)
    acts = torch.tensor([[2.0, 0.0], [0.0, 3.0]])   # pos0 fires feature0, pos1 fires feature1
    grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # attr[f] = sum_p feats[p,f] * (grad[p] · W_dec[f]) -> feature0=2, feature1=3
    out = sae_mod.attribution_topk(acts, grad, keep=[0, 1], cap=10)
    assert [c["index"] for c in out] == [1, 0]       # ranked by attribution desc
    assert out[0]["attr"] == 3.0 and out[0]["act"] == 3.0
    assert out[1]["attr"] == 2.0 and out[1]["act"] == 2.0
    assert all(c["source"] == sae_mod.config.NP_SOURCE for c in out)


def test_attribution_topk_respects_keep_and_drops_nonpositive(monkeypatch):
    eye = torch.eye(2)
    _install_fake_sae(monkeypatch, W_enc=eye, W_dec=eye)
    acts = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # keep only position 0 -> only feature0 has positive attribution; feature1 (attr 0) is dropped
    out = sae_mod.attribution_topk(acts, grad, keep=[0], cap=10)
    assert [c["index"] for c in out] == [0]
    assert out[0]["attr"] == 2.0


def test_attribution_topk_empty_keep_returns_empty(monkeypatch):
    eye = torch.eye(2)
    _install_fake_sae(monkeypatch, W_enc=eye, W_dec=eye)
    acts = torch.zeros((2, 2))
    grad = torch.zeros((2, 2))
    assert sae_mod.attribution_topk(acts, grad, keep=[], cap=5) == []


def test_feature_attribution_vector(monkeypatch):
    eye = torch.eye(2)
    _install_fake_sae(monkeypatch, W_enc=eye, W_dec=eye)
    acts = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attr, act_max = sae_mod.feature_attribution(acts, grad, keep=[0, 1])
    assert attr.tolist() == [2.0, 3.0]
    assert act_max.tolist() == [2.0, 3.0]


def test_attribution_topk_contrastive_baseline_cancels_always_on(monkeypatch):
    # feature0 has high raw attribution (2) but is "always-on" (baseline 2.5) -> cancels to -0.5
    # and is dropped; feature1 (attr 3, baseline 0) survives. This is the discourse-feature fix.
    eye = torch.eye(2)
    _install_fake_sae(monkeypatch, W_enc=eye, W_dec=eye)
    acts = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    baseline = torch.tensor([2.5, 0.0])
    out = sae_mod.attribution_topk(acts, grad, keep=[0, 1], cap=10, baseline=baseline)
    assert [c["index"] for c in out] == [1]          # always-on feature0 removed
    assert out[0]["attr"] == 3.0


def test_rank_features_uses_attribution_when_present(monkeypatch):
    monkeypatch.setattr(
        analyze.labels, "get_feature_stats",
        lambda i, **k: {"label": f"label-{i}", "max_act": 1.0, "density": 0.001},
    )
    candidates = [
        {"index": 1, "act": 5.0, "attr": 0.2, "source": "s"},
        {"index": 2, "act": 9.0, "attr": 0.9, "source": "s"},
        {"index": 3, "act": 1.0, "attr": 0.5, "source": "s"},
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [2, 3, 1]   # by attribution, NOT activation
    assert ranked[0]["label"] == "label-2"
    assert ranked[0]["caveat"] and ranked[0]["tracked"] is None
    assert "attr" not in ranked[0]                      # stays internal; not in the event schema


def test_rank_features_attribution_keeps_unlabeled(monkeypatch):
    # the attribution path must NOT drop unlabeled features (the old DROP_UNLABELED footgun)
    stats = {
        1: {"label": "pregnancy", "max_act": 1.0, "density": 0.001},
        2: {"label": "feature 2", "max_act": None, "density": None},  # unlabeled
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [
        {"index": 1, "act": 5.0, "attr": 0.3, "source": "s"},
        {"index": 2, "act": 5.0, "attr": 0.8, "source": "s"},  # unlabeled but highest attribution
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [2, 1]   # unlabeled high-attr feature still surfaces


def test_attribution_demotes_structural_below_concept(monkeypatch):
    monkeypatch.setattr(analyze.config, "STRUCTURAL_PENALTY", 0.15)
    stats = {
        1: {"label": "paragraph breaks", "max_act": 1.0, "density": 0.1, "is_structural": True},
        2: {"label": "pregnancy and childbirth", "max_act": 1.0, "density": 0.001, "is_structural": False},
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    # structural feature has HIGHER raw attribution but must sink below the concept after the penalty
    candidates = [
        {"index": 1, "act": 5.0, "attr": 1.0, "source": "s"},  # 1.0 * 0.15 = 0.15
        {"index": 2, "act": 5.0, "attr": 0.5, "source": "s"},  # 0.5 * 1.0  = 0.5
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [2, 1]


def test_attribution_structural_survives_if_dominant(monkeypatch):
    # demote, NOT drop: a structural feature that overwhelmingly dominates the answer still appears
    monkeypatch.setattr(analyze.config, "STRUCTURAL_PENALTY", 0.15)
    stats = {
        1: {"label": "paragraph breaks", "max_act": 1.0, "density": 0.1, "is_structural": True},
        2: {"label": "pregnancy", "max_act": 1.0, "density": 0.001, "is_structural": False},
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [
        {"index": 1, "act": 5.0, "attr": 100.0, "source": "s"},  # 100 * 0.15 = 15 > 0.5
        {"index": 2, "act": 5.0, "attr": 0.5, "source": "s"},
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [1, 2]
    assert len(ranked) == 2  # nothing dropped


def test_attribution_structural_detected_via_label_marker(monkeypatch):
    # no is_structural flag (e.g. a Neuronpedia-labelled feature) — the label markers still flag it
    monkeypatch.setattr(analyze.config, "STRUCTURAL_PENALTY", 0.1)
    stats = {
        1: {"label": "newline / paragraph break token", "max_act": 1.0, "density": 0.1},
        2: {"label": "pregnancy", "max_act": 1.0, "density": 0.001},
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [
        {"index": 1, "act": 5.0, "attr": 1.0, "source": "s"},
        {"index": 2, "act": 5.0, "attr": 0.5, "source": "s"},
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [2, 1]  # "newline"/"paragraph" marker -> demoted


def test_rank_features_attribution_truncates_to_topk_event(monkeypatch):
    monkeypatch.setattr(
        analyze.labels, "get_feature_stats", lambda i, **k: {"label": f"l{i}", "max_act": 1.0, "density": 0.01}
    )
    candidates = [{"index": i, "act": 1.0, "attr": float(i), "source": "s"} for i in range(80)]
    ranked = analyze._rank_features(candidates)
    assert len(ranked) == analyze.config.TOPK_EVENT
    assert ranked[0]["index"] == 79   # highest attribution first


def test_provider_uses_attribution_branch_with_grad(monkeypatch):
    captured = {}

    def fake_attr(activations, grad, keep, cap=50, baseline=None):
        captured["keep"] = keep
        captured["cap"] = cap
        return [{"index": 100 + p, "act": 1.0, "attr": float(p), "source": "s"} for p in keep]

    monkeypatch.setattr(fp, "attribution_topk", fake_attr)
    monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")
    prov = fp.LocalSAEProvider()
    out = prov.features_for(
        "txt", activations=_Shape(4), token_ids=[1, 999, 2, 3], special_ids={999},
        grad=object(), cap=50,
    )
    assert captured["keep"] == [0, 2, 3]            # special position 1 (token 999) excluded
    assert {c["index"] for c in out} == {100, 102, 103}
    assert all("attr" in c for c in out)


def test_provider_preamble_skip_filters_keep(monkeypatch):
    captured = {}

    def fake_attr(activations, grad, keep, cap=50, baseline=None):
        captured["keep"] = keep
        return [{"index": 100 + p, "act": 1.0, "attr": float(p), "source": "s"} for p in keep]

    monkeypatch.setattr(fp, "attribution_topk", fake_attr)
    monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")
    monkeypatch.setattr(fp.config, "PREAMBLE_SKIP", 2)
    fp.LocalSAEProvider().features_for(
        "txt", activations=_Shape(6), token_ids=[1, 2, 3, 4, 5, 6], grad=object(), cap=50
    )
    assert captured["keep"] == [2, 3, 4, 5]   # first 2 (preamble) response positions dropped


def test_provider_preamble_skip_falls_back_when_too_large(monkeypatch):
    captured = {}

    def fake_attr(activations, grad, keep, cap=50, baseline=None):
        captured["keep"] = keep
        return []

    monkeypatch.setattr(fp, "attribution_topk", fake_attr)
    monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")
    monkeypatch.setattr(fp.config, "PREAMBLE_SKIP", 100)  # larger than the whole response
    fp.LocalSAEProvider().features_for(
        "txt", activations=_Shape(4), token_ids=[1, 2, 3, 4], grad=object(), cap=50
    )
    assert captured["keep"] == [0, 1, 2, 3]   # over-skip -> fall back to full response


def test_provider_falls_back_to_activation_without_grad(monkeypatch):
    def fake_topk(act, k=15):
        return [{"index": 100 + act[1], "act": 1.0, "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    # no grad -> activation path; attribution_topk must NOT be called
    monkeypatch.setattr(fp, "attribution_topk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    prov = fp.LocalSAEProvider()
    out = prov.features_for("txt", activations=_RowShape(3), token_ids=[1, 2, 3], grad=None)
    assert {c["index"] for c in out} == {100, 101, 102}
    assert all("attr" not in c for c in out)


def test_provider_falls_back_when_attribution_raises(monkeypatch):
    def fake_topk(act, k=15):
        return [{"index": 200 + act[1], "act": 1.0, "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    monkeypatch.setattr(fp, "attribution_topk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(fp.config, "RANK_METHOD", "attribution")
    prov = fp.LocalSAEProvider()
    out = prov.features_for("txt", activations=_RowShape(2), token_ids=[1, 2], grad=object())
    assert {c["index"] for c in out} == {200, 201}   # degraded to activation path, no crash


class _Shape:
    """Activations stand-in that only needs .shape (provider attribution branch is monkeypatched)."""

    def __init__(self, n):
        self.shape = (n, 4)


class _RowShape(_Shape):
    """Adds row indexing for the activation-path fallback (sae_topk reads act[1] = position)."""

    def __getitem__(self, pos):
        return ("row", pos)
