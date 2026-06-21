from backend import analyze


def test_rank_prefers_relative_engagement(monkeypatch):
    # isolate the REL ranking: disable the density filter so both features survive
    monkeypatch.setattr(analyze.config, "DENSITY_MAX", 1.0)
    # #1: huge raw act but fires at half its ceiling (grammatical-style)
    # #2: smaller raw act but near its ceiling (specific / clinical)
    stats = {
        1: {"label": "it's contractions", "max_act": 10000.0, "density": 0.02},
        2: {"label": "pregnancy and pregnant status", "max_act": 2100.0, "density": 0.001},
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [
        {"index": 1, "act": 5000.0, "source": "s"},  # rel 0.50
        {"index": 2, "act": 2000.0, "source": "s"},  # rel 0.95
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [2, 1]
    assert ranked[0]["label"] == "pregnancy and pregnant status"
    assert ranked[0]["caveat"] and ranked[0]["tracked"] is None


def test_rank_unknown_ceiling_sinks_below_scored(monkeypatch):
    stats = {
        1: {"label": "known", "max_act": 1000.0, "density": 0.001},   # rel 0.9
        2: {"label": "loud token", "max_act": None, "density": None},  # rel 0 -> sinks
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [
        {"index": 2, "act": 9999.0, "source": "s"},  # huge raw, unknown ceiling
        {"index": 1, "act": 900.0, "source": "s"},   # near its ceiling
    ]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [1, 2]


def test_rank_drops_unlabeled_features(monkeypatch):
    monkeypatch.setattr(analyze.config, "DROP_UNLABELED", True)
    stats = {
        1: {"label": "pregnancy", "max_act": 2000.0, "density": 0.001},
        2: {"label": "feature 2", "max_act": 5000.0, "density": 0.02},  # unlabeled -> dropped
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    candidates = [{"index": 2, "act": 4000.0, "source": "s"}, {"index": 1, "act": 1900.0, "source": "s"}]
    ranked = analyze._rank_features(candidates)
    assert [f["index"] for f in ranked] == [1]


def test_syntactic_labels_are_downranked(monkeypatch):
    monkeypatch.setattr(analyze.config, "DENSITY_MAX", 1.0)
    monkeypatch.setattr(analyze.config, "SYNTACTIC_PENALTY", 0.12)
    # both fire near their ceiling (rel ~0.95); the syntactic one must sink below the concept
    stats = {
        1: {"label": "adverbs modifying subsequent words", "max_act": 1000.0, "density": 0.001},
        2: {"label": "pregnancy and pregnant status", "max_act": 1000.0, "density": 0.001},
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    cands = [{"index": 1, "act": 950.0, "source": "s"}, {"index": 2, "act": 950.0, "source": "s"}]
    ranked = analyze._rank_features(cands)
    assert [f["index"] for f in ranked] == [2, 1]  # concept outranks syntactic


def test_is_syntactic_classifier():
    assert analyze._is_syntactic("it's followed by verb or adjective")
    assert analyze._is_syntactic("adverbs modifying subsequent words")
    assert analyze._is_syntactic("contextual prepositions")
    assert analyze._is_syntactic("Okay,")
    assert analyze._is_syntactic("numbered lists and code snippets")
    assert not analyze._is_syntactic("pregnancy and pregnant status")
    assert not analyze._is_syntactic("heart and cardiac events")


def test_rank_drops_dense_grammatical_features(monkeypatch):
    monkeypatch.setattr(analyze.config, "DROP_UNLABELED", True)
    monkeypatch.setattr(analyze.config, "DENSITY_MAX", 0.01)
    stats = {
        1: {"label": "pregnancy", "max_act": 2000.0, "density": 0.001},      # rare -> kept
        2: {"label": "forms of to be", "max_act": 3000.0, "density": 0.05},  # dense -> dropped
    }
    monkeypatch.setattr(analyze.labels, "get_feature_stats", lambda i, **k: stats[i])
    cands = [{"index": 2, "act": 2900.0, "source": "s"}, {"index": 1, "act": 1900.0, "source": "s"}]
    ranked = analyze._rank_features(cands)
    assert [f["index"] for f in ranked] == [1]


def test_rank_falls_back_when_all_unlabeled(monkeypatch):
    monkeypatch.setattr(analyze.config, "DROP_UNLABELED", True)
    monkeypatch.setattr(
        analyze.labels, "get_feature_stats",
        lambda i, **k: {"label": f"feature {i}", "max_act": None, "density": None},
    )
    candidates = [{"index": 1, "act": 10.0, "source": "s"}, {"index": 2, "act": 5.0, "source": "s"}]
    ranked = analyze._rank_features(candidates)
    assert {f["index"] for f in ranked} == {1, 2}  # never an empty cloud


def test_rank_truncates_to_topk_event(monkeypatch):
    monkeypatch.setattr(
        analyze.labels, "get_feature_stats",
        lambda i, **k: {"label": f"f{i}", "max_act": 100.0, "density": 0.01},
    )
    candidates = [{"index": i, "act": float(i), "source": "s"} for i in range(80)]
    ranked = analyze._rank_features(candidates)
    assert len(ranked) == analyze.config.TOPK_EVENT
