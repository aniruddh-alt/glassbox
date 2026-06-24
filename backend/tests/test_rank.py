from backend.analyze import _rank_features
from backend.config import AppConfig, FeatureCloudConfig


def test_rank_drops_unlabeled_when_enabled():
    fc = FeatureCloudConfig()  # drop_unlabeled=True
    candidates = [
        {"index": 1, "label": "drug safety", "act": 5.0},
        {"index": 2, "label": "feature 2", "act": 9.0},  # bare label → dropped
    ]
    out = _rank_features(candidates, fc, AppConfig())
    labels = {f["label"] for f in out}
    assert "feature 2" not in labels


def test_rank_keeps_unlabeled_when_disabled():
    fc = FeatureCloudConfig(drop_unlabeled=False)
    candidates = [
        {"index": 1, "label": "drug safety", "act": 5.0},
        {"index": 2, "label": "feature 2", "act": 9.0},
    ]
    out = _rank_features(candidates, fc, AppConfig())
    assert len(out) == 2


def test_rank_caps_at_topk_event():
    fc = FeatureCloudConfig(topk_event=3, drop_unlabeled=False)
    candidates = [{"index": i, "label": f"concept {i}", "act": float(10 - i)} for i in range(10)]
    out = _rank_features(candidates, fc, AppConfig())
    assert len(out) <= 3
