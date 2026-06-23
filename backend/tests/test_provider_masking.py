from backend.config import FeatureCloudConfig, ModelConfig, SAEConfig
from backend.science import feature_provider as fp


class FakeActs:
    """Minimal stand-in for a [n, d] activation tensor: exposes .shape and row indexing."""

    def __init__(self, n: int):
        self.shape = (n, 4)

    def __getitem__(self, pos: int):
        return ("row", pos)


def test_local_provider_skips_special_positions(monkeypatch):
    # one synthetic feature per position, index encodes the position
    def fake_topk(act, sae, feature_cloud, *, np_source=None, k=None):
        pos = act[1]
        return [{"index": 100 + pos, "act": float(10 - pos), "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    prov = fp.LocalSAEProvider()
    feats = prov.features_for(
        "txt", activations=FakeActs(4), token_ids=[1, 999, 2, 3], special_ids={999},
        sae=SAEConfig(), feature_cloud=FeatureCloudConfig(), model=ModelConfig(),
        np_source="17-gemmascope-2-res-16k",
    )
    idxs = {f["index"] for f in feats}
    assert 101 not in idxs                       # position 1 (token 999) masked
    assert {100, 102, 103} <= idxs               # other positions kept
    # ranked by activation descending
    assert feats[0]["act"] >= feats[-1]["act"]


def test_local_provider_no_special_keeps_all(monkeypatch):
    def fake_topk(act, sae, feature_cloud, *, np_source=None, k=None):
        pos = act[1]
        return [{"index": 100 + pos, "act": 1.0, "source": "s"}]

    monkeypatch.setattr(fp, "sae_topk", fake_topk)
    prov = fp.LocalSAEProvider()
    feats = prov.features_for(
        "txt", activations=FakeActs(3), token_ids=[1, 2, 3],
        sae=SAEConfig(), feature_cloud=FeatureCloudConfig(), model=ModelConfig(),
        np_source="17-gemmascope-2-res-16k",
    )
    assert {f["index"] for f in feats} == {100, 101, 102}
