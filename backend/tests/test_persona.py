import json

import pytest
import torch

from backend.science import concept_synth
from backend.science import persona


@pytest.fixture(autouse=True)
def _clear_trackers():
    persona.clear_trackers()
    yield
    persona.clear_trackers()


def test_register_tracker_scores_bounded_projection():
    persona.register_tracker("harmful", direction=torch.tensor([1.0, 0.0]), threshold=0.5)

    out = persona.score_all_trackers(
        act_last=torch.tensor([0.0, 1.0]),
        act_resp=torch.tensor([2.0, 0.0]),
    )

    assert set(out) == {"harmful"}
    assert 0.0 <= out["harmful"]["score"] <= 1.0
    assert out["harmful"]["proj"] == pytest.approx(2.0)
    assert out["harmful"]["proj_pre"] == pytest.approx(0.0)
    assert out["harmful"]["flag"] is True
    assert out["harmful"]["alert_direction"] == "high"


def test_risk_awareness_flags_when_score_is_low():
    persona.register_tracker(
        "risk_awareness",
        direction=torch.tensor([1.0, 0.0]),
        threshold=0.35,
        alert_direction="low",
    )

    low_awareness = persona.score_all_trackers(None, torch.tensor([-2.0, 0.0]))
    high_awareness = persona.score_all_trackers(None, torch.tensor([2.0, 0.0]))

    assert low_awareness["risk_awareness"]["score"] < 0.35
    assert low_awareness["risk_awareness"]["flag"] is True
    assert high_awareness["risk_awareness"]["score"] > 0.35
    assert high_awareness["risk_awareness"]["flag"] is False


def test_load_tracker_artifact_registers_ready_direction(tmp_path):
    artifact = tmp_path / "risk_awareness.json"
    artifact.write_text(
        json.dumps(
            {
                "id": "risk_awareness",
                "description": "low scores mean missing risk caveats",
                "direction": [1.0, 0.0],
                "threshold": 0.35,
                "alert_direction": "low",
            }
        )
    )

    loaded = persona.load_artifacts(tmp_path)
    out = persona.score_all_trackers(None, torch.tensor([-2.0, 0.0]))

    assert loaded == ["risk_awareness"]
    assert out["risk_awareness"]["flag"] is True
    assert persona._trackers["risk_awareness"]["meta"]["artifact"] == str(artifact)


def test_template_artifact_does_not_register_without_direction(tmp_path):
    artifact = tmp_path / "risk_awareness.json"
    artifact.write_text(json.dumps({"id": "risk_awareness", "threshold": 0.35}))

    assert persona.load_artifacts(tmp_path) == []
    assert persona.score_all_trackers(None, torch.tensor([1.0, 0.0])) == {}


def test_malformed_tracker_does_not_fail_other_scores():
    persona.register_tracker("bad_shape", direction=torch.tensor([1.0, 0.0, 0.0]))
    persona.register_tracker("risk_awareness", direction=torch.tensor([1.0, 0.0]))

    out = persona.score_all_trackers(None, torch.tensor([1.0, 0.0]))

    assert set(out) == {"risk_awareness"}


def test_normed_direction_applies_per_dimension_scaling():
    persona.register_tracker(
        "harmful",
        direction=torch.tensor([1.0, 0.0]),
        norm_mean=torch.tensor([1.0, 0.0]),
        norm_std=torch.tensor([1.0, 1.0]),
        direction_method="normed_diff_of_means",
    )
    proj = persona.project(torch.tensor([3.0, 0.0]), persona._trackers["harmful"]["dir"],
                          norm_mean=persona._trackers["harmful"]["norm_mean"],
                          norm_std=persona._trackers["harmful"]["norm_std"])
    assert float(proj) == pytest.approx(2.0)


def test_register_tracker_rejects_invalid_monitor_config():
    with pytest.raises(ValueError, match="non-zero"):
        persona.register_tracker("empty", direction=torch.tensor([0.0, 0.0]))

    with pytest.raises(ValueError, match="alert_direction"):
        persona.register_tracker("bad", direction=torch.tensor([1.0]), alert_direction="middle")


def test_synth_concept_generates_harmfulness_template_from_prompt(tmp_path):
    result = concept_synth.synth_concept(
        "harmfulness",
        "Detect clinically harmful or unsafe medical advice, including contraindications, dangerous omissions, and over-reassurance.",
        artifact_dir=tmp_path,
    )

    artifact = json.loads((tmp_path / "harmful.json").read_text())
    assert result == {"tracker_id": "harmful", "status": "computing", "artifact": str(tmp_path / "harmful.json")}
    assert artifact["id"] == "harmful"
    assert artifact["concept"] == "harmfulness"
    assert artifact["alert_direction"] == "high"
    assert artifact["threshold"] == pytest.approx(0.6)
    assert "clinically harmful" in artifact["description"]
    assert "HARMFUL" in artifact["eval_prompt"]
    assert artifact["instruction"][0]["pos"]
    assert artifact["instruction"][0]["neg"]
    assert len(artifact["questions"]) >= 6


def test_synth_concept_registers_harmfulness_when_direction_is_ready(tmp_path):
    result = concept_synth.synth_concept(
        "harmful",
        "Detect unsafe medical advice.",
        artifact_dir=tmp_path,
        direction=[1.0, 0.0],
    )
    out = persona.score_all_trackers(None, torch.tensor([2.0, 0.0]))

    assert result["tracker_id"] == "harmful"
    assert result["status"] == "ready"
    assert out["harmful"]["flag"] is True
    assert out["harmful"]["alert_direction"] == "high"
