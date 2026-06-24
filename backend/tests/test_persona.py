import json
from pathlib import Path

import pytest
import torch

from backend.config import ProbeConfig
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

    probes = ProbeConfig(artifacts_dir=tmp_path, disabled=[])
    loaded = persona.load_artifacts(probes)
    out = persona.score_all_trackers(None, torch.tensor([-2.0, 0.0]))

    assert loaded == ["risk_awareness"]
    assert out["risk_awareness"]["flag"] is True
    assert persona._trackers["risk_awareness"]["meta"]["artifact"] == str(artifact)


def test_template_artifact_does_not_register_without_direction(tmp_path):
    artifact = tmp_path / "risk_awareness.json"
    artifact.write_text(json.dumps({"id": "risk_awareness", "threshold": 0.35}))

    probes = ProbeConfig(artifacts_dir=tmp_path, disabled=[])
    assert persona.load_artifacts(probes) == []
    assert persona.score_all_trackers(None, torch.tensor([1.0, 0.0])) == {}


def test_load_artifacts_respects_disabled(tmp_path):
    # two artifacts on disk: one enabled-named, one disabled-named
    (tmp_path / "harmful.json").write_text(
        '{"tracker_id": "harmful", "direction": [0.1, 0.2], "norm_mean": null, "norm_std": null, "threshold": 0.5}'
    )
    (tmp_path / "uncertainty.json").write_text(
        '{"tracker_id": "uncertainty", "direction": [0.1, 0.2], "norm_mean": null, "norm_std": null, "threshold": 0.5}'
    )
    persona.clear_trackers()
    probes = ProbeConfig(artifacts_dir=tmp_path, disabled=["uncertainty"])
    loaded = persona.load_artifacts(probes)
    assert "harmful" in loaded
    assert "uncertainty" not in loaded


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


# NOTE: tests for the branch's old concept_synth.synth_concept were removed during the
# merge with main — main's job-based concept_synth (create_job/get_job) supersedes it.
# The job-based /api/track flow is currently untested (a pre-existing gap on main).


def _noncpu_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_project_reconciles_artifact_direction_to_activation_device():
    """The probe pipeline runs on the GPU microservice: activations live on the model device
    (CUDA on the pod) while the direction + norm tensors come from JSON artifacts (CPU). project()
    must move them onto the activation device, or the matmul raises a device mismatch and the
    tracker is silently skipped. Reproduced on CUDA/MPS; skipped on CPU-only, where no mismatch
    can occur."""
    dev = _noncpu_device()
    if dev is None:
        pytest.skip("needs a non-CPU device (CUDA on the pod, MPS locally)")
    acts = torch.tensor([3.0, 0.0], device=dev)
    proj = persona.project(acts, [1.0, 0.0], norm_mean=[1.0, 0.0], norm_std=[1.0, 1.0])
    assert float(proj) == pytest.approx(2.0)


def test_score_all_trackers_scores_activations_on_device():
    """End-to-end of the live-probe bug: an artifact-loaded tracker (CPU direction) must score
    activations that live on the model device. Before the device fix every tracker was silently
    skipped, leaving the probe meters empty despite all artifacts being loaded."""
    dev = _noncpu_device()
    if dev is None:
        pytest.skip("needs a non-CPU device (CUDA on the pod, MPS locally)")
    persona.register_tracker("uncertainty", direction=[1.0, 0.0], threshold=0.5)
    out = persona.score_all_trackers(
        act_last=torch.tensor([0.0, 1.0], device=dev),
        act_resp=torch.tensor([2.0, 0.0], device=dev),
    )
    assert set(out) == {"uncertainty"}
    assert out["uncertainty"]["proj"] == pytest.approx(2.0)
    assert out["uncertainty"]["flag"] is True
