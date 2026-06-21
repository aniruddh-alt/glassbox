from backend.science import concept_synth as cs


def test_create_job_returns_id_and_pending():
    tid = cs.create_job("watch for sycophancy")
    job = cs.get_job(tid)
    assert job is not None
    assert job["status"] == "pending"
    assert job["request"] == "watch for sycophancy"
    assert job["auroc"] is None


def test_update_job_transitions_status():
    tid = cs.create_job("watch for hedging")
    cs.update_job(tid, status="fitting", auroc=0.91, trait_name="hedging")
    job = cs.get_job(tid)
    assert job["status"] == "fitting"
    assert job["auroc"] == 0.91
    assert job["trait_name"] == "hedging"


def test_get_job_unknown_returns_none():
    assert cs.get_job("nope") is None
