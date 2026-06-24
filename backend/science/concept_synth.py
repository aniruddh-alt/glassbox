"""User-defined concepts on demand."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

_jobs: dict[str, dict] = {}
JOBS_DIR = Path(__file__).with_name("probe_jobs")
_TERMINAL = frozenset({"ready", "rejected", "error"})


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    return "".join(keep).strip("-")[:24] or "concept"


def _persist_job(tracker_id: str) -> None:
    job = _jobs.get(tracker_id)
    if job is None:
        return
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / f"{tracker_id}.json").write_text(json.dumps(job, indent=2))


def load_persisted_jobs(*, mark_orphans: bool = True) -> int:
    """Reload job records from disk. Orphan in-flight jobs become errors after pod restart."""
    if not JOBS_DIR.exists():
        return 0
    loaded = 0
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            job = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tid = job.get("tracker_id") or path.stem
        job["tracker_id"] = tid
        if mark_orphans and job.get("status") not in _TERMINAL:
            job["status"] = "error"
            job["error"] = job.get("error") or "pod restarted while build was in progress"
        _jobs[tid] = job
        loaded += 1
    return loaded


def active_job_count() -> int:
    return sum(1 for j in _jobs.values() if j.get("status") not in _TERMINAL)


def create_job(request: str) -> str:
    """Register a new tracking job in 'pending'. Returns its tracker_id."""
    tracker_id = f"{_slug(request)}-{uuid.uuid4().hex[:6]}"
    _jobs[tracker_id] = {
        "tracker_id": tracker_id,
        "request": request,
        "status": "pending",
        "progress": {"step": "queued", "pct": 0},
        "trait_name": None,
        "auroc": None,
        "baseline_auroc": None,
        "n_kept": None,
        "verdict": None,
        "error": None,
    }
    _persist_job(tracker_id)
    return tracker_id


def update_job(tracker_id: str, **fields) -> None:
    job = _jobs.get(tracker_id)
    if job is not None:
        job.update(fields)
        _persist_job(tracker_id)


def get_job(tracker_id: str) -> dict | None:
    return _jobs.get(tracker_id)


def fit_and_validate(rows: list[dict]) -> dict:
    """Stratified train/held-out split → diff-of-means direction + calibrated probe.
    Measures held-out AUROC and a plain-LogReg baseline. direction/calibrator/threshold
    are fit on ALL rows for deployment. Returns insufficient_data if a class is too small."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from . import persona

    fail = {
        "status": "insufficient_data", "auroc": None, "baseline_auroc": None,
        "n_kept": len(rows), "calibrated": False,
        "direction": None, "calibrator": None, "threshold": None,
    }
    y = np.array([int(r["label"]) for r in rows])
    if len(rows) < 6 or (y == 1).sum() < 3 or (y == 0).sum() < 3:
        return fail

    X = torch.stack([r["act_resp"].float() for r in rows]).cpu()
    idx = np.arange(len(rows))
    try:
        tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=0)
    except ValueError:
        return fail
    if y[te].sum() == 0 or y[te].sum() == len(te):
        return fail

    clf, _ = persona.train_probe(X[tr], y[tr])
    probs = clf.predict_proba(X[te].numpy())[:, 1]
    auroc = float(roc_auc_score(y[te], probs))

    base = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr].numpy(), y[tr])
    baseline_auroc = float(roc_auc_score(y[te], base.predict_proba(X[te].numpy())[:, 1]))

    direction = persona.persona_vector(X[y == 1], X[y == 0])
    clf_all, threshold = persona.train_probe(X, y)
    calibrated = clf_all.__class__.__name__ == "CalibratedClassifierCV"
    return {
        "status": "ok", "auroc": auroc, "baseline_auroc": baseline_auroc,
        "n_kept": len(rows), "calibrated": calibrated,
        "direction": direction, "calibrator": clf_all, "threshold": threshold,
    }


def generate_contrastive(
    spec: dict,
    *,
    generate_fn=None,
    max_new: int = 64,
    on_progress=None,
) -> list[dict]:
    """For each question, run the model under pos and neg system prompts; capture activations."""
    if generate_fn is None:
        from .. import engine

        def generate_fn(messages, max_new=max_new):
            return engine.generate_and_capture(
                messages, max_new=max_new, attribution=False
            )

    rows: list[dict] = []
    total = len(spec["questions"]) * 2
    done = 0
    for label, prompt in ((1, spec["pos_prompt"]), (0, spec["neg_prompt"])):
        for q in spec["questions"]:
            messages = [{"role": "user", "content": f"{prompt}\n\n{q}"}]
            cap = generate_fn(messages, max_new=max_new)
            acts = cap["acts"]
            start = cap["resp_start"]
            if start <= 0 or acts.shape[0] <= start:
                raise ValueError("model returned no response tokens for contrastive generation")
            rows.append({
                "response": cap["answer"],
                "act_resp": acts[start:].float().mean(0),
                "act_last": acts[start - 1].float(),
                "intended_label": label,
            })
            done += 1
            if on_progress:
                on_progress(done, total)
    return rows


def _judge_batch(
    spec: dict,
    batch: list[dict],
    *,
    client,
    model: str,
) -> list[int]:
    """Score one batch of responses; retries once on count mismatch."""
    import json

    from ..agent import prompts

    responses = [r["response"] for r in batch]
    last_err: Exception | None = None
    for attempt in range(2):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            output_config={"format": prompts.judge_schema(len(responses))},
            messages=[{"role": "user", "content": prompts.judge_prompt(spec, responses)}],
        )
        text_blocks = [b.text for b in resp.content if b.type == "text"]
        if not text_blocks:
            last_err = ValueError("judge returned no text block (check JUDGE_MODEL / structured output)")
            continue
        try:
            scores = json.loads(text_blocks[0])["scores"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_err = ValueError(f"judge returned invalid JSON: {e}")
            continue
        if len(scores) != len(responses):
            last_err = ValueError(
                f"judge returned {len(scores)} scores for {len(responses)} responses"
            )
            continue
        return [int(s) for s in scores]
    raise last_err or ValueError("judge batch failed")


def judge_filter(spec: dict, rows: list[dict], builder, anthropic_api_key: str, *, client=None) -> list[dict]:
    """Score each response 1-5; keep unambiguous positives (>=4) and negatives (<=2)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_api_key)
    model = builder.judge_model
    batch_size = max(1, builder.judge_batch_size)

    all_scores: list[int] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        all_scores.extend(_judge_batch(spec, batch, client=client, model=model))

    kept: list[dict] = []
    for row, score in zip(rows, all_scores):
        if row["intended_label"] == 1 and score >= 4:
            kept.append({**row, "label": 1})
        elif row["intended_label"] == 0 and score <= 2:
            kept.append({**row, "label": 0})
    return kept
