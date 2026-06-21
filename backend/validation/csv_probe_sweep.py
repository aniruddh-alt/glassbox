"""CSV harmful/benign layer sweep — GlassBox-native version of the external nnsight script.

Differences from the upstream Colab script are documented in docs/probe-training.md §
"External CSV sweep alignment". Uses backend.engine (no nnsight) and optional response-token
pooling to match live gpu_service scoring.

Usage:
  # Export contrastive prompts from harmful.json into fixtures/probe_prompts/*.csv
  uv run python -m backend.validation.csv_probe_sweep --export-from-artifact

  # Layer sweep (needs HF model cached; offline ok)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    uv run python -m backend.validation.csv_probe_sweep \\
      --harmful fixtures/probe_prompts/harmful.csv \\
      --benign fixtures/probe_prompts/benign.csv \\
      --layers 12,13,14,15,16,17,18,19,20 \\
      --pool response
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import config
from backend.science import persona
from backend.validation.harmfulness_pipeline import _direction


def load_prompts(path: str | Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["prompt"].strip() for row in csv.DictReader(f) if row.get("prompt", "").strip()]


def export_from_artifact(
    artifact_path: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write harmful.csv / benign.csv from a contrastive artifact (pos/neg × questions)."""
    data = json.loads(Path(artifact_path).read_text())
    pair = (data.get("instruction") or [{}])[0]
    pos, neg = pair.get("pos"), pair.get("neg")
    questions = data.get("questions") or []
    if not pos or not neg or not questions:
        raise ValueError("artifact needs instruction[0].pos/neg and questions")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    harmful = root / "harmful.csv"
    benign = root / "benign.csv"
    with harmful.open("w", newline="", encoding="utf-8") as hf, benign.open(
        "w", newline="", encoding="utf-8"
    ) as bf:
        hw = csv.DictWriter(hf, fieldnames=["prompt"])
        bw = csv.DictWriter(bf, fieldnames=["prompt"])
        hw.writeheader()
        bw.writeheader()
        for q in questions:
            hw.writerow({"prompt": f"{pos}\n\nPatient question: {q}"})
            bw.writerow({"prompt": f"{neg}\n\nPatient question: {q}"})
    return harmful, benign


class EngineCsvProvider:
    """Extract mean-pooled residual activations via backend.engine."""

    def __init__(self, *, pool: str = "response", max_new: int = 128):
        if pool not in {"response", "prompt"}:
            raise ValueError("pool must be 'response' or 'prompt'")
        self.pool = pool
        self.max_new = max_new

    def activation(self, prompt: str, *, layer: int) -> np.ndarray:
        from backend import engine

        old_layer = config.LAYER
        config.LAYER = int(layer)
        try:
            engine.load_engine()
            if self.pool == "prompt":
                act = engine.probe_activation(prompt)
                return act.float().mean(0).detach().cpu().numpy()
            res = engine.generate_and_capture(
                [{"role": "user", "content": prompt}],
                max_new=self.max_new,
                attribution=False,
            )
            acts = res["acts"][res["resp_start"] :]
            if acts.shape[0] == 0:
                raise RuntimeError("empty response activations")
            return acts.float().mean(0).detach().cpu().numpy()
        finally:
            config.LAYER = old_layer

    def collect(self, prompts: list[str], *, layer: int) -> np.ndarray:
        rows = []
        for i, p in enumerate(prompts):
            print(f"  layer {layer} prompt {i + 1}/{len(prompts)}", flush=True)
            rows.append(self.activation(p, layer=layer))
        return np.asarray(rows, dtype="float32")


def normed_diff_of_means_cv(X: np.ndarray, y: np.ndarray, *, cv_splits: int = 5) -> tuple[float, np.ndarray]:
    """Cross-validated normed diff-of-means scores (matches external script logic)."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        d = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
        d /= np.linalg.norm(d) + 1e-8
        oof[te] = ((X[te] - mu) / sd) @ d
    return float(roc_auc_score(y, oof)), oof


def sweep(
    harmful: list[str],
    benign: list[str],
    *,
    layers: list[int],
    provider: EngineCsvProvider,
    cv_splits: int = 5,
) -> dict:
    y = np.array([1] * len(harmful) + [0] * len(benign))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    results: dict[int, dict] = {}

    print(f"\nCollecting harmful={len(harmful)} benign={len(benign)} pool={provider.pool}")
    for layer in layers:
        H = provider.collect(harmful, layer=layer)
        B = provider.collect(benign, layer=layer)
        X = np.concatenate([H, B])

        lr_oof = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
            X,
            y,
            cv=cv,
            method="decision_function",
        )
        lr_auroc = float(roc_auc_score(y, lr_oof))

        raw_d = H.mean(0) - B.mean(0)
        raw_auroc = float(roc_auc_score(y, np.concatenate([H, B]) @ (raw_d / (np.linalg.norm(raw_d) + 1e-8))))

        nrm_auroc, nrm_oof = normed_diff_of_means_cv(X, y, cv_splits=cv_splits)
        recall = float((nrm_oof[y == 1] > np.percentile(nrm_oof[y == 0], 99)).mean())

        results[layer] = {
            "lr": lr_auroc,
            "raw": raw_auroc,
            "normed": nrm_auroc,
            "recall_at_1pct_fpr": recall,
        }
        print(
            f"{layer:>5} | LR {lr_auroc:>7.4f} | raw {raw_auroc:>7.4f} | "
            f"normed {nrm_auroc:>7.4f} | recall@1%FPR {recall:>6.1%}"
        )
    return results


def write_artifact_from_best(
    artifact_path: str | Path,
    *,
    layer: int,
    harmful: list[str],
    benign: list[str],
    provider: EngineCsvProvider,
) -> None:
    """Train normed direction on full CSV data and write into a GlassBox artifact JSON."""
    H = provider.collect(harmful, layer=layer)
    B = provider.collect(benign, layer=layer)
    X = np.concatenate([H, B])
    y = np.concatenate([np.ones(len(H)), np.zeros(len(B))]).astype(int)
    bundle = _direction(X, y, method="normed")
    clf, threshold = persona.train_probe(X, y)

    path = Path(artifact_path)
    artifact = json.loads(path.read_text()) if path.exists() else {"id": path.stem}
    artifact.update(
        {
            "layer": layer,
            "threshold": float(threshold),
            "auroc": float(roc_auc_score(y, clf.predict_proba(X)[:, 1])),
            "direction_method": bundle["direction_method"],
            "direction": bundle["direction"].tolist(),
            "norm_mean": bundle["norm_mean"].tolist(),
            "norm_std": bundle["norm_std"].tolist(),
        }
    )
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote normed probe -> {path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CSV harmful/benign probe layer sweep (GlassBox engine)")
    p.add_argument("--harmful", default="fixtures/probe_prompts/harmful.csv")
    p.add_argument("--benign", default="fixtures/probe_prompts/benign.csv")
    p.add_argument("--layers", default="12,13,14,15,16,17,18,19,20")
    p.add_argument("--pool", choices=["response", "prompt"], default="response")
    p.add_argument("--export-from-artifact", default=None, metavar="PATH")
    p.add_argument("--out-dir", default="fixtures/probe_prompts")
    p.add_argument("--write-artifact", default=None, help="Update artifact JSON with best normed layer")
    args = p.parse_args(argv)

    if args.export_from_artifact:
        h, b = export_from_artifact(args.export_from_artifact, args.out_dir)
        print(f"Exported\n  {h}\n  {b}")
        return

    harmful = load_prompts(args.harmful)
    benign = load_prompts(args.benign)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    provider = EngineCsvProvider(pool=args.pool)

    print("=" * 70)
    print(f"{'layer':>5} | {'LR':>10} | {'raw DoM':>10} | {'normed DoM':>12} | {'recall@1%FPR':>12}")
    print("=" * 70)
    results = sweep(harmful, benign, layers=layers, provider=provider)
    print("=" * 70)

    best = max(results, key=lambda L: (results[L]["normed"], -abs(L - config.LAYER)))
    r = results[best]
    print(f"\nBEST layer (normed DoM): {best}")
    print(f"  normed AUROC: {r['normed']:.4f}")
    print(f"  LR AUROC    : {r['lr']:.4f}")
    print(f"  recall@1%FPR: {r['recall_at_1pct_fpr']:.1%}")

    if args.write_artifact:
        write_artifact_from_best(
            args.write_artifact,
            layer=best,
            harmful=harmful,
            benign=benign,
            provider=provider,
        )


if __name__ == "__main__":
    main()
