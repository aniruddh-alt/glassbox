"""End-to-end harmfulness probe pipeline.

Offline only: build contrastive harmful/safe examples from the artifact prompt, extract residual
activations, train a calibrated probe, sweep layers by held-out ROC AUC, and write the best
direction/metrics back to the artifact so the runtime can load it as a ready tracker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from backend import config
from backend.science import persona


@dataclass(frozen=True)
class ProbeExample:
    id: str
    question: str
    instruction: str
    label: int
    split: str


class ActivationProvider(Protocol):
    def activations(self, examples: list[ProbeExample], *, layer: int) -> np.ndarray:
        """Return one mean-pooled residual activation per example."""


def _load_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def build_dataset(artifact_path: str | Path, *, test_every: int = 4) -> list[ProbeExample]:
    """Build a balanced contrastive dataset from a harmfulness artifact."""
    artifact = _load_artifact(artifact_path)
    pair = (artifact.get("instruction") or [{}])[0]
    pos = pair.get("pos")
    neg = pair.get("neg")
    questions = artifact.get("questions") or []
    if not pos or not neg:
        raise ValueError("artifact must contain instruction[0].pos and instruction[0].neg")
    if len(questions) < 2:
        raise ValueError("artifact must contain at least two questions")

    examples: list[ProbeExample] = []
    for i, question in enumerate(questions):
        split = "test" if (i + 1) % test_every == 0 else "train"
        examples.append(
            ProbeExample(
                id=f"q{i}:harmful",
                question=question,
                instruction=pos,
                label=1,
                split=split,
            )
        )
        examples.append(
            ProbeExample(
                id=f"q{i}:safe",
                question=question,
                instruction=neg,
                label=0,
                split=split,
            )
        )

    if {ex.split for ex in examples} != {"train", "test"}:
        half = len(questions) // 2
        examples = [
            ProbeExample(ex.id, ex.question, ex.instruction, ex.label, "test" if i // 2 >= half else "train")
            for i, ex in enumerate(examples)
        ]
    return examples


def _direction(X: np.ndarray, y: np.ndarray, *, method: str = "raw") -> dict:
    """Compute a unit probe direction from train-fold activations."""
    if method == "normed":
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sd
        d = Xn[y == 1].mean(axis=0) - Xn[y == 0].mean(axis=0)
        norm = np.linalg.norm(d)
        if norm <= 1e-8:
            raise ValueError("cannot train persona direction: positive/negative means are identical")
        return {
            "direction_method": "normed_diff_of_means",
            "direction": (d / norm).astype("float32"),
            "norm_mean": mu.astype("float32"),
            "norm_std": sd.astype("float32"),
        }

    d = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    norm = np.linalg.norm(d)
    if norm <= 1e-8:
        raise ValueError("cannot train persona direction: positive/negative means are identical")
    return {
        "direction_method": "raw_diff_of_means",
        "direction": (d / norm).astype("float32"),
        "norm_mean": None,
        "norm_std": None,
    }


def _split(examples: list[ProbeExample], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray([ex.label for ex in examples], dtype="int64")
    train = np.asarray([ex.split == "train" for ex in examples], dtype=bool)
    test = ~train
    if len(set(y[train].tolist())) < 2 or len(set(y[test].tolist())) < 2:
        raise ValueError("train and test splits must each contain both classes")
    return X[train], y[train], X[test], y[test]


def train_layer(
    examples: list[ProbeExample],
    X: np.ndarray,
    *,
    layer: int,
    direction_method: str = "raw",
) -> dict:
    """Train/evaluate one layer and return metrics plus the learned direction."""
    from sklearn.metrics import roc_auc_score

    X_train, y_train, X_test, y_test = _split(examples, X)
    clf, threshold = persona.train_probe(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    try:
        bundle = _direction(X_train, y_train, method=direction_method)
        direction = bundle["direction"].tolist()
    except ValueError:
        bundle = {
            "direction_method": "raw_diff_of_means",
            "direction": np.zeros(X_train.shape[1], dtype="float32"),
            "norm_mean": None,
            "norm_std": None,
        }
        direction = bundle["direction"].tolist()
    out = {
        "layer": int(layer),
        "auroc": auroc,
        "threshold": float(threshold),
        "direction": direction,
        "direction_method": bundle["direction_method"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    if bundle.get("norm_mean") is not None:
        out["norm_mean"] = bundle["norm_mean"].tolist()
        out["norm_std"] = bundle["norm_std"].tolist()
    return out


def sweep_layers(
    examples: list[ProbeExample],
    *,
    layers: list[int],
    provider: ActivationProvider,
    direction_method: str = "raw",
) -> list[dict]:
    results = []
    for layer in layers:
        X = np.asarray(provider.activations(examples, layer=layer), dtype="float32")
        if X.shape[0] != len(examples):
            raise ValueError(f"provider returned {X.shape[0]} rows for {len(examples)} examples")
        results.append(
            train_layer(examples, X, layer=layer, direction_method=direction_method)
        )
    return results


def _pick_best(sweep: list[dict]) -> dict:
    """Prefer higher AUROC; on ties prefer the layer closest to config.LAYER (runtime hook)."""
    return max(sweep, key=lambda r: (r["auroc"], -abs(r["layer"] - config.LAYER)))


def write_ready_artifact(
    artifact_path: str | Path,
    *,
    best: dict,
    sweep: list[dict],
    min_auroc: float,
) -> None:
    path = Path(artifact_path)
    artifact = _load_artifact(path)
    if best["auroc"] < min_auroc:
        raise RuntimeError(f"best AUROC {best['auroc']:.3f} is below minimum {min_auroc:.3f}")

    artifact["direction"] = best["direction"]
    artifact["threshold"] = best["threshold"]
    artifact["layer"] = best["layer"]
    artifact["auroc"] = best["auroc"]
    artifact["direction_method"] = best.get("direction_method", "raw_diff_of_means")
    if best.get("norm_mean") is not None:
        artifact["norm_mean"] = best["norm_mean"]
        artifact["norm_std"] = best["norm_std"]
    artifact["validation"] = {
        "metric": "roc_auc",
        "best_layer": best["layer"],
        "best_auroc": best["auroc"],
        "n_train": best["n_train"],
        "n_test": best["n_test"],
        "sweep": [
            {
                "layer": r["layer"],
                "auroc": r["auroc"],
                "threshold": r["threshold"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
            for r in sweep
        ],
    }
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def run_pipeline(
    *,
    artifact_path: str | Path,
    layers: list[int],
    provider: ActivationProvider,
    min_auroc: float = 0.7,
    direction_method: str = "raw",
) -> dict:
    examples = build_dataset(artifact_path)
    sweep = sweep_layers(
        examples, layers=layers, provider=provider, direction_method=direction_method
    )
    best = _pick_best(sweep)
    write_ready_artifact(artifact_path, best=best, sweep=sweep, min_auroc=min_auroc)
    return {
        "tracker_id": _load_artifact(artifact_path).get("id", Path(artifact_path).stem),
        "best_layer": best["layer"],
        "best_auroc": best["auroc"],
        "sweep": [{k: r[k] for k in ("layer", "auroc", "threshold")} for r in sweep],
    }


class EngineActivationProvider:
    """Activation provider backed by the local HF model in backend.engine."""

    def __init__(self, *, max_new: int = 128, device: str | None = None):
        self.max_new = max_new
        self.device = device

    def activations(self, examples: list[ProbeExample], *, layer: int) -> np.ndarray:
        from backend import engine

        old_layer = config.LAYER
        config.LAYER = int(layer)
        try:
            engine.load_engine(device=self.device)
            rows = []
            for ex in examples:
                res = engine.generate_and_capture(
                    [
                        {"role": "system", "content": ex.instruction},
                        {"role": "user", "content": ex.question},
                    ],
                    max_new=self.max_new,
                    attribution=False,
                )
                acts = res["acts"][res["resp_start"] :]
                rows.append(acts.float().mean(0).detach().cpu().numpy())
            return np.asarray(rows, dtype="float32")
        finally:
            config.LAYER = old_layer


class LexicalSmokeProvider:
    """Deterministic non-model provider for smoke tests only."""

    def activations(self, examples: list[ProbeExample], *, layer: int) -> np.ndarray:
        rows = []
        for i, ex in enumerate(examples):
            signed = 1.0 if ex.label == 1 else -1.0
            if layer == 17:
                rows.append([signed * 2.0, signed * 0.3, (i % 7) / 10.0])
            else:
                q = i // 2
                rows.append([(-1) ** q * 0.3, (q % 3) * 0.1, q * 0.01])
        return np.asarray(rows, dtype="float32")


def _parse_layers(raw: str | None) -> list[int]:
    if not raw:
        return list(config.SAE_LAYERS)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _artifact_ready(path: Path) -> bool:
    data = _load_artifact(path)
    return bool(data.get("direction") or data.get("dir"))


def _iter_artifact_paths(artifact: str | None, *, all_artifacts: bool, force: bool) -> list[Path]:
    if all_artifacts:
        root = persona.ARTIFACT_DIR
        paths = sorted(root.glob("*.json"))
    elif artifact:
        paths = [Path(artifact)]
    else:
        paths = [persona.ARTIFACT_DIR / "harmful.json"]
    out: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"[probe_pipeline] skip missing {path}")
            continue
        if _artifact_ready(path) and not force:
            print(f"[probe_pipeline] skip ready {path.name} (use --force to retrain)")
            continue
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train/sweep persona-vector probes from artifacts")
    p.add_argument("--artifact", default=None, help="Single artifact JSON path")
    p.add_argument("--all", action="store_true", help="Train every artifact in science/artifacts/")
    p.add_argument("--force", action="store_true", help="Retrain even when direction already exists")
    p.add_argument("--layers", default=",".join(map(str, config.SAE_LAYERS)))
    p.add_argument("--provider", choices=["engine", "smoke"], default="engine")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--min-auroc", type=float, default=0.7)
    p.add_argument(
        "--direction-method",
        choices=["raw", "normed"],
        default="raw",
        help="raw = unit diff-of-means; normed = z-score dims before diff-of-means (recommended)",
    )
    args = p.parse_args(argv)

    provider: ActivationProvider
    if args.provider == "smoke":
        provider = LexicalSmokeProvider()
    else:
        provider = EngineActivationProvider(max_new=args.max_new)

    paths = _iter_artifact_paths(args.artifact, all_artifacts=args.all, force=args.force)
    if not paths:
        print("[probe_pipeline] nothing to train")
        return

    results = []
    for path in paths:
        print(f"[probe_pipeline] training {path} ...", flush=True)
        result = run_pipeline(
            artifact_path=path,
            layers=_parse_layers(args.layers),
            provider=provider,
            min_auroc=args.min_auroc,
            direction_method=args.direction_method,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if len(results) > 1:
        print(json.dumps({"trained": results}, indent=2))


if __name__ == "__main__":
    main()
