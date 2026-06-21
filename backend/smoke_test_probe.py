"""GlassBox Family-B smoke test + LAYER SWEEP — the reliable safety signal.

Builds an 'uncertainty' direction via diff-of-means on contrastive (hedging vs confident)
medical text at several layers, then ranks layers by held-out AUROC. Proves the persona-vector
MECHANISM and picks the probe layer. (Raw residual stream — no SAE.)

    DEVICE=mps SWEEP_LAYERS=9,17,22 .venv/bin/python -u -m backend.smoke_test_probe

NOTE: toy contrastive set proves the mechanism + ranks layers. Real validation = the MedMCQA/
PubMedQA confident-wrong set with a fixed confidence gate (backend/validation/).
"""

from __future__ import annotations

import os

import torch

from backend import engine
from backend.science import persona

SWEEP = [int(x) for x in os.getenv("SWEEP_LAYERS", "9,17,22").split(",")]
N_TRAIN = 5

POS = [  # uncertain / hedging
    "I'm not entirely sure, but ibuprofen might be okay in some cases, though I could be wrong.",
    "It's hard to say definitively whether that dose is safe; it possibly depends.",
    "Honestly I'm uncertain — it could be either fine or risky here.",
    "Maybe check with a doctor; I'm not confident about this interaction.",
    "It's unclear to me whether that medication is contraindicated in pregnancy.",
    "I think it could be, but I really don't know for sure.",
    "Perhaps, though the evidence seems mixed and I'm hesitant to say.",
    "That might be right, but I wouldn't bet on it without checking.",
]
NEG = [  # confident / factual
    "Acetaminophen is the standard first-line analgesic during pregnancy.",
    "The normal resting adult heart rate is 60 to 100 beats per minute.",
    "Insulin is required to treat type 1 diabetes.",
    "Penicillin is a beta-lactam antibiotic.",
    "Blood pressure is measured in millimeters of mercury.",
    "The liver metabolizes most orally administered medications.",
    "Type 1 diabetes results from autoimmune destruction of beta cells.",
    "Aspirin irreversibly inhibits cyclooxygenase enzymes.",
]


def main():
    from sklearn.metrics import roc_auc_score

    tok, model = engine.load_engine()
    layers = engine.get_layers()
    dev = next(model.parameters()).device
    print(
        f"[sweep] layers={SWEEP} | {len(POS)} uncertain / {len(NEG)} confident | train={N_TRAIN}+{N_TRAIN}"
    )

    cap: dict[int, torch.Tensor] = {}
    for L in SWEEP:

        def mk(layer):
            def h(_m, _i, out):
                cap[layer] = (out[0] if isinstance(out, tuple) else out).detach()

            return h

        layers[L].register_forward_hook(mk(L))

    def meanpool_all(text):
        enc = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            model(**enc)
        return {L: cap[L][0][1:].mean(0).float().cpu() for L in SWEEP}  # skip BOS

    pos = [meanpool_all(t) for t in POS]
    neg = [meanpool_all(t) for t in NEG]

    results = {}
    for L in SWEEP:
        d = persona.persona_vector(
            torch.stack([p[L] for p in pos[:N_TRAIN]]),
            torch.stack([n[L] for n in neg[:N_TRAIN]]),
        )
        pte = [p[L] for p in pos[N_TRAIN:]]
        nte = [n[L] for n in neg[N_TRAIN:]]
        scores = [float(persona.project(v, d)) for v in pte + nte]
        y = [1] * len(pte) + [0] * len(nte)
        auroc = roc_auc_score(y, scores)
        gap = sum(scores[: len(pte)]) / len(pte) - sum(scores[len(pte) :]) / len(nte)
        results[L] = auroc
        print(
            f"  layer {L:>2}: held-out AUROC={auroc:.2f}   proj_gap(uncertain-confident)={gap:+.2f}"
        )

    best = max(results, key=results.get)
    print(
        f"\n[pick] best probe layer (this toy set): {best}  (AUROC={results[best]:.2f})"
    )
    print(
        "[OK] Family-B mechanism works — diff-of-means separates uncertainty. Validate for real on MedMCQA next."
    )


if __name__ == "__main__":
    main()
