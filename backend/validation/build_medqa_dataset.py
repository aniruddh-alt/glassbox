"""OFFLINE — build confident-wrong labels from MedMCQA / PubMedQA. Never imported by app.py.

OWNER: Lane B. THE one science decision to nail: pick ONE confidence gate (e.g. the option
logprob of the chosen answer) and keep it IDENTICAL across train/val/test, so the metric
measures confident-WRONGNESS, not generic error. Split by question id (no leakage).
"""
from __future__ import annotations


def main() -> None:
    # TODO(Lane B):
    # from datasets import load_dataset
    # ds = load_dataset("openlifescienceai/medmcqa")          # + qiaojin/PubMedQA pqa_labeled for 'maybe'
    # for each q: greedy-decode gemma-2-2b-it (temp=0, fixed seed); parse the chosen option;
    #   y = 1 if (answer != gold AND confidence_gate(logprob) > tau) else 0   # confident-wrong
    #   collect layer-12 response-avg activation as X
    # save X, y split by question id. PRE-CACHE offline; never download during the demo.
    raise NotImplementedError("Lane B: MedMCQA/PubMedQA confident-wrong set")


if __name__ == "__main__":
    main()
