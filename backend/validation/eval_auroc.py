"""OFFLINE validation — never imported by app.py. Proves the probe beats a plain baseline.

OWNER: Lane B. Honesty requirement: show AUROC of the calibrated persona/probe vs a plain
logistic-regression-on-activations baseline, plus a reliability diagram. A judge WILL ask
"does this beat a linear probe?" — the answer must be measured, not asserted.
"""
from __future__ import annotations


def main() -> None:
    # TODO(Lane B):
    # 1. load activations + confident-wrong labels from build_medqa_dataset.py
    # 2. fit calibrated probe + a plain LogisticRegression baseline on the SAME features
    # 3. report AUROC for both on held-out split; print a reliability diagram
    # 4. NEVER present paper AUROC numbers as your own — rerun on this model+data.
    raise NotImplementedError("Lane B: AUROC probe vs baseline")


if __name__ == "__main__":
    main()
