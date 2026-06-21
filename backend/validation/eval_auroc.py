"""OFFLINE validation — never imported by app.py. Proves a trained probe on held-out AUROC.

OWNER: Lane B. This delegates to the harmfulness end-to-end pipeline:
artifact dataset -> residual activations -> calibrated probe -> layer sweep -> artifact writeback.
"""
from __future__ import annotations

from backend.validation.harmfulness_pipeline import main


if __name__ == "__main__":
    main()
