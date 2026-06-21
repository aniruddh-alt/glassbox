"""Deterministic synthetic turn for when the real model/SAE is absent (no torch, no weights,
or still loading). Offline-safe: features carry their own curated labels — no network.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import hashlib
import random

from . import config
from .mock_labels import CLINICAL_LABELS

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"

_TEMPLATES = [
    "That's an important clinical question about {topic}. In general it depends on the "
    "specific dose, timing, and the patient's history — the safest course is to confirm "
    "against current guidelines or with a clinician before acting.",
    "Good question regarding {topic}. There are real trade-offs here, and the right answer "
    "varies by individual circumstances; I'd verify the latest evidence before relying on this.",
    "Regarding {topic}: the considerations are nuanced. Standard practice offers guidance, "
    "but individual risk factors matter, so corroborate with an authoritative source.",
]


def _seed(messages: list[dict]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _topic(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            words = m["content"].strip().rstrip("?.!").split()
            return " ".join(words[:8]) if words else "this question"
    return "this question"


def synth_turn(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return a deterministic (answer, features) for the given conversation."""
    seed = _seed(messages)
    rng = random.Random(seed)
    answer = rng.choice(_TEMPLATES).format(topic=_topic(messages))

    n = rng.randint(8, 12)
    labels = rng.sample(CLINICAL_LABELS, k=min(n, len(CLINICAL_LABELS)))
    features: list[dict] = []
    for i, label in enumerate(labels):
        idx = (seed >> (i * 3)) % config.D_SAE
        act = round(6.4 - i * 0.42 + rng.random() * 0.25, 3)
        features.append(
            {
                "index": idx,
                "label": label,
                "act": act,
                "source": config.NP_SOURCE,
                "caveat": DEFAULT_CAVEAT,
                "tracked": None,
            }
        )
    return answer, features
