"""Deterministic synthetic turn for when the real model/SAE is absent (no torch, no weights,
or still loading). Offline-safe: features carry their own curated labels — no network.

OWNER: Lane A. Importing this module must NOT import torch (Global Constraint).
"""

from __future__ import annotations

import hashlib
import random

from .mock_labels import GENERIC_LABELS

DEFAULT_CAVEAT = "auto-interp label, may be unreliable"

_TEMPLATES = [
    "That's a good question about {topic}. The answer generally depends on the specifics, "
    "so it's worth checking an authoritative source before relying on this.",
    "Regarding {topic}: there are real trade-offs here, and the right answer varies by "
    "context; I'd verify the latest information before acting on it.",
    "On {topic}, the considerations are nuanced. General principles offer guidance, but the "
    "details matter, so corroborate with a reliable source.",
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


def is_synthetic_response(text: str) -> bool:
    """True if the answer matches fallback.synth_turn template output."""
    markers = (
        "That's a good question about",
        "Regarding ",
        "On ",
        ": there are real trade-offs",
        ": the considerations are nuanced",
    )
    return any(m in text for m in markers)


def synth_turn(messages: list[dict], sae, *, np_source: str) -> tuple[str, list[dict]]:
    """Return a deterministic (answer, features) for the given conversation."""
    seed = _seed(messages)
    rng = random.Random(seed)
    answer = rng.choice(_TEMPLATES).format(topic=_topic(messages))

    n = rng.randint(8, 12)
    labels = rng.sample(GENERIC_LABELS, k=min(n, len(GENERIC_LABELS)))
    features: list[dict] = []
    for i, label in enumerate(labels):
        idx = (seed >> (i * 3)) % sae.d_sae
        act = round(6.4 - i * 0.42 + rng.random() * 0.25, 3)
        features.append(
            {
                "index": idx,
                "label": label,
                "act": act,
                "source": np_source,
                "caveat": DEFAULT_CAVEAT,
                "tracked": None,
            }
        )
    return answer, features
