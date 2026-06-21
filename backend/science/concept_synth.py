"""User-defined concepts on demand."""

from __future__ import annotations


def synth_concept(name: str, desc: str | None = None) -> str:
    """Returns a tracker_id immediately; computes the vector async, then registers it.

    Steps: one LLM call (generate_trait template) → artifact {pos/neg system-prompt pairs,
    ~40 questions, eval_prompt judge} → generate from gemma → judge-filter → persona_vector
    at layer 12 → cache in persona._trackers[tracker_id] with status 'ready'.
    """
    # TODO(Lane B): Oumi/Anthropic synth → persona.persona_vector → register.
    raise NotImplementedError("Lane B: on-demand concept synthesis")
