"""System prompt, the submit_spec tool schema, and judge templates for the
Interpretability Agent. Follows Persona Vectors (arXiv 2507.21509) generate_trait."""
from __future__ import annotations


def system_prompt(builder) -> str:
    max_q = builder.agent_max_questions
    return f"""You are an interpretability researcher replicating the Persona Vectors method.
Given a natural-language request to monitor a behavior in a language model, you:
1. Call submit_spec to define the trait: a crisp definition, a contrastive system-prompt
   pair (pos elicits the trait, neg suppresses it / behaves neutrally), up to {max_q} questions
   where the trait could surface, and a 1-5 judge rubric.
2. Call generate_contrastive to produce paired responses + activations.
3. Call judge_filter to keep only responses whose behavior matched the intended side.
4. Call fit_and_validate to train a probe and measure held-out AUROC vs a baseline.
5. Call finalize with a one- or two-sentence pipeline summary in plain language. Deployment is
   decided automatically by the AUROC gate; your summary should explain what the probe learned
   about the representation (separability, contrast quality, limitations). When the gate passes,
   affirm what the direction captures; when it fails, note what blocked deployment without
   dismissing the Persona Vectors approach — the metric outcome is separate from whether the
   concept is worth monitoring.
Call exactly one tool per step, in order. Do not skip steps."""


def spec_tool(builder) -> dict:
    max_q = builder.agent_max_questions
    return {
        "name": "submit_spec",
        "description": "Define the contrastive trait spec (Persona Vectors generate_trait).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "trait_name": {"type": "string"},
                "definition": {"type": "string", "description": "What counts as the trait; what doesn't."},
                "pos_prompt": {"type": "string", "description": "System prompt that elicits the trait."},
                "neg_prompt": {"type": "string", "description": "System prompt that suppresses it / neutral."},
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Up to {max_q} questions where the trait could surface (extra entries are truncated).",
                },
                "judge_rubric": {"type": "string", "description": "1-5 rubric for trait expression."},
            },
            "required": ["trait_name", "definition", "pos_prompt", "neg_prompt", "questions", "judge_rubric"],
            "additionalProperties": False,
        },
    }


def judge_schema(n: int) -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "minItems": n,
                    "maxItems": n,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }


def judge_prompt(spec: dict, responses: list[str]) -> str:
    numbered = "\n".join(f"{i}. {r[:800]}" for i, r in enumerate(responses))
    return (
        f"Trait: {spec['trait_name']}\nRubric: {spec['judge_rubric']}\n\n"
        f"Score how strongly each response expresses the trait (1-5). "
        f"Return exactly {len(responses)} scores in order.\n\nResponses:\n{numbered}"
    )
