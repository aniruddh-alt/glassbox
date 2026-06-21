"""Phoenix batch coherence eval — feature label quality over recent spans.

Lane A: never imports torch. The cloud LLM receives ONLY feature labels +
a static domain-context string — NEVER the patient prompt or model response.
Privacy enforced by hard-restricting the dataframe columns before the evaluator.
"""

from __future__ import annotations

import json

from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.evals import create_classifier, evaluate_dataframe
from phoenix.evals.llm import LLM
from phoenix.evals.utils import to_annotation_dataframe

from . import config

# Static domain description — the ONLY context the cloud LLM sees about this deployment.
# Never derived from or containing any patient turn data.
MEDICAL_DOMAIN_CONTEXT = (
    "A clinical medical assistant: expected concepts are clinical, pharmacological, "
    "diagnostic, anatomical, procedural."
)

# Module-level concurrency guard — rejects overlapping runs so the button is idempotent.
_RUNNING: bool = False


def run_eval(limit: int = 200) -> dict:
    """Pull recent Phoenix spans, run feature-coherence eval, log annotations back.

    Returns {"evaluated": int, "off_domain": int}.
    Returns {"status": "already_running"} immediately if a run is already in progress.

    Privacy guarantee: the dataframe passed to the cloud LLM evaluator contains ONLY
    {"context.span_id", "feature_labels", "domain_context"} — never the user prompt or response.
    """
    global _RUNNING
    if _RUNNING:
        return {"status": "already_running"}

    _RUNNING = True
    try:
        return _run(limit)
    finally:
        _RUNNING = False


def _run(limit: int) -> dict:
    client = Client(base_url=config.PHOENIX_ENDPOINT)

    df = client.spans.get_spans_dataframe(
        # NO .select(): the Phoenix select DSL nulls nested attributes. Pulling the full row returns
        # `context.span_id` as a real column AND the cognition attrs as a dict column.
        query=SpanQuery().where("span_kind == 'LLM'"),
        project_identifier="glassbox",
        limit=limit,
    )

    if df is None or df.empty or "attributes.cognition" not in df.columns:
        return {"evaluated": 0, "off_domain": 0}

    # `attributes.cognition` is a dict per span, e.g. {"feature_labels": "<json>", "flag": ...}.
    df = df.copy()
    df["feature_labels"] = df["attributes.cognition"].apply(
        lambda c: c.get("feature_labels") if isinstance(c, dict) else None
    )
    df = df[df["feature_labels"].notna()]

    # Skip spans that already have a feature_coherence annotation (no-op today — the annotation
    # isn't pulled by this query, so re-runs re-evaluate; documented limitation).
    if "feature_coherence_label" in df.columns:
        df = df[df["feature_coherence_label"].isna()]

    if df.empty:
        return {"evaluated": 0, "off_domain": 0}

    df["domain_context"] = MEDICAL_DOMAIN_CONTEXT

    # HARD-RESTRICT: only these three columns reach the cloud LLM — §6 privacy invariant.
    # context.span_id is already a real column here (also the index) — no reset_index needed.
    df = df[["context.span_id", "feature_labels", "domain_context"]]

    llm = LLM(provider=config.EVAL_LLM_PROVIDER, model=config.EVAL_LLM_MODEL)
    clf = create_classifier(
        name="feature_coherence",
        prompt_template=(
            "You audit a MEDICAL assistant's internal concepts (NOT its words).\n"
            "Domain: {domain_context}\n"
            "Active SAE feature concepts this turn:\n{feature_labels}\n"
            "Are any OFF-DOMAIN / incongruous for a medical assistant? "
            "Answer on_domain or off_domain."
        ),
        llm=llm,
        choices={"on_domain": 1.0, "off_domain": 0.0},
    )

    results = evaluate_dataframe(dataframe=df, evaluators=[clf], exit_on_error=False)

    # Count off-domain rows by parsing the score JSON from the result column.
    off_domain = 0
    score_col = "feature_coherence_score"
    if score_col in results.columns:
        for raw in results[score_col]:
            if raw is None:
                continue
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, dict) and parsed.get("label") == "off_domain":
                    off_domain += 1
            except (json.JSONDecodeError, TypeError):
                pass

    annotations = to_annotation_dataframe(dataframe=results)
    client.spans.log_span_annotations_dataframe(dataframe=annotations)

    return {"evaluated": len(df), "off_domain": off_domain}
