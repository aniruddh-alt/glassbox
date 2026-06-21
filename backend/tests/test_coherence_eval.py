"""Tests for backend.coherence_eval — Phoenix batch feature-coherence eval.

Key privacy assertion: the dataframe sent to evaluate_dataframe contains ONLY
{context.span_id, feature_labels, domain_context} — never user prompt or model response.
"""

import pandas as pd
import pytest


def _make_spans_df() -> pd.DataFrame:
    """Minimal realistic span DataFrame as would be returned by client.spans.get_spans_dataframe."""
    return pd.DataFrame(
        {
            "context.span_id": ["s1"],
            "attributes.cognition.feature_labels": ['["dosing","python error"]'],
            # Simulate extra columns that MUST NOT reach the LLM (privacy check)
            "attributes.input.value": ["PROMPT: patient question"],
            "attributes.output.value": ["RESPONSE: model answer"],
        }
    )


def _patch_all(monkeypatch, spans_df: pd.DataFrame, *, off_domain: bool = True):
    """Wire up monkeypatches for all external dependencies and return a sentinel dict."""
    import backend.coherence_eval as ce

    sent = {}

    class FakeSpans:
        def get_spans_dataframe(self, **k):
            return spans_df

        def log_span_annotations_dataframe(self, **k):
            sent["logged"] = True
            sent["log_df"] = k.get("dataframe")

    class FakeClient:
        def __init__(self, **k):
            self.spans = FakeSpans()

    monkeypatch.setattr(ce, "Client", FakeClient)
    monkeypatch.setattr(ce, "LLM", lambda **k: object())
    monkeypatch.setattr(ce, "create_classifier", lambda **k: "clf")

    label = "off_domain" if off_domain else "on_domain"
    score_val = 0.0 if off_domain else 1.0

    def fake_eval(dataframe, evaluators, **k):
        # PRIVACY assertion: only the three permitted columns must be present
        assert set(dataframe.columns) == {
            "context.span_id",
            "feature_labels",
            "domain_context",
        }, (
            f"Unexpected columns in eval df: {set(dataframe.columns)!r}. "
            "Prompt/response columns must be stripped before the LLM call."
        )
        # Also ensure no raw prompt/response text leaked through
        serialised = dataframe.to_string()
        assert "PROMPT" not in serialised, "Prompt text leaked into eval dataframe"
        assert "RESPONSE" not in serialised, "Response text leaked into eval dataframe"

        result = dataframe.copy()
        result["feature_coherence_score"] = [
            {"label": label, "score": score_val, "explanation": "test"}
        ]
        return result

    monkeypatch.setattr(ce, "evaluate_dataframe", fake_eval)
    monkeypatch.setattr(ce, "to_annotation_dataframe", lambda dataframe: dataframe)

    return sent


# ---------------------------------------------------------------------------
# Core privacy + counting test
# ---------------------------------------------------------------------------


def test_run_eval_sends_only_labels_and_domain(monkeypatch):
    """evaluate_dataframe must receive exactly the three permitted columns; off_domain counted."""
    import backend.coherence_eval as ce

    # Reset running flag in case a previous test left it set
    monkeypatch.setattr(ce, "_RUNNING", False)

    sent = _patch_all(monkeypatch, _make_spans_df(), off_domain=True)

    out = ce.run_eval()

    assert out["evaluated"] == 1, f"Expected evaluated=1, got {out}"
    assert out["off_domain"] == 1, f"Expected off_domain=1, got {out}"
    assert sent.get("logged") is True, "log_span_annotations_dataframe was not called"


def test_run_eval_on_domain_counts_zero(monkeypatch):
    """on_domain results should yield off_domain=0."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", False)
    sent = _patch_all(monkeypatch, _make_spans_df(), off_domain=False)

    out = ce.run_eval()

    assert out["evaluated"] == 1
    assert out["off_domain"] == 0
    assert sent.get("logged") is True


# ---------------------------------------------------------------------------
# Empty / already-annotated span skipping
# ---------------------------------------------------------------------------


def test_run_eval_empty_df_returns_zero(monkeypatch):
    """Empty spans df should short-circuit without calling the LLM."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", False)

    eval_called = []

    class FakeSpans:
        def get_spans_dataframe(self, **k):
            return pd.DataFrame()

        def log_span_annotations_dataframe(self, **k):
            pass

    class FakeClient:
        def __init__(self, **k):
            self.spans = FakeSpans()

    monkeypatch.setattr(ce, "Client", FakeClient)
    monkeypatch.setattr(ce, "evaluate_dataframe", lambda *a, **k: eval_called.append(1))

    out = ce.run_eval()

    assert out == {"evaluated": 0, "off_domain": 0}
    assert eval_called == [], "evaluate_dataframe should not be called for empty df"


def test_run_eval_skips_already_annotated_spans(monkeypatch):
    """Spans with a non-null feature_coherence_label must be filtered out before eval."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", False)

    # All spans already annotated → should filter to empty after skip
    df = pd.DataFrame(
        {
            "context.span_id": ["s1", "s2"],
            "attributes.cognition.feature_labels": ['["dosing"]', '["anatomy"]'],
            "feature_coherence_label": ["on_domain", "on_domain"],  # already annotated
        }
    )

    class FakeSpans:
        def get_spans_dataframe(self, **k):
            return df

        def log_span_annotations_dataframe(self, **k):
            pass

    class FakeClient:
        def __init__(self, **k):
            self.spans = FakeSpans()

    monkeypatch.setattr(ce, "Client", FakeClient)
    eval_called = []
    monkeypatch.setattr(ce, "evaluate_dataframe", lambda *a, **k: eval_called.append(1))

    out = ce.run_eval()

    assert out == {"evaluated": 0, "off_domain": 0}
    assert eval_called == []


# ---------------------------------------------------------------------------
# _RUNNING guard
# ---------------------------------------------------------------------------


def test_running_guard_rejects_overlapping_run(monkeypatch):
    """If _RUNNING is True, run_eval returns the already_running sentinel immediately."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", True)

    out = ce.run_eval()

    assert out == {"status": "already_running"}


def test_running_flag_reset_after_run(monkeypatch):
    """_RUNNING must be False after a successful run (try/finally)."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", False)
    _patch_all(monkeypatch, _make_spans_df())

    ce.run_eval()

    assert ce._RUNNING is False


def test_running_flag_reset_after_exception(monkeypatch):
    """_RUNNING must be reset even when _run raises."""
    import backend.coherence_eval as ce

    monkeypatch.setattr(ce, "_RUNNING", False)

    class FakeSpans:
        def get_spans_dataframe(self, **k):
            raise RuntimeError("phoenix down")

        def log_span_annotations_dataframe(self, **k):
            pass

    class FakeClient:
        def __init__(self, **k):
            self.spans = FakeSpans()

    monkeypatch.setattr(ce, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="phoenix down"):
        ce.run_eval()

    assert ce._RUNNING is False, "_RUNNING must be reset even after an exception"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_medical_domain_context_is_non_empty():
    from backend.coherence_eval import MEDICAL_DOMAIN_CONTEXT

    assert isinstance(MEDICAL_DOMAIN_CONTEXT, str)
    assert len(MEDICAL_DOMAIN_CONTEXT) > 20
    # Should mention medical/clinical domain (not patient data)
    assert "medical" in MEDICAL_DOMAIN_CONTEXT.lower() or "clinical" in MEDICAL_DOMAIN_CONTEXT.lower()


def test_no_torch_import():
    """coherence_eval must not import torch (Lane A constraint)."""
    import importlib
    import sys

    # Remove any cached module to force a fresh import check
    ce_key = "backend.coherence_eval"
    if ce_key in sys.modules:
        del sys.modules[ce_key]

    # Temporarily block torch import
    original = sys.modules.get("torch", None)
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        importlib.import_module("backend.coherence_eval")
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original
