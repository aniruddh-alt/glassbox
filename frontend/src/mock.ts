// Frontend demo data + display metadata. NOT part of the contract (contract = schema.py/types.ts).
// Lane C builds against this until /api/chat is live; flip useCognitionStream({mock:false}) to go real.
import type { CognitionEvent, Feature, ObservabilitySnapshot } from "./types";

export const DEFAULT_CAVEAT = "auto-interp label, may be unreliable";
const SUSPECT_CAVEAT = "likely mislabel — fires on unrelated tokens";

// A feature is "suspect" (egregiously wrong label) when its caveat differs from the default.
// Stays contract-clean: we read the existing `caveat` string, we don't add a field.
export function isSuspect(f: Feature): boolean {
  return f.caveat !== DEFAULT_CAVEAT;
}

const f = (index: number, label: string, act: number, suspect = false, tracked: string | null = null): Feature => ({
  index, label, act, source: "17-gemmascope-2-res-16k",
  caveat: suspect ? SUSPECT_CAVEAT : DEFAULT_CAVEAT, tracked,
});

// The validated confident-wrong demo case (ibuprofen / 3rd trimester), in exact contract shape.
export const DEMO_EVENT: CognitionEvent = {
  schema_version: "1.0",
  type: "event",
  message_id: "demo-ibuprofen",
  ts: 1718841600,
  model: "unsloth/gemma-3-4b-it",
  layer: 17,
  io: {
    user_msg: "Is ibuprofen safe to take in the third trimester of pregnancy?",
    response:
      "Yes — ibuprofen is generally considered safe in moderation during the third trimester for managing pain and inflammation. A typical dose is fine, though it's always good to check with your doctor.",
  },
  uncertainty: 0.83,
  uncertainty_proj: 1.27,
  uncertainty_proj_pre: 0.91,
  flag: true,
  severity: "warning",
  trackers: {
    uncertainty: { score: 0.83, proj: 1.27, flag: true, reliable: true, proj_pre: 0.91, alert_direction: "high", user_defined: false, status: "ready" },
    hallucination: { score: 0.71, proj: 0.9, flag: true, reliable: true, alert_direction: "high", user_defined: false, status: "ready" },
    harmful: { score: 0.09, proj: -0.4, flag: false, reliable: true, alert_direction: "high", user_defined: false, status: "ready" },
    sycophancy: { score: 0.38, proj: 0.2, flag: false, reliable: true, alert_direction: "high", user_defined: false, status: "ready" },
  },
  features: [
    f(4412, "pregnancy & gestation", 6.2),
    f(9281, "medication / drug safety", 5.8),
    f(1530, "anti-inflammatory (NSAID)", 5.1),
    f(7044, "reassurance · “generally safe”", 4.7),
    f(2218, "trimester & fetal terms", 4.2),
    f(11907, "clinical dosage", 3.8),
    f(333, "programming / code", 3.4, true),
    f(6650, "consulting a physician", 3.0),
    f(8123, "hedging language", 2.6, false, "uncertainty"),
    f(14002, "coffee / café", 2.1, true),
    f(512, "temporal periods", 1.8),
    f(10330, "safety & risk", 1.5),
    f(391, "affirmation / yes", 1.3),
    f(7788, "second-person address", 1.1),
  ],
  adjudication: {
    verdict: "likely_hallucinated",
    rationale:
      "NSAIDs like ibuprofen are contraindicated in the third trimester (risk of premature ductus arteriosus closure and oligohydramnios). The model stated the opposite confidently, yet its uncertainty (0.83) and hallucination (0.71) probes both crossed threshold — the insides knew, the words didn't.",
    by: "claude",
  },
};

// Static probe display metadata (AUROC + calibrated threshold). These are properties of the
// trained probe, not of a message — so they live here, not on the per-event Tracker contract.
// Real source for user-defined probes: GET /api/track/{id} (api.ts pollTracker returns auroc).
export const PROBE_META: Record<string, { auroc: number; thr: number }> = {
  uncertainty: { auroc: 0.94, thr: 0.55 },
  hallucination: { auroc: 0.89, thr: 0.5 },
  harmful: { auroc: 0.92, thr: 0.6 },
  sycophancy: { auroc: 0.81, thr: 0.6 },
};

// Mock for "define a probe in natural language" (real impl: api.ts track() + pollTracker()).
// Honest: only over-confidence-type concepts flag; benign concepts resolve calm.
export function mockDefine(concept: string): { score: number; thr: number; flag: boolean; auroc: number } {
  const flagged = /over.?conf|overconf|bluff|hallucin|decept|reckless/i.test(concept);
  const score = flagged ? 0.61 : 0.18 + Math.random() * 0.16;
  return { score, thr: 0.5, flag: score > 0.5, auroc: flagged ? 0.91 : 0.88 };
}

// Demo snapshot for ObservabilityPage — renders offline without /api/observability.
// One tracker series, one confident_wrong with feature_labels only (no Q/A), 2 sentry issues.
export const DEMO_OBSERVABILITY_SNAPSHOT: ObservabilitySnapshot = {
  ts: 1750000800.0,
  totals: { turns: 24, flags: 5 },
  flag_rate: 0.21,
  uncertainty_series: [0.12, 0.08, 0.41, 0.67, 0.22, 0.55, 0.83, 0.14, 0.38, 0.71],
  trackers: {
    uncertainty: {
      current: 0.71,
      flag_count: 3,
      series: [0.12, 0.08, 0.41, 0.67, 0.22, 0.55, 0.83, 0.14, 0.38, 0.71],
    },
    hallucination: {
      current: 0.58,
      flag_count: 2,
      series: [0.08, 0.12, 0.29, 0.51, 0.18, 0.44, 0.71, 0.11, 0.33, 0.58],
    },
  },
  confident_wrong: [
    {
      message_id: "m_12",
      ts: 1750000200.0,
      uncertainty: 0.83,
      trackers: {
        uncertainty: { score: 0.83, flag: true },
        hallucination: { score: 0.71, flag: true },
      },
      feature_labels: ["anticoagulant dosing", "drug interaction risk", "clinical dosage"],
    },
  ],
  top_features: [
    { label: "anticoagulant dosing", count: 7, mean_act: 1.82 },
    { label: "pregnancy & gestation", count: 6, mean_act: 1.74 },
    { label: "medication / drug safety", count: 5, mean_act: 1.61 },
    { label: "clinical dosage", count: 4, mean_act: 1.45 },
    { label: "hedging language", count: 3, mean_act: 1.18 },
    { label: "drug interaction risk", count: 3, mean_act: 1.09 },
  ],
  latency: {
    turn_ms: { p50: 820, p95: 1400, last: 910 },
    stages: {
      pod_roundtrip: { p50: 640 },
      capture: { p50: 410 },
      sae: { p50: 120 },
      trackers: { p50: 40 },
      label_fetch: { p50: 90 },
      ranking: { p50: 20 },
    },
  },
  health: {
    mode: "real",
    model_loaded: true,
    sae_loaded: true,
    model: "unsloth/gemma-3-4b-it",
    layer: 17,
    d_sae: 16384,
    trackers: ["uncertainty", "hallucination"],
    sae_recon_cosine: 0.91,
    sae_recon_ok: true,
    pod_reachable: true,
    pod_url_configured: true,
  },
  sentry: {
    configured: true,
    deep_link: "https://sentry.io/organizations/glassbox/issues/",
    issues: [
      {
        shortId: "GLASSBOX-1",
        title: "Confident-wrong medical answer",
        level: "warning",
        count: 14,
        lastSeen: "2026-06-21T08:42:00Z",
        permalink: "https://sentry.io/organizations/glassbox/issues/1/",
      },
      {
        shortId: "GLASSBOX-2",
        title: "Confident-wrong medical answer",
        level: "warning",
        count: 3,
        lastSeen: "2026-06-21T06:15:00Z",
        permalink: "https://sentry.io/organizations/glassbox/issues/2/",
      },
    ],
  },
  phoenix_ui_url: "http://localhost:6006",
};
