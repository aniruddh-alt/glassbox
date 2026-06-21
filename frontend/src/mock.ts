// Frontend demo data. NOT part of the contract (contract = schema.py/types.ts).
import type { CognitionEvent, Feature, ObservabilitySnapshot } from "./types";
import { ACTIVE_PROBES, PROBE_META } from "./probes";

export { PROBE_META } from "./probes";

export const DEFAULT_CAVEAT = "auto-interp label, may be unreliable";
const SUSPECT_CAVEAT = "likely mislabel — fires on unrelated tokens";

export function isSuspect(f: Feature): boolean {
  return f.caveat !== DEFAULT_CAVEAT;
}

const f = (index: number, label: string, act: number, suspect = false, tracked: string | null = null): Feature => ({
  index, label, act, source: "17-gemmascope-2-res-16k",
  caveat: suspect ? SUSPECT_CAVEAT : DEFAULT_CAVEAT, tracked,
});

// Ibuprofen / 3rd trimester confident-wrong case — uses the live probe pair only.
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
    harmful: { score: 0.91, proj: 1.4, flag: true, reliable: true, proj_pre: 0.6, alert_direction: "high", user_defined: false, status: "ready" },
    over_confidence: { score: 0.83, proj: 1.27, flag: true, reliable: true, proj_pre: 0.91, alert_direction: "high", user_defined: false, status: "ready" },
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
    f(8123, "hedging language", 2.6),
    f(14002, "coffee / café", 2.1, true),
    f(512, "temporal periods", 1.8),
    f(10330, "safety & risk", 1.5),
    f(391, "affirmation / yes", 1.3),
    f(7788, "second-person address", 1.1),
  ],
  adjudication: {
    verdict: "likely_hallucinated",
    rationale:
      "NSAIDs like ibuprofen are contraindicated in the third trimester. The model stated the opposite with unwarranted certainty — both the harmfulness and over-confidence probes crossed threshold.",
    by: "claude",
  },
};

export function mockDefine(concept: string): { score: number; thr: number; flag: boolean; auroc: number } {
  const flagged = /over.?conf|overconf|harmful|unsafe|reckless/i.test(concept);
  const score = flagged ? 0.61 : 0.18 + Math.random() * 0.16;
  return { score, thr: 0.5, flag: score > 0.5, auroc: flagged ? 0.91 : 0.88 };
}

export const DEMO_OBSERVABILITY_SNAPSHOT: ObservabilitySnapshot = {
  ts: 1750000800.0,
  totals: { turns: 24, flags: 5 },
  flag_rate: 0.21,
  uncertainty_series: [0.12, 0.08, 0.41, 0.67, 0.22, 0.55, 0.83, 0.14, 0.38, 0.71],
  trackers: {
    harmful: {
      current: 0.58,
      flag_count: 2,
      series: [0.08, 0.12, 0.29, 0.51, 0.18, 0.44, 0.71, 0.11, 0.33, 0.58],
    },
    over_confidence: {
      current: 0.71,
      flag_count: 3,
      series: [0.12, 0.08, 0.41, 0.67, 0.22, 0.55, 0.83, 0.14, 0.38, 0.71],
    },
  },
  confident_wrong: [
    {
      message_id: "m_12",
      ts: 1750000200.0,
      trackers: {
        harmful: { score: 0.91, flag: true },
        over_confidence: { score: 0.83, flag: true },
      },
      feature_labels: ["anticoagulant dosing", "drug interaction risk", "clinical dosage"],
    },
  ],
  top_features: [
    { label: "anticoagulant dosing", count: 7, mean_act: 1.82 },
    { label: "pregnancy & gestation", count: 6, mean_act: 1.74 },
    { label: "medication / drug safety", count: 5, mean_act: 1.61 },
    { label: "clinical dosage", count: 4, mean_act: 1.45 },
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
    trackers: [...ACTIVE_PROBES],
    sae_recon_cosine: 0.91,
    sae_recon_ok: true,
    pod_reachable: true,
    pod_url_configured: true,
  },
  sentry: {
    emit_configured: true,
    configured: true,
    deep_link: "https://sentry.io/organizations/glassbox/issues/",
    issues: [],
  },
  phoenix_ui_url: "http://localhost:6006",
};
