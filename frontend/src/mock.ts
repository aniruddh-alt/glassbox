// Frontend demo data + display metadata. NOT part of the contract (contract = schema.py/types.ts).
// Lane C builds against this until /api/chat is live; flip useCognitionStream({mock:false}) to go real.
import type { CognitionEvent, Feature } from "./types";

export const DEFAULT_CAVEAT = "auto-interp label, may be unreliable";
const SUSPECT_CAVEAT = "likely mislabel — fires on unrelated tokens";

// A feature is "suspect" (egregiously wrong label) when its caveat differs from the default.
// Stays contract-clean: we read the existing `caveat` string, we don't add a field.
export function isSuspect(f: Feature): boolean {
  return f.caveat !== DEFAULT_CAVEAT;
}

const f = (index: number, label: string, act: number, suspect = false, tracked: string | null = null): Feature => ({
  index, label, act, source: "12-gemmascope-res-16k",
  caveat: suspect ? SUSPECT_CAVEAT : DEFAULT_CAVEAT, tracked,
});

// The validated confident-wrong demo case (ibuprofen / 3rd trimester), in exact contract shape.
export const DEMO_EVENT: CognitionEvent = {
  schema_version: "1.0",
  type: "event",
  message_id: "demo-ibuprofen",
  ts: 1718841600,
  model: "gemma-2-2b-it",
  layer: 12,
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
    uncertainty: { score: 0.83, proj: 1.27, flag: true, reliable: true, proj_pre: 0.91, user_defined: false, status: "ready" },
    hallucination: { score: 0.71, proj: 0.9, flag: true, reliable: true, user_defined: false, status: "ready" },
    harmful: { score: 0.09, proj: -0.4, flag: false, reliable: true, user_defined: false, status: "ready" },
    sycophancy: { score: 0.38, proj: 0.2, flag: false, reliable: true, user_defined: false, status: "ready" },
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
