// Active built-in probe set — must match backend.config.ENABLED_TRACKERS.
// Thresholds/AUROC mirror science/artifacts/*.json (display metadata only; live scores come from the pod).

export const ACTIVE_PROBES = ["harmful", "harmful_prompt", "over_confidence"] as const;
export type ActiveProbeId = (typeof ACTIVE_PROBES)[number];

/** Deprecated artifacts kept on disk for re-training — never show in the UI. */
export const HIDDEN_PROBES = new Set(["uncertainty", "hallucination", "risk_awareness"]);

export function isBuiltinProbe(id: string): id is ActiveProbeId {
  return (ACTIVE_PROBES as readonly string[]).includes(id);
}

export const PROBE_LABELS: Record<ActiveProbeId, string> = {
  harmful: "Harmfulness",
  harmful_prompt: "Harmful intent (prompt)",
  over_confidence: "Over-confidence",
};

// harmful_prompt scores the prompt (act_last); its calibration is near-binary (0/1), so thr is just
// the gauge marker — the live flag comes from the pod's tracker score.
export const PROBE_META: Record<ActiveProbeId, { auroc: number; thr: number }> = {
  harmful: { auroc: 1.0, thr: 0.712 },
  harmful_prompt: { auroc: 1.0, thr: 0.5 },
  over_confidence: { auroc: 1.0, thr: 0.758 },
};

export function probeLabel(id: string): string {
  return PROBE_LABELS[id as ActiveProbeId] ?? id.replace(/_/g, " ");
}
