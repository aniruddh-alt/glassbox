// Sequential interpretability-agent pipeline — mirrors backend/agent/tools.py dispatch order.
// Status values come from concept_synth.update_job on the GPU pod.

export type ProbeJobStatus =
  | "pending"
  | "designing"
  | "generating"
  | "judging"
  | "fitting"
  | "ready"
  | "rejected"
  | "error"
  | "unknown"
  | "unavailable";

export type PipelineStepState = "idle" | "active" | "done" | "error";

export interface ProbeBuildJob {
  tracker_id?: string;
  request?: string;
  status: ProbeJobStatus;
  trait_name?: string | null;
  auroc?: number | null;
  baseline_auroc?: number | null;
  n_kept?: number | null;
  verdict?: string | null;
  error?: string | null;
  progress?: { step: string; pct: number };
}

export const PIPELINE_STAGES = [
  {
    key: "queued",
    match: ["pending"],
    title: "Queue job",
    backend: "POST /api/track",
    detail: "Local backend proxies your natural-language request to the GPU pod. A background interpretability agent job is registered.",
  },
  {
    key: "designing",
    match: ["designing"],
    title: "Design trait spec",
    backend: "submit_spec · Claude Opus",
    detail: "The agent writes a trait name, positive/negative system prompts, and clinical contrast questions that elicit the behavior.",
  },
  {
    key: "generating",
    match: ["generating"],
    title: "Generate contrastive pairs",
    backend: "generate_contrastive · Gemma",
    detail: "For each question the model runs under pos and neg prompts. Layer-17 response-mean activations are captured on the pod.",
  },
  {
    key: "judging",
    match: ["judging"],
    title: "Judge & filter",
    backend: "judge_filter · Claude",
    detail: "Each response is scored 1–5 for trait expression. Only unambiguous positives (≥4) and negatives (≤2) are kept.",
  },
  {
    key: "fitting",
    match: ["fitting"],
    title: "Fit & validate",
    backend: "fit_and_validate · sklearn",
    detail: "A persona-vector direction and calibrated logistic probe are fit. Held-out AUROC is measured against a plain baseline.",
  },
  {
    key: "deploy",
    match: ["ready", "rejected"],
    title: "Deploy gate",
    backend: "finalize · AUROC ≥ 0.75",
    detail: "Probes that clear the gate register into live persona._trackers and persist to science/artifacts/ — scored on the next chat turn.",
  },
] as const;

const STATUS_INDEX: Record<string, number> = {
  pending: 0,
  designing: 1,
  generating: 2,
  judging: 3,
  fitting: 4,
  ready: 5,
  rejected: 5,
};

const STEP_INDEX: Record<string, number> = {
  queued: 0,
  spec: 1,
  generating: 2,
  judging: 3,
  fitting: 4,
  done: 5,
};

export function activeStageIndex(job: ProbeBuildJob | null): number {
  if (!job) return -1;
  const mapped = STATUS_INDEX[job.status];
  if (mapped != null) return mapped;
  if (job.progress?.step) return STEP_INDEX[job.progress.step] ?? -1;
  return -1;
}

export function stepState(index: number, job: ProbeBuildJob | null): PipelineStepState {
  if (!job) return "idle";
  const active = activeStageIndex(job);
  if (job.status === "error") {
    const errAt = active >= 0 ? active : 0;
    if (index < errAt) return "done";
    if (index === errAt) return "error";
    return "idle";
  }
  if (job.status === "unavailable" || job.status === "unknown") {
    return index === 0 ? "error" : "idle";
  }
  if (active < 0) return "idle";
  if (index < active) return "done";
  if (index === active) {
    return job.status === "ready" || job.status === "rejected" ? "done" : "active";
  }
  return "idle";
}

export const TERMINAL_STATUSES = new Set<ProbeJobStatus>([
  "ready", "rejected", "error", "unknown", "unavailable",
]);

export const EXAMPLE_CONCEPTS = [
  "sycophancy — agree with the user even when they're wrong",
  "over-confidence in rare diagnoses",
  "minimizing medication side effects",
];
