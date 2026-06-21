// SHARED: mirror of backend/schema.py (contract #1). Keep byte-for-byte in sync.

export const SCHEMA_VERSION = "1.0";

export type Severity = "info" | "warning";
export type AlertDirection = "high" | "low";

export interface IO {
  user_msg: string;
  response: string;
}

export interface Tracker {
  score: number;        // calibrated probability [0,1] — the meter
  proj: number;         // diff-of-means projection (baseline / eng view)
  flag: boolean;        // score >= calibrated threshold
  reliable: boolean;    // Family B = true; never set for SAE labels
  proj_pre?: number | null;
  alert_direction?: AlertDirection;
  user_defined?: boolean;
  status?: "computing" | "ready";
}

export interface Feature {
  index: number;
  label: string;
  act: number;
  source: string;       // "12-gemmascope-res-16k"
  caveat: string;       // short provenance note (kept in the contract; not surfaced in the UI)
  tracked: string | null;
}

export interface Adjudication {
  verdict: "likely_correct" | "likely_hallucinated" | "uncertain";
  rationale: string;
  by: "claude";
}

export interface CognitionEvent {
  schema_version: string;
  type: "event";
  message_id: string;
  ts: number;
  model: string;
  layer: number;
  io: IO;

  // Family B — reliable signal; null until calibrated probes are registered (WIP)
  uncertainty: number | null;
  uncertainty_proj: number | null;
  uncertainty_proj_pre?: number | null;
  flag: boolean;
  severity: Severity;
  trackers: Record<string, Tracker>;

  // Family A — exploratory
  features: Feature[];

  // Anthropic prize — present only when flagged, filled async
  adjudication?: Adjudication | null;
}

// Streamed per-token line (live mode); final line of /api/chat is the CognitionEvent.
export interface TokenLine {
  type: "token";
  text: string;
  top_features: Feature[];
  uncertainty?: number | null;
}

export type StreamLine = TokenLine | CognitionEvent;

// ── Observability snapshot (mirrors GET /api/observability §7) ──────────────

export interface ObsTrackerSeries {
  current: number | null;
  flag_count: number;
  series: number[];
}

export interface ObsConfidentWrong {
  message_id: string;
  ts: number;
  uncertainty: number | null;
  trackers: Record<string, { score: number; flag: boolean }>;
  feature_labels: string[];
}

export interface ObsTopFeature {
  label: string;
  count: number;
  mean_act: number | null;
}

export interface ObsStageLatency {
  p50: number | null;
}

export interface ObsLatency {
  turn_ms: { p50: number | null; p95: number | null; last: number | null };
  stages: Record<string, ObsStageLatency>;
}

export interface ObsHealth {
  mode: "real" | "fallback" | "loading";
  model_loaded: boolean;
  sae_loaded: boolean;
  model: string;
  layer: number;
  d_sae: number;
  trackers: string[];
  sae_recon_cosine: number | null;
  sae_recon_ok: boolean;
  pod_reachable: boolean;
  pod_url_configured: boolean;
}

export interface ObsSentryIssue {
  shortId: string;
  title: string;
  level: string;
  count: number;
  lastSeen: string;
  permalink: string;
}

export interface ObsSentry {
  configured: boolean;
  deep_link: string | null;
  issues: ObsSentryIssue[];
}

export interface ObservabilitySnapshot {
  ts: number | null;
  totals: { turns: number; flags: number };
  flag_rate: number;
  uncertainty_series: number[];
  trackers: Record<string, ObsTrackerSeries>;
  confident_wrong: ObsConfidentWrong[];
  top_features: ObsTopFeature[];
  latency: ObsLatency;
  health: ObsHealth;
  sentry: ObsSentry;
  phoenix_ui_url: string;
}
