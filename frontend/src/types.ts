// SHARED: mirror of backend/schema.py (contract #1). Keep byte-for-byte in sync.

export const SCHEMA_VERSION = "1.0";

export type Severity = "info" | "warning";

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
  user_defined?: boolean;
  status?: "computing" | "ready";
}

export interface Feature {
  index: number;
  label: string;
  act: number;
  source: string;       // "12-gemmascope-res-16k"
  caveat: string;       // "auto-interp label, may be unreliable"
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
