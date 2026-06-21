// Small fetch helpers. OWNER: Lane C.
import type { ObservabilitySnapshot } from "./types";

export async function track(concept: string, description?: string) {
  const r = await fetch("/api/track", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request: concept, concept, description }),
  });
  // status is the pod job lifecycle (pending | designing | generating | judging | fitting | ready
  // | rejected | error), plus "unavailable" when the pod can't be reached (tracker_id absent then).
  return r.json() as Promise<{ tracker_id?: string; status: string; artifact?: string }>;
}

export async function pollTracker(trackerId: string) {
  const r = await fetch(`/api/track/${trackerId}`);
  return r.json() as Promise<{
    tracker_id?: string;
    status: string;
    auroc?: number | null;
    baseline_auroc?: number | null;
    verdict?: string;
    progress?: { step: string; pct: number };
    artifact?: string;
  }>;
}

export async function featureLabel(index: number) {
  const r = await fetch(`/api/feature/${index}`);
  return r.json() as Promise<{ index: number; label: string; source: string; caveat: string }>;
}

export function getObservability(): Promise<ObservabilitySnapshot> {
  // fetch only rejects on network failure — a 404/5xx still resolves. Throw on non-OK so the
  // useObservability hook's .catch fires (status="error") and the page falls back to demo data
  // instead of rendering a {detail:"Not Found"} body as a snapshot and crashing on missing fields.
  return fetch("/api/observability").then((r) => {
    if (!r.ok) throw new Error(`GET /api/observability -> ${r.status}`);
    return r.json();
  }) as Promise<ObservabilitySnapshot>;
}

export function runEval(): Promise<{ evaluated: number; off_domain: number } | { status: string }> {
  return fetch("/api/observability/eval", { method: "POST" }).then((r) => {
    if (!r.ok) throw new Error(`POST /api/observability/eval -> ${r.status}`);
    return r.json();
  });
}
