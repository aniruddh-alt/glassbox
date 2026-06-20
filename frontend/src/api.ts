// Small fetch helpers. OWNER: Lane C.

export async function track(concept: string, description?: string) {
  const r = await fetch("/api/track", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ concept, description }),
  });
  return r.json() as Promise<{ tracker_id: string; status: string }>;
}

export async function pollTracker(trackerId: string) {
  const r = await fetch(`/api/track/${trackerId}`);
  return r.json() as Promise<{ status: "computing" | "ready"; auroc?: number }>;
}

export async function featureLabel(index: number) {
  const r = await fetch(`/api/feature/${index}`);
  return r.json() as Promise<{ index: number; label: string; source: string; caveat: string }>;
}
