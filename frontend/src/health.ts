import { useCallback, useEffect, useState } from "react";

export interface Health {
  mode: "loading" | "real" | "fallback";
  model: string;
  layer: number;
  model_loaded: boolean;
  sae_loaded: boolean;
  d_sae: number;
  trackers: string[];
  pod_reachable?: boolean;
  anthropic_configured?: boolean;
  active_probe_jobs?: number;
}

const POLL_MS = 8000;

// Polls /api/health on mount and every few seconds so custom probes appear after deploy.
export function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/health");
      if (!r.ok) return;
      setHealth((await r.json()) as Health);
    } catch {
      /* backend may be restarting */
    }
  }, []);

  useEffect(() => {
    let alive = true;
    let tries = 0;
    const poll = async () => {
      if (!alive) return;
      try {
        const r = await fetch("/api/health");
        const j = (await r.json()) as Health;
        if (!alive) return;
        setHealth(j);
        if (j.mode === "loading" && tries++ < 40) {
          setTimeout(poll, 1500);
        }
      } catch {
        if (alive && tries++ < 40) setTimeout(poll, 2000);
      }
    };
    poll();
    const id = setInterval(() => { void refresh(); }, POLL_MS);
    const onRefresh = () => { void refresh(); };
    window.addEventListener("glassbox-health-refresh", onRefresh);
    return () => {
      alive = false;
      clearInterval(id);
      window.removeEventListener("glassbox-health-refresh", onRefresh);
    };
  }, [refresh]);

  return health;
}

export function refreshHealth() {
  window.dispatchEvent(new Event("glassbox-health-refresh"));
}

export const MODE_BADGE: Record<Health["mode"], string> = {
  loading: "warming up",
  real: "live model",
  fallback: "synthetic · offline",
};
