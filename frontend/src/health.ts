import { useEffect, useState } from "react";

export interface Health {
  mode: "loading" | "real" | "fallback";
  model: string;
  layer: number;
  model_loaded: boolean;
  sae_loaded: boolean;
  d_sae: number;
  trackers: string[];
}

// Polls /api/health; keeps polling while the backend is still warming up the model.
export function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let alive = true;
    let tries = 0;
    const poll = async () => {
      try {
        const r = await fetch("/api/health");
        const j = (await r.json()) as Health;
        if (!alive) return;
        setHealth(j);
        if (j.mode === "loading" && tries++ < 40) setTimeout(poll, 1500);
      } catch {
        if (alive && tries++ < 40) setTimeout(poll, 2000);
      }
    };
    poll();
    return () => { alive = false; };
  }, []);
  return health;
}

export const MODE_BADGE: Record<Health["mode"], string> = {
  loading: "warming up",
  real: "live model",
  fallback: "synthetic · offline",
};
