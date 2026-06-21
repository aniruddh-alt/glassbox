// Polls GET /api/observability every 2s; returns {snapshot, status}.
// Cleans up the interval on unmount.
import { useEffect, useState } from "react";

import type { ObservabilitySnapshot } from "./types";
import { getObservability } from "./api";

export type ObsStatus = "loading" | "ok" | "error";

export function useObservability(): { snapshot: ObservabilitySnapshot | null; status: ObsStatus } {
  const [snapshot, setSnapshot] = useState<ObservabilitySnapshot | null>(null);
  const [status, setStatus] = useState<ObsStatus>("loading");

  useEffect(() => {
    let alive = true;

    const poll = () => {
      getObservability()
        .then((s) => {
          if (!alive) return;
          setSnapshot(s);
          setStatus("ok");
        })
        .catch(() => {
          if (!alive) return;
          setStatus("error");
        });
    };

    poll();
    const id = setInterval(poll, 2000);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return { snapshot, status };
}
