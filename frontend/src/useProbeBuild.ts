// Polls a probe-build job on the GPU pod until it reaches a terminal status.
import { useCallback, useEffect, useRef, useState } from "react";

import { pollTracker, track } from "./api";
import type { ProbeBuildJob } from "./probePipeline";
import { TERMINAL_STATUSES } from "./probePipeline";

const STORAGE_KEY = "glassbox.probeBuild";

type BuildPhase = "idle" | "starting" | "running" | "done" | "failed";

function loadSaved(): { trackerId: string; concept: string } | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const j = JSON.parse(raw) as { trackerId?: string; concept?: string };
    return j.trackerId && j.concept ? { trackerId: j.trackerId, concept: j.concept } : null;
  } catch {
    return null;
  }
}

function saveBuild(trackerId: string, concept: string) {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ trackerId, concept }));
}

function clearSaved() {
  sessionStorage.removeItem(STORAGE_KEY);
}

export function useProbeBuild() {
  const [concept, setConcept] = useState("");
  const [job, setJob] = useState<ProbeBuildJob | null>(null);
  const [phase, setPhase] = useState<BuildPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const trackerRef = useRef<string | null>(null);

  const pollOnce = useCallback(async (tid: string) => {
    const j = (await pollTracker(tid)) as ProbeBuildJob;
    setJob(j);
    if (TERMINAL_STATUSES.has(j.status)) {
      setPhase(j.status === "ready" || j.status === "rejected" ? "done" : "failed");
      clearSaved();
    }
    return j;
  }, []);

  // Resume an in-flight build after tab switch / reload.
  useEffect(() => {
    const saved = loadSaved();
    if (!saved) return;
    trackerRef.current = saved.trackerId;
    setConcept(saved.concept);
    setPhase("running");
    pollOnce(saved.trackerId).catch(() => setPhase("failed"));
  }, [pollOnce]);

  useEffect(() => {
    const tid = trackerRef.current;
    if (phase !== "running" || !tid) return;
    const id = setInterval(() => {
      pollOnce(tid).catch(() => {
        setPhase("failed");
        setError("Lost connection to probe job — check the GPU pod.");
      });
    }, 2000);
    return () => clearInterval(id);
  }, [phase, pollOnce]);

  async function start(request?: string) {
    const text = (request ?? concept).trim();
    if (!text || phase === "starting" || phase === "running") return;
    setError(null);
    setJob(null);
    setPhase("starting");
    try {
      const created = await track(text);
      if (created.status === "unavailable" || !created.tracker_id) {
        setPhase("failed");
        setError("GPU pod unavailable — start the tunnel and gpu_service, then retry.");
        return;
      }
      trackerRef.current = created.tracker_id;
      setConcept(text);
      saveBuild(created.tracker_id, text);
      setJob({ tracker_id: created.tracker_id, request: text, status: "pending", progress: { step: "queued", pct: 0 } });
      setPhase("running");
      await pollOnce(created.tracker_id);
    } catch {
      setPhase("failed");
      setError("Failed to submit build job.");
    }
  }

  function reset() {
    trackerRef.current = null;
    clearSaved();
    setJob(null);
    setPhase("idle");
    setError(null);
  }

  return { concept, setConcept, job, phase, error, start, reset };
}
