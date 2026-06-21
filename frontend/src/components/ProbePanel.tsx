// Family B — the reliable detectors. Built-in trackers come from event.trackers; AUROC + threshold
// are static probe metadata (PROBE_META), not per-event contract fields. "Define a probe in natural
// language" calls the concept_synth flow through api.ts track() + pollTracker().
import { useEffect, useRef, useState } from "react";

import type { Tracker } from "../types";
import { pollTracker, track } from "../api";
import { PROBE_META } from "../mock";

type Row = {
  id?: string;
  name: string;
  score: number;
  thr: number;
  flag: boolean;
  auroc?: number;
  userDefined?: boolean;
  computing?: boolean;
  step?: string;
  alertDirection?: "high" | "low";
};

function Gauge({ score, thr, flag, alertDirection = "high" }: { score: number; thr: number; flag: boolean; alertDirection?: "high" | "low" }) {
  const [w, setW] = useState(0);
  useEffect(() => { const id = setTimeout(() => setW(score * 100), 60); return () => clearTimeout(id); }, [score]);
  const dangerClass = flag ? "hot" : "calm";
  return (
    <div className={`gauge ${dangerClass} ${alertDirection === "low" ? "low-alert" : ""}`}>
      <div className="thr" style={{ left: `${thr * 100}%` }} />
      <div className="fill" style={{ width: `${w}%` }} />
    </div>
  );
}

function ProbeRow({ row }: { row: Row }) {
  return (
    <div className={`probe ${row.flag ? "hotrow" : ""}`}>
      <div className="pn">
        <b>{row.name}</b>
        <div className="tr">
          {row.userDefined && <span className="ud">custom</span>}
          {row.computing
            ? <span className="auroc calc">{row.step ?? "computing…"}</span>
            : row.auroc != null && <span className="auroc">AUROC {row.auroc.toFixed(2)}</span>}
        </div>
      </div>
      {row.computing
        ? <div className="gauge calm"><div className="fill" /></div>
        : <Gauge score={row.score} thr={row.thr} flag={row.flag} alertDirection={row.alertDirection} />}
      <div className={`val ${row.flag ? "fl" : ""}`}>{row.computing ? "…" : row.score.toFixed(2)}</div>
    </div>
  );
}

export function ProbePanel({ trackers }: { trackers: Record<string, Tracker> }) {
  const [custom, setCustom] = useState<Row[]>([]);
  const [draft, setDraft] = useState("");
  const idRef = useRef(0);

  const builtins: Row[] = Object.entries(trackers).map(([name, t]) => ({
    name, score: t.score, flag: t.flag, thr: PROBE_META[name]?.thr ?? 0.5,
    auroc: PROBE_META[name]?.auroc, userDefined: t.user_defined,
    alertDirection: t.alert_direction ?? "high",
  }));
  const noBuiltins = builtins.length === 0;

  async function define() {
    const concept = (draft.trim() || "over-confidence"); setDraft("");
    const id = `c${++idRef.current}`;
    setCustom((c) => [...c, { id, name: concept, score: 0, thr: 0.5, flag: false, userDefined: true, computing: true, step: "queued" }]);
    const patch = (p: Partial<Row>) =>
      setCustom((c) => c.map((row) => (row.id === id ? { ...row, ...p } : row)));

    // The probe trains on the pod (Claude designs the trait, gemma generates contrastive pairs,
    // they're judged and fit). That takes a while, so poll until a terminal state — the previous
    // single read always caught the job mid-flight and never showed it go live.
    const TERMINAL = new Set(["ready", "rejected", "error", "unknown", "unavailable"]);
    try {
      const created = await track(concept);
      if (!created.tracker_id || created.status === "unavailable") {
        patch({ computing: false, name: `${concept} — unavailable` });
        return;
      }
      const tid = created.tracker_id;
      const deadline = Date.now() + 5 * 60 * 1000; // give the pod loop up to ~5 min
      let status = await pollTracker(tid);
      while (!TERMINAL.has(status.status) && Date.now() < deadline) {
        patch({ step: status.progress?.step ?? status.status });
        await new Promise((r) => setTimeout(r, 3000));
        status = await pollTracker(tid);
      }
      patch({
        name: tid,
        thr: PROBE_META[tid]?.thr ?? 0.5,
        auroc: status.auroc ?? PROBE_META[tid]?.auroc,
        computing: !TERMINAL.has(status.status),
        step: status.status === "ready" ? "live" : status.status,
      });
    } catch {
      patch({ computing: false, name: `${concept} — failed` });
    }
  }

  return (
    <div className="panel">
      <div className="ph">
        <h2>Probes</h2>
        <span className="sub">calibrated</span>
        <span className="right coral">outside threshold</span>
      </div>
      {noBuiltins && custom.length === 0 && (
        <div className="wip">
          <b>Family B — calibrated uncertainty / safety probes</b>
          <span>In progress. Once probes are trained, each message's meter fills toward its red threshold here.</span>
        </div>
      )}
      {/* once a custom probe deploys it arrives as a live tracker in event.trackers (a builtin row),
          so drop the local progress row for it to avoid showing the same probe twice. */}
      {[...builtins, ...custom.filter((r) => !(r.name in trackers))].map((row, i) => <ProbeRow key={row.id ?? `b${i}`} row={row} />)}
      <div className="define preview">
        <span className="tag">preview</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") define(); }}
          placeholder="Define a probe in natural language…  e.g. over-confidence"
        />
        <button onClick={define}>+ track</button>
      </div>
    </div>
  );
}
