// Family B — harmfulness + over-confidence probes. Custom probes are built on the Build tab.
import type { Tracker } from "../types";
import { ACTIVE_PROBES, PROBE_META, probeLabel } from "../probes";
import { useProbeVisibility } from "../useProbeVisibility";

type Row = {
  name: string;
  label: string;
  score: number;
  thr: number;
  flag: boolean;
  auroc?: number;
  alertDirection?: "high" | "low";
  awaiting?: boolean;
  userDefined?: boolean;
};

function Gauge({ score, thr, flag, alertDirection = "high" }: { score: number; thr: number; flag: boolean; alertDirection?: "high" | "low" }) {
  const dangerClass = flag ? "hot" : "calm";
  return (
    <div className={`gauge ${dangerClass} ${alertDirection === "low" ? "low-alert" : ""}`}>
      <div className="thr" style={{ left: `${thr * 100}%` }} />
      <div className="fill" style={{ width: `${score * 100}%`, transition: "width .5s ease" }} />
    </div>
  );
}

function ProbeRow({ row }: { row: Row }) {
  return (
    <div className={`probe ${row.flag ? "hotrow" : ""}${row.awaiting ? " awaiting" : ""}`}>
      <div className="pn">
        <b>{row.label}</b>
        <div className="tr">
          {row.userDefined && <span className="ud">custom</span>}
          {row.awaiting
            ? <span className="auroc calc">awaiting turn</span>
            : row.auroc != null && <span className="auroc">AUROC {row.auroc.toFixed(2)}</span>}
        </div>
      </div>
      {row.awaiting
        ? <div className="gauge calm"><div className="fill" /></div>
        : <Gauge score={row.score} thr={row.thr} flag={row.flag} alertDirection={row.alertDirection} />}
      <div className={`val ${row.flag ? "fl" : ""}`}>{row.awaiting ? "…" : row.score.toFixed(2)}</div>
    </div>
  );
}

export function ProbePanel({
  trackers,
  registered = ACTIVE_PROBES as unknown as string[],
  onOpenBuilder,
}: {
  trackers: Record<string, Tracker>;
  registered?: string[];
  onOpenBuilder?: () => void;
}) {
  const { isEnabled } = useProbeVisibility();
  const orderedIds = ACTIVE_PROBES.filter(
    (id) => (!registered.length || registered.includes(id)) && isEnabled(id),
  );

  const builtins: Row[] = orderedIds.map((name) => {
    const t = trackers[name];
    const meta = PROBE_META[name];
    if (t) {
      return {
        name,
        label: probeLabel(name),
        score: t.score,
        flag: t.flag,
        thr: meta?.thr ?? 0.5,
        auroc: meta?.auroc,
        userDefined: t.user_defined,
        alertDirection: t.alert_direction ?? "high",
      };
    }
    return {
      name,
      label: probeLabel(name),
      score: 0,
      flag: false,
      thr: meta?.thr ?? 0.5,
      auroc: meta?.auroc,
      alertDirection: "high" as const,
      awaiting: true,
    };
  });

  // Custom probes registered on the pod (health.trackers) — show even before the first scored turn.
  const customRegistered = (registered ?? []).filter(
    (id) => !(ACTIVE_PROBES as readonly string[]).includes(id) && isEnabled(id),
  );
  const custom: Row[] = customRegistered.map((name) => {
    const t = trackers[name];
    if (t) {
      if (t.status && t.status !== "ready") return null;
      return {
        name,
        label: probeLabel(name),
        score: t.score,
        flag: t.flag,
        thr: 0.5,
        auroc: undefined,
        userDefined: true,
        alertDirection: t.alert_direction ?? "high",
      };
    }
    return {
      name,
      label: probeLabel(name),
      score: 0,
      flag: false,
      thr: 0.5,
      auroc: undefined,
      userDefined: true,
      alertDirection: "high" as const,
      awaiting: true,
    };
  }).filter((row): row is Row => row != null);

  const visibleRows = [...builtins, ...custom];

  return (
    <div className="panel">
      <div className="ph">
        <h2>Probes</h2>
        <span className="sub">{visibleRows.map((row) => row.label).join(" · ") || "built-ins"}</span>
        <span className="right coral">outside threshold</span>
      </div>
      {visibleRows.map((row) => (
        <ProbeRow key={row.name} row={row} />
      ))}
      {onOpenBuilder && (
        <button type="button" className="pb-chat-link" onClick={onOpenBuilder}>
          Build a custom probe →
        </button>
      )}
    </div>
  );
}
