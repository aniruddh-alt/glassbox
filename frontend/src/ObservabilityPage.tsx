// Observability page — live GET /api/observability snapshot (no demo fallback on the happy path).
// Health fields populate infrastructure panels even before the first chat turn lands in the store.
import { useState } from "react";

import type { ObservabilitySnapshot, ObsConfidentWrong, ObsHealth } from "./types";
import { useObservability } from "./useObservability";
import { runEval, testSentryAlarm, replaySentryAlarm } from "./api";
import { ACTIVE_PROBES, probeLabel } from "./probes";
import { collectProbeIds, useProbeVisibility } from "./useProbeVisibility";
import { ProbeVisibilityPanel } from "./components/ProbeVisibilityPanel";

// ── Formatting ────────────────────────────────────────────────────────────────

function fmtMs(ms: number | null | undefined): string {
  return ms == null ? "—" : `${Math.round(ms).toLocaleString()} ms`;
}

function fmtPct(rate: number): string {
  return `${(rate * 100).toFixed(0)}%`;
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

function humanize(s: string): string {
  return s.replace(/_/g, " ");
}

function shortModel(model: string): string {
  const parts = model.split("/");
  return parts[parts.length - 1] ?? model;
}

function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

// ── Sparkline ─────────────────────────────────────────────────────────────────

function Sparkline({
  series,
  width = 96,
  height = 32,
  accent = false,
}: {
  series: number[];
  width?: number;
  height?: number;
  accent?: boolean;
}) {
  if (!series.length) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="var(--line-2)" strokeWidth="1" />
      </svg>
    );
  }
  const max = Math.max(...series, 0.01);
  const pts = series
    .map((v, i) => {
      const x = series.length === 1 ? width / 2 : (i / (series.length - 1)) * width;
      const y = height - 2 - (v / max) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const lastVal = series[series.length - 1];
  const hot = accent || lastVal / max >= 0.6;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }} aria-hidden="true">
      <polyline
        points={pts}
        fill="none"
        stroke={hot ? "var(--accent)" : "var(--ink)"}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

// ── Loading skeleton ──────────────────────────────────────────────────────────

function PanelSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <section className="panel obs-section obs-skeleton">
      <div className="ph"><div className="obs-sk-line w40" /><div className="obs-sk-line w20" /></div>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="obs-sk-row"><div className="obs-sk-line w60" /><div className="obs-sk-bar" /></div>
      ))}
    </section>
  );
}

// ── System status hero (always from health — wired even with 0 turns) ─────────

function SystemHero({ health, turns }: { health: ObsHealth; turns: number }) {
  const probes = collectProbeIds(ACTIVE_PROBES, health.trackers);
  return (
    <section className="obs-hero panel">
      <div className="obs-hero-main">
        <span className={`badge ${health.mode}`}>
          {health.mode === "real" ? "live model" : health.mode === "fallback" ? "synthetic · offline" : "warming up"}
        </span>
        <h2 className="obs-hero-title">{shortModel(health.model)} · layer {health.layer}</h2>
        <p className="obs-hero-sub">
          {turns > 0
            ? `${turns} turn${turns !== 1 ? "s" : ""} in the in-process ledger`
            : "Infrastructure ready — send a chat message to start recording turns"}
        </p>
      </div>
      <dl className="obs-hero-grid">
        <div className="obs-hero-stat">
          <dt>GPU pod</dt>
          <dd className={health.pod_reachable ? "ok" : "warn"}>
            {health.pod_reachable ? "reachable" : "unreachable"}
          </dd>
        </div>
        <div className="obs-hero-stat">
          <dt>SAE</dt>
          <dd className={health.sae_loaded ? "ok" : ""}>{health.sae_loaded ? `${health.d_sae.toLocaleString()} latents` : "loading"}</dd>
        </div>
        <div className="obs-hero-stat">
          <dt>Recon cosine</dt>
          <dd className={health.sae_recon_ok ? "ok" : health.sae_recon_cosine != null ? "warn" : ""}>
            {health.sae_recon_cosine != null ? health.sae_recon_cosine.toFixed(3) : "—"}
          </dd>
        </div>
        <div className="obs-hero-stat">
          <dt>Probes loaded</dt>
          <dd>{probes.length ? probes.map(probeLabel).join(" · ") : "none"}</dd>
        </div>
      </dl>
    </section>
  );
}

// ── KPI strip ─────────────────────────────────────────────────────────────────

type Kpi = { label: string; value: string; unit?: string; tone?: "accent" | "ok" | "muted" };

function KpiStrip({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const { totals, flag_rate, trackers, latency, uncertainty_series, health } = snapshot;
  const probeCount = collectProbeIds(ACTIVE_PROBES, health.trackers).length;
  const uncMean = mean(uncertainty_series);

  const kpis: Kpi[] = [
    { label: "Turns recorded", value: `${totals.turns}`, tone: totals.turns > 0 ? undefined : "muted" },
    {
      label: "Flag rate",
      value: totals.turns ? fmtPct(flag_rate) : "—",
      tone: flag_rate >= 0.5 ? "accent" : totals.turns ? undefined : "muted",
    },
    {
      label: "Flagged turns",
      value: `${totals.flags}`,
      tone: totals.flags > 0 ? "accent" : "muted",
    },
    { label: "Probes", value: `${probeCount}`, tone: probeCount ? "ok" : "muted" },
    {
      label: "Turn p50",
      value: latency.turn_ms.p50 != null ? `${Math.round(latency.turn_ms.p50).toLocaleString()}` : "—",
      unit: latency.turn_ms.p50 != null ? "ms" : undefined,
      tone: latency.turn_ms.p50 != null ? undefined : "muted",
    },
    {
      label: "Mean over-confidence",
      value: uncMean != null ? uncMean.toFixed(2) : "—",
      tone: uncMean != null && uncMean >= 0.6 ? "accent" : uncMean != null ? undefined : "muted",
    },
  ];

  return (
    <div className="obs-kpis">
      {kpis.map((k) => (
        <div key={k.label} className="obs-kpi">
          <span className="obs-kpi-val" data-tone={k.tone}>
            {k.value}
            {k.unit && <i className="obs-kpi-unit">{k.unit}</i>}
          </span>
          <span className="obs-kpi-label">{k.label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Tracker strip (registered probes from health, series from store) ──────────

function TrackerStrip({
  snapshot,
  visibleIds,
}: {
  snapshot: ObservabilitySnapshot;
  visibleIds: string[];
}) {
  const registered = snapshot.health.trackers ?? [];
  const orderedIds = visibleIds.filter(
    (id) => !(ACTIVE_PROBES as readonly string[]).includes(id) || !registered.length || registered.includes(id),
  );
  const data = snapshot.trackers;
  const totalFlags = orderedIds.reduce((a, id) => a + (data[id]?.flag_count ?? 0), 0);

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Probes</h2>
        <span className="sub">{orderedIds.map(probeLabel).join(" · ")}</span>
        <span className="right">session flags: <b>{totalFlags}</b></span>
      </div>
      <div className="obs-tracker-strip">
        {orderedIds.map((id) => {
          const t = data[id];
          const hot = (t?.current ?? 0) >= 0.6;
          return (
            <div key={id} className={`obs-tracker-row${hot ? " hot" : ""}`}>
              <span className="obs-tracker-name">{probeLabel(id)}</span>
              <Sparkline series={t?.series ?? []} width={120} accent={hot} />
              <span className={`obs-tracker-val${hot ? " fl" : ""}`}>
                {t?.current != null ? t.current.toFixed(2) : "—"}
              </span>
              <span className="obs-tracker-flags">
                {t ? `${t.flag_count} flag${t.flag_count !== 1 ? "s" : ""}` : "awaiting turn"}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── Uncertainty trend ─────────────────────────────────────────────────────────

function OverConfidenceTrend({ series }: { series: number[] }) {
  const last = series.length ? series[series.length - 1] : null;
  const avg = mean(series);

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Over-confidence trend</h2>
        <span className="sub">{series.length ? `${series.length} turns` : "no samples"}</span>
        {last != null && (
          <span className="right">latest <b className={last >= 0.6 ? "fl" : ""}>{last.toFixed(2)}</b></span>
        )}
      </div>
      {series.length ? (
        <div className="obs-unc-body">
          <Sparkline series={series} width={280} height={48} accent={last != null && last >= 0.6} />
          <div className="obs-unc-meta">
            <span>mean {avg != null ? avg.toFixed(2) : "—"}</span>
            <span>min {Math.min(...series).toFixed(2)}</span>
            <span>max {Math.max(...series).toFixed(2)}</span>
          </div>
        </div>
      ) : (
        <p className="obs-empty">Over-confidence probe scores plot here once chat turns are recorded.</p>
      )}
    </section>
  );
}

// ── Confident-wrong feed ──────────────────────────────────────────────────────

function ConfidentWrongFeed({
  items,
  visibleIds,
  sentryBase,
}: {
  items: ObsConfidentWrong[];
  visibleIds: string[];
  sentryBase?: string | null;
}) {
  if (!items.length) {
    return (
      <section className="panel obs-section">
        <div className="ph"><h2>Confident-wrong feed</h2><span className="sub">no flagged turns</span></div>
        <p className="obs-empty">
          Flagged turns surface here with feature labels and probe scores — never the raw question or answer.
        </p>
      </section>
    );
  }

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Confident-wrong feed</h2>
        <span className="right"><b>{items.length}</b> flagged</span>
      </div>
      <div className="obs-cw-list">
        {[...items].reverse().map((item) => {
          const trackerEntries = Object.entries(item.trackers ?? {}).filter(([tid]) =>
            visibleIds.includes(tid),
          );
          return (
            <div key={item.message_id} className="obs-cw-item">
              <div className="obs-cw-meta">
                {sentryBase ? (
                  <a
                    className="obs-cw-id obs-cw-id-link"
                    href={`${sentryBase}?query=message_id%3A${item.message_id}`}
                    target="_blank"
                    rel="noreferrer"
                    title={`Open ${item.message_id} in Sentry`}
                  >
                    {shortId(item.message_id)} ↗
                  </a>
                ) : (
                  <span className="obs-cw-id" title={item.message_id}>{shortId(item.message_id)}</span>
                )}
                <span className="obs-cw-ts">{new Date(item.ts * 1000).toLocaleString()}</span>
              </div>
              {trackerEntries.length > 0 && (
                <div className="obs-cw-trackers">
                  {trackerEntries.map(([tid, t]) => {
                    const score = typeof t.score === "number" ? t.score : null;
                    if (score == null) return null;
                    return (
                      <span key={tid} className={`obs-cw-score${t.flag ? " fl" : ""}`}>
                        {probeLabel(tid)} {score.toFixed(2)}
                      </span>
                    );
                  })}
                </div>
              )}
              {item.feature_labels.length > 0 && (
                <div className="obs-cw-labels">
                  {item.feature_labels.map((lbl) => (
                    <span key={lbl} className="obs-label-chip">{lbl}</span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── Feature leaderboard ───────────────────────────────────────────────────────

function FeatureLeaderboard({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const features = snapshot.top_features;
  if (!features.length) {
    return (
      <section className="panel obs-section">
        <div className="ph"><h2>Feature leaderboard</h2><span className="sub">no activations yet</span></div>
        <p className="obs-empty">Top SAE features by activation count across all recorded turns.</p>
      </section>
    );
  }

  const maxCount = features[0].count;
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Feature leaderboard</h2>
        <span className="sub">top {features.length} by count</span>
      </div>
      <div className="obs-leaderboard">
        {features.map((f) => (
          <div key={f.label} className="obs-lb-row">
            <span className="obs-lb-label" title={f.label}>{f.label}</span>
            <span className="obs-lb-meta">
              <span className="obs-lb-count">{f.count}×</span>
              {f.mean_act != null && <span className="obs-lb-act">μ {f.mean_act.toFixed(2)}</span>}
            </span>
            <div className="obs-lb-bar-wrap">
              <div className="obs-lb-bar" style={{ width: `${(f.count / maxCount) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Latency & health ──────────────────────────────────────────────────────────

function LatencyBar({ label, ms, maxMs }: { label: string; ms: number | null; maxMs: number }) {
  const pct = ms != null && maxMs > 0 ? (ms / maxMs) * 100 : 0;
  return (
    <div className="obs-lat-row">
      <span className="obs-lat-label">{label}</span>
      <div className="obs-lat-bar-wrap">
        <div className="obs-lat-bar" style={{ width: `${pct}%` }} />
      </div>
      <span className="obs-lat-val">{fmtMs(ms)}</span>
    </div>
  );
}

function LatencyHealth({
  snapshot,
  onEval,
  evalRunning,
}: {
  snapshot: ObservabilitySnapshot;
  onEval: () => void;
  evalRunning: boolean;
}) {
  const { latency, health } = snapshot;
  const stageOrder = ["pod_roundtrip", "capture", "sae", "trackers", "label_fetch", "ranking"] as const;
  const presentStages = stageOrder.filter((s) => latency.stages[s]?.p50 != null);
  const maxStageMs = Math.max(...presentStages.map((s) => latency.stages[s]!.p50!), 1);
  const hasTurnLatency = latency.turn_ms.p50 != null;

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Latency</h2>
        {hasTurnLatency ? (
          <span className="right">
            p50 <b>{fmtMs(latency.turn_ms.p50)}</b>
            {" · "}
            p95 <b>{fmtMs(latency.turn_ms.p95)}</b>
            {latency.turn_ms.last != null && <> · last <b>{fmtMs(latency.turn_ms.last)}</b></>}
          </span>
        ) : (
          <span className="sub">percentiles appear after the first turn</span>
        )}
      </div>

      {presentStages.length ? (
        <div className="obs-lat-list">
          {presentStages.map((s) => (
            <LatencyBar key={s} label={humanize(s)} ms={latency.stages[s]!.p50} maxMs={maxStageMs} />
          ))}
        </div>
      ) : (
        <div className="obs-lat-empty">
          <p>Pipeline stage timings (pod roundtrip, SAE encode, probe scoring, label fetch) populate here from the in-process perf ledger.</p>
          <ul>
            <li>Model: <b>{shortModel(health.model)}</b></li>
            <li>Pod: <b className={health.pod_reachable ? "ok" : "warn"}>{health.pod_reachable ? "connected" : "disconnected"}</b></li>
            <li>SAE recon: <b className={health.sae_recon_ok ? "ok" : ""}>{health.sae_recon_cosine?.toFixed(3) ?? "—"}</b></li>
          </ul>
        </div>
      )}

      <div className="obs-lat-footer">
        <button
          className="obs-eval-btn"
          onClick={onEval}
          disabled={evalRunning || snapshot.totals.turns === 0}
          aria-busy={evalRunning}
          title={snapshot.totals.turns === 0 ? "Record at least one chat turn first" : undefined}
        >
          {evalRunning ? (
            <>
              <span className="obs-spinner" aria-hidden="true" /> Running coherence eval…
            </>
          ) : (
            "Run coherence eval"
          )}
        </button>
      </div>
    </section>
  );
}

// ── Sentry (inline issues — already wired via /api/observability) ───────────────

function SentryPanel({
  snapshot,
  onTestAlarm,
  onReplayAlarm,
  testRunning,
  testResult,
  replayRunning,
  replayResult,
}: {
  snapshot: ObservabilitySnapshot;
  onTestAlarm: () => void;
  onReplayAlarm: () => void;
  testRunning: boolean;
  testResult: string | null;
  replayRunning: boolean;
  replayResult: string | null;
}) {
  const { sentry } = snapshot;
  const issues = sentry.issues ?? [];
  const canRead = sentry.configured;
  const canEmit = sentry.emit_configured !== false;

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Sentry alarms</h2>
        {canRead && sentry.deep_link && (
          <a className="obs-ext-link" href={sentry.deep_link} target="_blank" rel="noreferrer">
            open project ↗
          </a>
        )}
        {!canRead && !canEmit && <span className="sub">not configured</span>}
      </div>
      {!canEmit && (
        <p className="obs-empty">Set <code>SENTRY_DSN</code> on the backend to emit alarms when a probe flags a turn.</p>
      )}
      {canEmit && !canRead && (
        <p className="obs-empty">Alarms emit on flagged turns. Set <code>SENTRY_AUTH_TOKEN</code> to list recent issues here.</p>
      )}
      {canEmit && (
        <div className="obs-lat-footer">
          <button
            type="button"
            className="obs-eval-btn"
            onClick={onReplayAlarm}
            disabled={replayRunning || snapshot.confident_wrong.length === 0}
            title={snapshot.confident_wrong.length === 0 ? "No flagged turns to replay" : undefined}
          >
            {replayRunning ? (
              <>
                <span className="obs-spinner" aria-hidden="true" /> Replaying last flagged turn…
              </>
            ) : (
              "Replay Sentry for last flagged turn"
            )}
          </button>
          <button
            type="button"
            className="obs-eval-btn"
            onClick={onTestAlarm}
            disabled={testRunning}
            aria-busy={testRunning}
          >
            {testRunning ? (
              <>
                <span className="obs-spinner" aria-hidden="true" /> Sending test alarm…
              </>
            ) : (
              "Send test Sentry alarm"
            )}
          </button>
          {(replayResult || testResult) && (
            <span className="obs-eval-result">{replayResult ?? testResult}</span>
          )}
        </div>
      )}
      {canRead && !issues.length ? (
        <p className="obs-empty">No recent alarms — flagged turns and system errors surface here once emitted.</p>
      ) : canRead ? (
        <div className="obs-sentry-list">
          {issues.slice(0, 6).map((issue) => (
            <a
              key={issue.shortId}
              href={issue.permalink}
              target="_blank"
              rel="noreferrer"
              className="obs-sentry-issue"
            >
              <span className={`obs-sentry-level obs-level-${issue.level}`}>{issue.level}</span>
              <span className="obs-sentry-title">{issue.title}</span>
              <span className="obs-sentry-meta">
                {issue.shortId} · {issue.count} events · {new Date(issue.lastSeen).toLocaleString()}
              </span>
            </a>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function PhoenixPanel({ url }: { url: string }) {
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Phoenix ledger</h2>
        <a className="obs-ext-link" href={url} target="_blank" rel="noreferrer">open ↗</a>
      </div>
      <a className="obs-link-row" href={url} target="_blank" rel="noreferrer">
        <span className="obs-link-text">
          <b>Open trace waterfall &amp; evals</b>
          <span className="obs-link-sub">{url} — span ledger for every cognition turn (no prompt/response)</span>
        </span>
        <span className="obs-link-arrow" aria-hidden="true">↗</span>
      </a>
    </section>
  );
}

// ── Page root ─────────────────────────────────────────────────────────────────

export function ObservabilityPage() {
  const { snapshot, status } = useObservability();
  const { filterIds } = useProbeVisibility();
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalResult, setEvalResult] = useState<string | null>(null);
  const [sentryTestRunning, setSentryTestRunning] = useState(false);
  const [sentryTestResult, setSentryTestResult] = useState<string | null>(null);
  const [sentryReplayRunning, setSentryReplayRunning] = useState(false);
  const [sentryReplayResult, setSentryReplayResult] = useState<string | null>(null);

  async function handleSentryReplay() {
    if (sentryReplayRunning) return;
    setSentryReplayRunning(true);
    setSentryReplayResult(null);
    try {
      const res = await replaySentryAlarm();
      setSentryReplayResult(
        res.ok
          ? `Replayed ${res.flag_reason} alarm for ${res.message_id?.slice(0, 8)}…`
          : "Replay failed",
      );
    } catch (e) {
      setSentryReplayResult(e instanceof Error ? e.message : "Replay failed");
    } finally {
      setSentryReplayRunning(false);
    }
  }

  async function handleSentryTest() {
    if (sentryTestRunning) return;
    setSentryTestRunning(true);
    setSentryTestResult(null);
    try {
      const res = await testSentryAlarm();
      setSentryTestResult(
        res.ok
          ? `Test alarm sent (${res.message_id}) — check Sentry Issues`
          : "Test alarm failed",
      );
    } catch (e) {
      setSentryTestResult(e instanceof Error ? e.message : "Test alarm failed");
    } finally {
      setSentryTestRunning(false);
    }
  }

  async function handleEval() {
    if (evalRunning || !snapshot) return;
    setEvalRunning(true);
    setEvalResult(null);
    try {
      const res = await runEval();
      setEvalResult("status" in res ? res.status : `evaluated ${res.evaluated}, off-domain ${res.off_domain}`);
    } catch {
      setEvalResult("eval failed — check Phoenix is running on :6006");
    } finally {
      setEvalRunning(false);
    }
  }

  if (status === "loading" && !snapshot) {
    return (
      <div className="obs-page">
        <div className="obs-head">
          <h1>Observability</h1>
          <span className="obs-head-status">Connecting to backend…</span>
        </div>
        <PanelSkeleton rows={2} />
        <div className="obs-grid"><PanelSkeleton /><PanelSkeleton /></div>
        <PanelSkeleton rows={4} />
      </div>
    );
  }

  if (status === "error" && !snapshot) {
    return (
      <div className="obs-page">
        <div className="obs-head">
          <h1>Observability</h1>
          <span className="obs-head-status err">Backend unreachable</span>
        </div>
        <section className="panel obs-section obs-error-card">
          <p><b>Cannot reach GET /api/observability.</b></p>
          <p className="obs-empty">Start the backend on port 8000 (Vite proxies <code>/api</code> in dev). The dashboard reads the in-process store populated by <code>fanout()</code> after each chat turn.</p>
        </section>
      </div>
    );
  }

  if (!snapshot) return null;

  const turns = snapshot.totals.turns;
  const allProbeIds = collectProbeIds(
    ACTIVE_PROBES,
    snapshot.health.trackers,
    snapshot.trackers,
    ...snapshot.confident_wrong.map((item) => item.trackers),
  );
  const visibleProbeIds = filterIds(allProbeIds);

  return (
    <div className="obs-page">
      <div className="obs-head">
        <h1>Observability</h1>
        <span className="obs-head-status ok">
          Live · {turns} turn{turns !== 1 ? "s" : ""} · {fmtPct(snapshot.flag_rate)} flagged
        </span>
        {snapshot.ts != null && (
          <span className="obs-head-ts">last event {new Date(snapshot.ts * 1000).toLocaleTimeString()}</span>
        )}
        {evalResult && <span className="obs-eval-result">{evalResult}</span>}
      </div>

      <SystemHero health={snapshot.health} turns={turns} />
      <KpiStrip snapshot={snapshot} />
      <ProbeVisibilityPanel probeIds={allProbeIds} compact />

      {turns === 0 && (
        <p className="obs-hint">
          Ledger is empty — infrastructure is wired and healthy. Send a message on <b>Chat</b> to record turns,
          populate probe series, latency percentiles, and the feature leaderboard.
        </p>
      )}

      <div className="obs-grid">
        <TrackerStrip snapshot={snapshot} visibleIds={visibleProbeIds} />
        <OverConfidenceTrend series={snapshot.uncertainty_series} />
      </div>

      <div className="obs-grid">
        <ConfidentWrongFeed items={snapshot.confident_wrong} visibleIds={visibleProbeIds} sentryBase={snapshot.sentry.configured ? snapshot.sentry.deep_link : null} />
        <FeatureLeaderboard snapshot={snapshot} />
      </div>

      <LatencyHealth snapshot={snapshot} onEval={handleEval} evalRunning={evalRunning} />

      <div className="obs-grid">
        <SentryPanel
          snapshot={snapshot}
          onTestAlarm={handleSentryTest}
          onReplayAlarm={handleSentryReplay}
          testRunning={sentryTestRunning}
          testResult={sentryTestResult}
          replayRunning={sentryReplayRunning}
          replayResult={sentryReplayResult}
        />
        <PhoenixPanel url={snapshot.phoenix_ui_url} />
      </div>
    </div>
  );
}
