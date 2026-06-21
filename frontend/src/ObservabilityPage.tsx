// Observability page — six sections surfacing the GET /api/observability snapshot.
// No new npm deps; hand-rolled SVG for sparklines/bars. Uses CSS tokens from styles.css.
// OWNER: Lane C.
import { useState } from "react";

import type { ObservabilitySnapshot, ObsConfidentWrong } from "./types";
import { useObservability } from "./useObservability";
import { DEMO_OBSERVABILITY_SNAPSHOT } from "./mock";
import { runEval } from "./api";

// ── Formatting helpers ────────────────────────────────────────────────────────

// Latency arrives as floats off the wire (e.g. 15929.5961…); show whole ms, grouped.
function fmtMs(ms: number | null): string {
  return ms == null ? "—" : `${Math.round(ms).toLocaleString()}ms`;
}

// Message ids can be 32-char hashes — clip so they don't dominate the row.
function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

// Stage keys are snake_case engineering names; read better with spaces.
function humanize(s: string): string {
  return s.replace(/_/g, " ");
}

// ── Sparkline (SVG polyline over a series of [0,1]-ish values) ────────────────

function Sparkline({ series, width = 96, height = 32 }: { series: number[]; width?: number; height?: number }) {
  if (!series.length) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="var(--line-2)" strokeWidth="1" />
      </svg>
    );
  }
  const max = Math.max(...series, 1);
  const pts = series
    .map((v, i) => {
      const x = series.length === 1 ? width / 2 : (i / (series.length - 1)) * width;
      const y = height - 2 - ((v / max) * (height - 4));
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const lastVal = series[series.length - 1];
  const hot = lastVal / max >= 0.6;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
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

// ── TrackerStrip ────────────────────────────────────────────────────────────

function TrackerStrip({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const entries = Object.entries(snapshot.trackers);
  if (!entries.length) {
    return (
      <section className="panel obs-section">
        <div className="ph"><h2>Probes</h2><span className="sub">no probes registered</span></div>
        <p className="obs-empty">Calibrated probes appear here once registered.</p>
      </section>
    );
  }
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Probes</h2>
        <span className="sub">{entries.length} tracker{entries.length !== 1 ? "s" : ""}</span>
        <span className="right">flags: <b>{entries.reduce((a, [, t]) => a + t.flag_count, 0)}</b></span>
      </div>
      <div className="obs-tracker-strip">
        {entries.map(([id, tracker]) => {
          const hot = (tracker.current ?? 0) >= 0.6;
          return (
            <div key={id} className={`obs-tracker-row${hot ? " hot" : ""}`}>
              <span className="obs-tracker-name">{id}</span>
              <Sparkline series={tracker.series} />
              <span className={`obs-tracker-val${hot ? " fl" : ""}`}>
                {tracker.current != null ? tracker.current.toFixed(2) : "—"}
              </span>
              <span className="obs-tracker-flags">{tracker.flag_count} flags</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── ConfidentWrongFeed ──────────────────────────────────────────────────────
// Shows ONLY feature_labels + tracker scores — no question/answer in the data.

function ConfidentWrongFeed({ items }: { items: ObsConfidentWrong[] }) {
  if (!items.length) {
    return (
      <section className="panel obs-section">
        <div className="ph"><h2>Confident-wrong feed</h2><span className="sub">no flagged turns</span></div>
        <p className="obs-empty">Flagged turns (low uncertainty, wrong answer) appear here.</p>
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
        {items.map((item) => {
          const trackerEntries = Object.entries(item.trackers);
          return (
            <div key={item.message_id} className="obs-cw-item">
              <div className="obs-cw-meta">
                {item.uncertainty != null && (
                  <span className="obs-cw-unc">unc {item.uncertainty.toFixed(2)}</span>
                )}
                <span className="obs-cw-id" title={item.message_id}>{shortId(item.message_id)}</span>
                <span className="obs-cw-ts">{new Date(item.ts * 1000).toLocaleTimeString()}</span>
              </div>
              {trackerEntries.length > 0 && (
                <div className="obs-cw-trackers">
                  {trackerEntries.map(([tid, t]) => (
                    <span key={tid} className={`obs-cw-score${t.flag ? " fl" : ""}`}>
                      {tid}: {t.score.toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
              <div className="obs-cw-labels">
                {item.feature_labels.map((lbl) => (
                  <span key={lbl} className="obs-label-chip">{lbl}</span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── FeatureLeaderboard ──────────────────────────────────────────────────────

function FeatureLeaderboard({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const features = snapshot.top_features;
  if (!features.length) {
    return (
      <section className="panel obs-section">
        <div className="ph"><h2>Feature leaderboard</h2><span className="sub">no data yet</span></div>
        <p className="obs-empty">Top activated SAE features across all turns.</p>
      </section>
    );
  }
  const maxCount = features[0].count;
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Feature leaderboard</h2>
        <span className="sub">top {features.length} by activation count</span>
      </div>
      <div className="obs-leaderboard">
        {features.map((f) => (
          <div key={f.label} className="obs-lb-row">
            <span className="obs-lb-label" title={f.label}>{f.label}</span>
            <span className="obs-lb-meta">
              <span className="obs-lb-count">{f.count}</span>
              {f.mean_act != null && (
                <span className="obs-lb-act">{f.mean_act.toFixed(2)}</span>
              )}
            </span>
            <div className="obs-lb-bar-wrap">
              <div
                className="obs-lb-bar"
                style={{ width: `${(f.count / maxCount) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── LatencyHealth ───────────────────────────────────────────────────────────

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

function ReconGauge({ cosine }: { cosine: number | null }) {
  const pct = cosine != null ? cosine * 100 : 0;
  const ok = cosine == null || cosine >= 0.8;
  return (
    <div className="obs-recon">
      <span className="obs-recon-label">SAE recon cosine</span>
      <div className="obs-recon-track">
        <div
          className="obs-recon-fill"
          style={{ width: `${pct}%`, background: ok ? "var(--ok)" : "var(--accent)" }}
        />
      </div>
      <span className="obs-recon-val" style={{ color: ok ? "var(--ok)" : "var(--accent)" }}>
        {cosine != null ? cosine.toFixed(3) : "—"}
      </span>
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
  const maxStageMs = Math.max(
    ...stageOrder.map((s) => latency.stages[s]?.p50 ?? 0),
    1
  );

  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Latency &amp; health</h2>
        <span className={`badge ${health.mode}`} style={{ fontSize: "12px" }}>
          {health.mode === "real" ? "live model" : health.mode === "fallback" ? "synthetic · offline" : "warming up"}
        </span>
        <span className="right">
          p50 <b>{fmtMs(latency.turn_ms.p50)}</b>
          {" "}&middot;{" "}
          p95 <b>{fmtMs(latency.turn_ms.p95)}</b>
        </span>
      </div>
      <div className="obs-lat-list">
        {stageOrder.map((s) =>
          latency.stages[s] ? (
            <LatencyBar key={s} label={humanize(s)} ms={latency.stages[s].p50} maxMs={maxStageMs} />
          ) : null
        )}
      </div>
      <div className="obs-lat-footer">
        <ReconGauge cosine={health.sae_recon_cosine} />
        <button
          className="obs-eval-btn"
          onClick={onEval}
          disabled={evalRunning}
          aria-busy={evalRunning}
        >
          {evalRunning ? (
            <>
              <span className="obs-spinner" aria-hidden="true" /> Running eval&hellip;
            </>
          ) : (
            "Run coherence eval"
          )}
        </button>
      </div>
    </section>
  );
}

// ── PhoenixEmbed ────────────────────────────────────────────────────────────

function PhoenixEmbed({ url }: { url: string }) {
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Phoenix ledger</h2>
        <a className="obs-ext-link" href={url} target="_blank" rel="noreferrer">
          open in new tab
        </a>
      </div>
      <iframe
        src={url}
        title="Arize Phoenix"
        className="obs-phoenix-iframe"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
      />
    </section>
  );
}

// ── SentryStrip ─────────────────────────────────────────────────────────────
// REST issues → permalink deep-links. NOT an iframe (X-Frame-Options: deny).

function SentryStrip({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const { sentry } = snapshot;
  return (
    <section className="panel obs-section">
      <div className="ph">
        <h2>Sentry alarms</h2>
        {sentry.deep_link && (
          <a className="obs-ext-link" href={sentry.deep_link} target="_blank" rel="noreferrer">
            open in Sentry
          </a>
        )}
        {!sentry.configured && <span className="sub">not configured</span>}
      </div>
      {!sentry.issues.length ? (
        <p className="obs-empty">
          {sentry.configured ? "No recent alarms." : "Set SENTRY_AUTH_TOKEN to enable the issues feed."}
        </p>
      ) : (
        <div className="obs-sentry-list">
          {sentry.issues.map((issue) => (
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
                {issue.shortId} &middot; {issue.count} events &middot; {new Date(issue.lastSeen).toLocaleString()}
              </span>
            </a>
          ))}
        </div>
      )}
    </section>
  );
}

// ── KpiStrip ─────────────────────────────────────────────────────────────────
// Headline numbers lifted out of the panels so the state of the system reads at a glance.

type Kpi = { label: string; value: string; unit?: string; tone?: "accent" | "ok" };

function KpiStrip({ snapshot }: { snapshot: ObservabilitySnapshot }) {
  const { totals, flag_rate, trackers, latency, health } = snapshot;
  const kpis: Kpi[] = [
    { label: "Turns", value: `${totals.turns}` },
    {
      label: "Flag rate",
      value: `${(flag_rate * 100).toFixed(0)}`,
      unit: "%",
      tone: flag_rate >= 0.5 ? "accent" : undefined,
    },
    { label: "Flags", value: `${totals.flags}`, tone: totals.flags > 0 ? "accent" : undefined },
    { label: "Probes", value: `${Object.keys(trackers).length}` },
    {
      label: "p50 latency",
      value: latency.turn_ms.p50 != null ? Math.round(latency.turn_ms.p50).toLocaleString() : "—",
      unit: latency.turn_ms.p50 != null ? "ms" : undefined,
    },
    {
      label: "SAE recon",
      value: health.sae_recon_cosine != null ? health.sae_recon_cosine.toFixed(3) : "—",
      tone: health.sae_recon_cosine == null ? undefined : health.sae_recon_ok ? "ok" : "accent",
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

// ── ObservabilityPage ────────────────────────────────────────────────────────

export function ObservabilityPage() {
  const { snapshot: liveSnapshot, status } = useObservability();
  const snapshot = liveSnapshot ?? DEMO_OBSERVABILITY_SNAPSHOT;

  const [evalRunning, setEvalRunning] = useState(false);
  const [evalResult, setEvalResult] = useState<string | null>(null);

  async function handleEval() {
    if (evalRunning) return;
    setEvalRunning(true);
    setEvalResult(null);
    try {
      const res = await runEval();
      if ("status" in res) {
        setEvalResult(res.status);
      } else {
        setEvalResult(`evaluated ${res.evaluated}, off-domain ${res.off_domain}`);
      }
    } catch {
      setEvalResult("error");
    } finally {
      setEvalRunning(false);
    }
  }

  return (
    <div className="obs-page">
      <div className="obs-head">
        <h1>Observability</h1>
        {status === "loading" && <span className="obs-head-status">Connecting to backend&hellip;</span>}
        {status === "error" && (
          <span className="obs-head-status err">Backend unreachable — showing demo data.</span>
        )}
        {status === "ok" && liveSnapshot && (
          <span className="obs-head-status ok">
            Live &mdash; {liveSnapshot.totals.turns} turns, {(liveSnapshot.flag_rate * 100).toFixed(0)}% flagged
          </span>
        )}
        {!liveSnapshot && status !== "loading" && status !== "error" && (
          <span className="obs-head-status">Demo data</span>
        )}
        {evalResult && <span className="obs-eval-result">{evalResult}</span>}
      </div>

      <KpiStrip snapshot={snapshot} />

      {/* Pair the two short panels on the top row and the two tall ones below,
          so neither column ends in a ragged gap. */}
      <div className="obs-grid">
        <TrackerStrip snapshot={snapshot} />
        <LatencyHealth snapshot={snapshot} onEval={handleEval} evalRunning={evalRunning} />
        <ConfidentWrongFeed items={snapshot.confident_wrong} />
        <FeatureLeaderboard snapshot={snapshot} />
      </div>

      <PhoenixEmbed url={snapshot.phoenix_ui_url} />
      <SentryStrip snapshot={snapshot} />
    </div>
  );
}
