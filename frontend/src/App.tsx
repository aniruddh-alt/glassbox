// Root view. Left: multi-turn clinician chat. Right: cognition stage (feature field + probes +
// Claude verdict) reflecting the LATEST message's CognitionEvent. Real /api/chat (no mock).
import { useEffect, useMemo, useState } from "react";

import "./styles.css";
import type { CognitionEvent } from "./types";
import { useCognitionStream } from "./useCognitionStream";
import { useHealth, MODE_BADGE } from "./health";
import { ChatPanel, type Msg } from "./components/ChatPanel";
import { FeatureField } from "./components/FeatureField";
import { ProbePanel } from "./components/ProbePanel";
import { AdjudicationBanner } from "./components/AdjudicationBanner";
import { ObservabilityPage } from "./ObservabilityPage";
import { ProbeBuilderPage } from "./ProbeBuilderPage";
import { ProbeVisibilityPanel } from "./components/ProbeVisibilityPanel";
import { collectProbeIds } from "./useProbeVisibility";

type View = "chat" | "observe" | "build";

function parseView(raw: string | null): View {
  if (raw === "observe" || raw === "build") return raw;
  return "chat";
}

export function App() {
  const { answer, event, status, send } = useCognitionStream();
  const [thread, setThread] = useState<Msg[]>([]);
  const [latest, setLatest] = useState<CognitionEvent | null>(null);
  const health = useHealth();
  const [view, setView] = useState<View>(
    () => parseView(localStorage.getItem("glassbox.view")),
  );
  const [observeMounted, setObserveMounted] = useState(view === "observe");
  const [buildMounted, setBuildMounted] = useState(view === "build");

  function onSend(content: string) {
    const history: Msg[] = [...thread, { role: "user", content }];
    setThread(history);
    send(history);
  }

  useEffect(() => {
    if (status === "done") {
      setThread((t) => (t.length && t[t.length - 1].role === "user"
        ? [...t, { role: "assistant", content: answer || "" }] : t));
      if (event) setLatest(event);
    } else if (status === "error") {
      setThread((t) => (t.length && t[t.length - 1].role === "user"
        ? [...t, { role: "assistant", content: "[generation failed]" }] : t));
    }
  }, [status, answer, event]);

  useEffect(() => {
    localStorage.setItem("glassbox.view", view);
    if (view === "observe") setObserveMounted(true);
    if (view === "build") setBuildMounted(true);
  }, [view]);

  const features = useMemo(() => latest?.features ?? [], [latest]);
  const trackers = useMemo(() => latest?.trackers ?? {}, [latest]);
  const probeIds = useMemo(
    () => collectProbeIds(health?.trackers, trackers),
    [health?.trackers, trackers],
  );

  return (
    <div className="app">
      <header className="glass">
        <div className="mark">
          <svg className="mark-cube" width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <g stroke="var(--ink)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 9 H15 V20 H4 Z" />
              <path d="M9 4 H20 V15" />
              <path d="M4 9 L9 4" />
              <path d="M15 9 L20 4" />
              <path d="M15 20 L20 15" />
              <path d="M9 4 V15 H20" />
              <path d="M9 15 L4 20" />
            </g>
          </svg>
          Glassbox
        </div>
        <div className="meta">
          {health && <span className={`badge ${health.mode}`}>{MODE_BADGE[health.mode]}</span>}
        </div>
        <nav className="app-nav">
          <button
            className={`app-nav-btn${view === "chat" ? " active" : ""}`}
            onClick={() => setView("chat")}
          >Chat</button>
          <button
            className={`app-nav-btn${view === "build" ? " active" : ""}`}
            onClick={() => setView("build")}
          >Build</button>
          <button
            className={`app-nav-btn${view === "observe" ? " active" : ""}`}
            onClick={() => setView("observe")}
          >Observe</button>
        </nav>
        <div className="live"><span className="d" />Live</div>
      </header>

      <main style={{ display: view === "chat" ? "" : "none" }}>
        <ChatPanel
          thread={thread}
          pending={status === "streaming" ? answer : null}
          status={status}
          onSend={onSend}
        />
        <section className="stage">
          <FeatureField features={features} latents={health?.d_sae} />
          <ProbePanel
            trackers={trackers}
            registered={health?.trackers}
            onOpenBuilder={() => setView("build")}
          />
          <ProbeVisibilityPanel probeIds={probeIds} compact />
          <AdjudicationBanner adjudication={latest?.adjudication ?? null} />
          <p className="ethos">
            <b>Read-only instrument.</b> It flags low-confidence turns; it never edits the assistant's output.
          </p>
        </section>
      </main>

      {buildMounted && (
        <div className="obs-shell" style={{ display: view === "build" ? "flex" : "none" }}>
          <ProbeBuilderPage />
        </div>
      )}

      {observeMounted && (
        <div className="obs-shell" style={{ display: view === "observe" ? "flex" : "none" }}>
          <ObservabilityPage />
        </div>
      )}
    </div>
  );
}
