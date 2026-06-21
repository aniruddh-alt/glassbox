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

export function App() {
  const { answer, event, status, send } = useCognitionStream();
  const [thread, setThread] = useState<Msg[]>([]);
  const [latest, setLatest] = useState<CognitionEvent | null>(null);
  const health = useHealth();
  const [view, setView] = useState<"chat" | "observe">("chat");

  function onSend(content: string) {
    const history: Msg[] = [...thread, { role: "user", content }];
    setThread(history);
    send(history);
  }

  // Commit the assistant turn + capture its event when a stream finishes.
  // The "last msg is user" guard makes this idempotent across re-renders.
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

  const features = useMemo(() => latest?.features ?? [], [latest]);
  const trackers = useMemo(() => latest?.trackers ?? {}, [latest]);
  const modelName = latest?.model ?? health?.model ?? "model";
  const layerLabel = latest?.layer ?? health?.layer ?? "—";

  return (
    <div className="app">
      <header className="glass">
        <div className="mark"><span className="led" />Glassbox</div>
        <div className="meta">
          <span className="pill">{modelName} · L{layerLabel}</span>
          {health && <span className={`badge ${health.mode}`}>{MODE_BADGE[health.mode]}</span>}
        </div>
        <nav className="app-nav">
          <button
            className={`app-nav-btn${view === "chat" ? " active" : ""}`}
            onClick={() => setView("chat")}
          >Chat</button>
          <button
            className={`app-nav-btn${view === "observe" ? " active" : ""}`}
            onClick={() => setView("observe")}
          >Observe</button>
        </nav>
        <div className="live"><span className="d" />Live</div>
      </header>

      {/* Chat view — stays MOUNTED when on Observe to preserve in-flight stream + history */}
      <main style={{ display: view === "chat" ? "" : "none" }}>
        <ChatPanel
          thread={thread}
          pending={status === "streaming" ? answer : null}
          status={status}
          modelName={modelName}
          onSend={onSend}
        />
        <section className="stage">
          <FeatureField features={features} latents={health?.d_sae} />
          <ProbePanel trackers={trackers} />
          <AdjudicationBanner adjudication={latest?.adjudication ?? null} />
          <p className="ethos">
            <b>Read-only instrument.</b> It flags low-confidence turns; it never edits the model's output.
          </p>
        </section>
      </main>

      {/* Observe view — mounted lazily but kept alive once first rendered */}
      {view === "observe" && <ObservabilityPage />}
    </div>
  );
}
