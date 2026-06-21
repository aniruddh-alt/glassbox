// Root view. Left: multi-turn clinician chat. Right: cognition stage (feature field + probes +
// Claude verdict) reflecting the LATEST message's CognitionEvent. Real /api/chat (no mock).
import { useEffect, useMemo, useState } from "react";

import "./styles.css";
import type { CognitionEvent } from "./types";
import { useCognitionStream } from "./useCognitionStream";
import { ChatPanel, type Msg } from "./components/ChatPanel";
import { FeatureField } from "./components/FeatureField";
import { ProbePanel } from "./components/ProbePanel";
import { AdjudicationBanner } from "./components/AdjudicationBanner";

export function App() {
  const { answer, event, status, send } = useCognitionStream();
  const [thread, setThread] = useState<Msg[]>([]);
  const [latest, setLatest] = useState<CognitionEvent | null>(null);

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
  }, [status]);

  const features = useMemo(() => latest?.features ?? [], [latest]);
  const trackers = useMemo(() => latest?.trackers ?? {}, [latest]);
  const modelName = latest?.model ?? "model";

  return (
    <div className="app">
      <header className="glass">
        <div className="mark"><span className="lens" /><span className="g">glass</span><b>box</b></div>
        <div className="meta"><span className="pill">{modelName} · L{latest?.layer ?? "—"}</span></div>
        <div className="live"><span className="d" />observing</div>
      </header>

      <main>
        <ChatPanel
          thread={thread}
          pending={status === "streaming" ? answer : null}
          status={status}
          modelName={modelName}
          onSend={onSend}
        />
        <section className="stage">
          <FeatureField features={features} />
          <ProbePanel trackers={trackers} />
          <AdjudicationBanner adjudication={latest?.adjudication ?? null} />
          <p className="ethos">
            <b>Surface, never suppress</b> — we flag when to double-check, never alter the answer.
          </p>
        </section>
      </main>
    </div>
  );
}
