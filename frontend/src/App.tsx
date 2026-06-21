// Root view — the ClinicianView demo hero. OWNER: Lane C.
// Chat (left) | cognition stage (right): the SAE feature field + probe panel + the Claude verdict.
// Built against the demo event (mock:true). Flip mock:false once /api/chat streams real NDJSON.
import { useEffect, useMemo, useState } from "react";

import "./styles.css";
import { useCognitionStream } from "./useCognitionStream";
import { DEMO_EVENT } from "./mock";
import { ChatPanel } from "./components/ChatPanel";
import { FeatureField } from "./components/FeatureField";
import { ProbePanel } from "./components/ProbePanel";
import { AdjudicationBanner } from "./components/AdjudicationBanner";

export function App() {
  const { answer, event, status, send } = useCognitionStream({ mock: true });
  const [hasRun, setHasRun] = useState(false);

  // auto-run once on load so the page shows the full arc immediately
  useEffect(() => {
    const id = setTimeout(() => send([{ role: "user", content: DEMO_EVENT.io.user_msg }]), 400);
    return () => clearTimeout(id);
  }, []);
  useEffect(() => { if (status === "done") setHasRun(true); }, [status]);

  // stable identity when empty so the canvas effect doesn't rebuild on every streamed token
  const features = useMemo(() => event?.features ?? [], [event]);
  const trackers = useMemo(() => event?.trackers ?? {}, [event]);

  return (
    <div className="app">
      <header className="glass">
        <div className="mark"><span className="lens" /><span className="g">glass</span><b>box</b></div>
        <div className="meta"><span className="pill">gemma-2-2b-it · L12</span></div>
        <div className="live"><span className="d" />observing</div>
      </header>

      <main>
        <ChatPanel
          userMsg={DEMO_EVENT.io.user_msg}
          answer={answer}
          status={status}
          hasRun={hasRun}
          onRun={(content) => send([{ role: "user", content }])}
        />
        <section className="stage">
          <FeatureField features={features} />
          <ProbePanel trackers={trackers} />
          <AdjudicationBanner adjudication={event?.adjudication ?? null} />
          <p className="ethos">
            <b>Surface, never suppress</b> — we flag when to double-check, never alter the answer.
          </p>
        </section>
      </main>
    </div>
  );
}
