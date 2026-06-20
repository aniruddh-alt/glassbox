// Root view. OWNER: Lane C.
// ClinicianView (demo hero): ChatPanel + UncertaintyMeter (green→red confident-wrong zone)
//   + FeatureCloud (react-force-graph-2d, top-k, "unverified" badges, k-slider) + TrackedConcepts.
// ObservabilityView: NOT a custom dashboard — linkout/iframe cards to the live Sentry project
//   and Phoenix (localhost:6006). Sells two sponsors in one click.
import { useCognitionStream } from "./useCognitionStream";

export function App() {
  const { answer, event, status, send } = useCognitionStream();

  // TODO(Lane C): build ClinicianView + ObservabilityView with a ViewToggle.
  // Build everything against fixtures/cognition_event.sample.json first.
  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 900, margin: "40px auto", padding: 20 }}>
      <h1>GlassBox</h1>
      <p style={{ color: "#667" }}>Cognition-observability for medical LLMs — scaffold.</p>
      <button onClick={() => send([{ role: "user", content: "Is metformin safe in pregnancy?" }])}>
        Send test message
      </button>
      <p>status: {status}</p>
      <pre style={{ whiteSpace: "pre-wrap" }}>{answer}</pre>
      {event && (
        <pre style={{ background: "#0a0e16", color: "#9fd0ff", padding: 12, overflowX: "auto" }}>
          {JSON.stringify(event, null, 2)}
        </pre>
      )}
    </div>
  );
}
