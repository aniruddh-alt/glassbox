// Probe builder — visualizes the GPU-pod interpretability agent pipeline step-by-step.
import { useProbeBuild } from "./useProbeBuild";
import {
  EXAMPLE_CONCEPTS,
  PIPELINE_STAGES,
  stepState,
  type ProbeBuildJob,
} from "./probePipeline";

function StageIcon({ state }: { state: ReturnType<typeof stepState> }) {
  if (state === "done") return <span className="pb-icon done" aria-hidden="true">✓</span>;
  if (state === "active") return <span className="pb-spinner" aria-hidden="true" />;
  if (state === "error") return <span className="pb-icon err" aria-hidden="true">!</span>;
  return <span className="pb-icon idle" aria-hidden="true" />;
}

function ResultCard({ job }: { job: ProbeBuildJob }) {
  const deployed = job.status === "ready";
  const rejected = job.status === "rejected";
  return (
    <section className={`panel pb-result${deployed ? " ok" : rejected ? " warn" : ""}`}>
      <div className="ph">
        <h2>{deployed ? "Probe deployed" : rejected ? "Not deployed" : "Build failed"}</h2>
        {job.tracker_id && <span className="sub mono">{job.tracker_id}</span>}
      </div>
      <dl className="pb-result-grid">
        {job.trait_name && (
          <><dt>Trait</dt><dd>{job.trait_name}</dd></>
        )}
        {job.auroc != null && (
          <><dt>Held-out AUROC</dt><dd>{job.auroc.toFixed(3)}</dd></>
        )}
        {job.baseline_auroc != null && (
          <><dt>Baseline AUROC</dt><dd>{job.baseline_auroc.toFixed(3)}</dd></>
        )}
        {job.n_kept != null && (
          <><dt>Clean rows</dt><dd>{job.n_kept}</dd></>
        )}
        {job.verdict && (
          <><dt>Pipeline summary</dt><dd>{job.verdict}</dd></>
        )}
        {job.error && (
          <><dt>Error</dt><dd className="err">{job.error}</dd></>
        )}
        {!job.error && job.status === "error" && (
          <><dt>Error</dt><dd className="err">Build failed at {job.progress?.step ?? job.status} — check GPU pod logs and retry.</dd></>
        )}
      </dl>
      {rejected && job.verdict && (
        <p className="obs-hint">
          The AUROC gate did not pass, so this probe was not registered — that reflects held-out
          separability on this run, not whether the concept is worth monitoring. Tune the trait
          spec or contrast set and rebuild.
        </p>
      )}
      {deployed && (
        <p className="obs-hint">Switch to <b>Chat</b> and send a message — the new probe scores on the next turn.</p>
      )}
    </section>
  );
}

export function ProbeBuilderPage() {
  const { concept, setConcept, job, phase, error, start, reset } = useProbeBuild();
  const busy = phase === "starting" || phase === "running";
  const pct = job?.progress?.pct ?? (phase === "starting" ? 2 : 0);

  return (
    <div className="pb-page">
      <div className="pb-head">
        <div>
          <h1>Probe builder</h1>
          <p className="pb-lead">
            Describe a behavior in natural language. The GPU pod runs a six-stage pipeline —
            Claude designs the trait, Gemma generates contrastive activations, a calibrated probe is fit and deployed live.
          </p>
        </div>
        {(phase === "done" || phase === "failed") && (
          <button type="button" className="obs-eval-btn" onClick={reset}>New build</button>
        )}
      </div>

      <div className="pb-layout">
        <section className="panel pb-form">
          <div className="ph"><h2>Concept</h2></div>
          <div className="pb-form-body">
            <label className="pb-label" htmlFor="pb-concept">What should this probe detect?</label>
            <textarea
              id="pb-concept"
              className="pb-input"
              rows={4}
              value={concept}
              disabled={busy}
              placeholder="e.g. sycophancy — the model agrees with a incorrect clinical assumption instead of correcting it"
              onChange={(e) => setConcept(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) start(); }}
            />
            <button
              type="button"
              className="pb-start"
              disabled={busy || !concept.trim()}
              onClick={() => start()}
            >
              {phase === "starting" ? "Submitting…" : busy ? "Building on GPU pod…" : "Start build"}
            </button>
            {(error || job?.error) && (
              <p className="pb-error">{error ?? job?.error}</p>
            )}
            <div className="pb-examples">
              <span className="pb-examples-label">Examples</span>
              {EXAMPLE_CONCEPTS.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  className="pb-example"
                  disabled={busy}
                  onClick={() => { setConcept(ex); }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="panel pb-pipeline">
          <div className="ph">
            <h2>Pipeline</h2>
            {job?.tracker_id && <span className="sub mono">{job.tracker_id}</span>}
            {busy && (
              <span className="right">{pct}%</span>
            )}
          </div>

          {busy && (
            <div className="pb-progress-wrap" aria-hidden="true">
              <div className="pb-progress" style={{ width: `${pct}%` }} />
            </div>
          )}

          <ol className="pb-steps">
            {PIPELINE_STAGES.map((stage, i) => {
              const state = stepState(i, job);
              return (
                <li key={stage.key} className={`pb-step ${state}`}>
                  <div className="pb-step-rail">
                    <StageIcon state={state} />
                    {i < PIPELINE_STAGES.length - 1 && <span className="pb-rail" />}
                  </div>
                  <div className="pb-step-body">
                    <div className="pb-step-head">
                      <b>{stage.title}</b>
                      <code className="pb-backend">{stage.backend}</code>
                    </div>
                    <p>{stage.detail}</p>
                    {state === "active" && job?.progress?.step && (
                      <span className="pb-step-live">Running: {job.progress.step}</span>
                    )}
                    {state === "done" && i === PIPELINE_STAGES.length - 1 && job?.status === "rejected" && (
                      <span className="pb-step-live warn">Gate not met — probe not registered</span>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        </section>
      </div>

      {job && (job.status === "ready" || job.status === "rejected" || job.status === "error") && (
        <ResultCard job={job} />
      )}
    </div>
  );
}
