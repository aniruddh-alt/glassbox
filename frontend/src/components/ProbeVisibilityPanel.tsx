import { collectProbeIds, isProbeEnabled, useProbeVisibility } from "../useProbeVisibility";
import { isBuiltinProbe, probeLabel } from "../probes";

export function ProbeVisibilityPanel({
  probeIds,
  compact = false,
}: {
  probeIds: string[];
  compact?: boolean;
}) {
  const { prefs, setEnabled } = useProbeVisibility();
  const ids = collectProbeIds(probeIds);

  if (!ids.length) {
    return compact ? null : (
      <section className="panel obs-section obs-probe-prefs">
        <div className="ph"><h2>Visible probes</h2><span className="sub">none loaded yet</span></div>
        <p className="obs-empty">Send a chat turn or deploy a custom probe to configure which scores appear.</p>
      </section>
    );
  }

  return (
    <section className={`panel obs-section obs-probe-prefs${compact ? " compact" : ""}`}>
      <div className="ph">
        <h2>Visible probes</h2>
        <span className="sub">choose which scores appear in Chat and Observe</span>
      </div>
      <div className="obs-probe-prefs-grid">
        {ids.map((id) => {
          const on = isProbeEnabled(id, prefs);
          return (
            <label key={id} className={`obs-probe-pref${on ? " on" : ""}`}>
              <input
                type="checkbox"
                checked={on}
                onChange={(e) => setEnabled(id, e.target.checked)}
              />
              <span className="obs-probe-pref-label">{probeLabel(id)}</span>
              {!isBuiltinProbe(id) && <span className="obs-probe-pref-tag">custom</span>}
            </label>
          );
        })}
      </div>
    </section>
  );
}
