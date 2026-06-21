// Which probes appear in Chat + Observe. Persisted in localStorage; synced across tabs.
import { useCallback, useEffect, useState } from "react";

import { ACTIVE_PROBES, HIDDEN_PROBES, isBuiltinProbe, probeLabel } from "./probes";

const STORAGE_KEY = "glassbox.probeVisibility";

export type ProbeDisplayPrefs = Record<string, boolean>;

function defaultEnabled(id: string): boolean {
  if (HIDDEN_PROBES.has(id)) return false;
  return isBuiltinProbe(id);
}

function loadPrefs(): ProbeDisplayPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as ProbeDisplayPrefs;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePrefs(prefs: ProbeDisplayPrefs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  window.dispatchEvent(new Event("glassbox-probe-visibility"));
}

export function isProbeEnabled(id: string, prefs: ProbeDisplayPrefs): boolean {
  if (HIDDEN_PROBES.has(id)) return false;
  if (id in prefs) return prefs[id];
  return defaultEnabled(id);
}

export function collectProbeIds(
  ...sources: Array<Record<string, unknown> | string[] | null | undefined>
): string[] {
  const ids = new Set<string>();
  for (const src of sources) {
    if (!src) continue;
    if (Array.isArray(src)) {
      src.forEach((id) => ids.add(id));
    } else {
      Object.keys(src).forEach((id) => ids.add(id));
    }
  }
  return [...ids]
    .filter((id) => !HIDDEN_PROBES.has(id))
    .sort((a, b) => {
      const ab = isBuiltinProbe(a);
      const bb = isBuiltinProbe(b);
      if (ab !== bb) return ab ? -1 : 1;
      return probeLabel(a).localeCompare(probeLabel(b));
    });
}

export function useProbeVisibility() {
  const [prefs, setPrefs] = useState<ProbeDisplayPrefs>(() => loadPrefs());

  useEffect(() => {
    const sync = () => setPrefs(loadPrefs());
    window.addEventListener("storage", sync);
    window.addEventListener("glassbox-probe-visibility", sync);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener("glassbox-probe-visibility", sync);
    };
  }, []);

  const setEnabled = useCallback((id: string, enabled: boolean) => {
    setPrefs((prev) => {
      const next = { ...prev, [id]: enabled };
      savePrefs(next);
      return next;
    });
  }, []);

  const isEnabled = useCallback((id: string) => isProbeEnabled(id, prefs), [prefs]);

  const filterIds = useCallback(
    (ids: string[]) => ids.filter((id) => isProbeEnabled(id, prefs)),
    [prefs],
  );

  return { prefs, isEnabled, setEnabled, filterIds };
}
