// The climax — Claude's confident-wrong adjudication. Renders only when present (filled async
// on flagged events). Mounts with the slide-in animation and scrolls itself into view.
import { useEffect, useRef } from "react";

import type { Adjudication } from "../types";

export function AdjudicationBanner({ adjudication }: { adjudication: Adjudication | null | undefined }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (adjudication) ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [adjudication]);
  if (!adjudication) return null;
  return (
    <div className="verdict glass" ref={ref}>
      <div className="mk">⚑</div>
      <div>
        <div className="vt">confident-wrong · claude adjudication</div>
        <div className="vb">{adjudication.rationale}</div>
      </div>
    </div>
  );
}
