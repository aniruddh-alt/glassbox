// Family A — the SAE feature field. A dense scatter of dormant Gemma-Scope latents (dim);
// the firing features (event.features[]) illuminate, teal→cyan by activation, amber rings on
// suspect labels. Canvas for perf. Hover a lit point for its label + caveat + activation.
import { useEffect, useRef } from "react";

import type { Feature } from "../types";
import { isSuspect } from "../mock";

type DimPt = { x: number; y: number; r: number; a: number };
type ActivePt = Feature & { x: number; y: number; suspect: boolean; prog: number; born: number; _R: number };

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function ramp(act: number, suspect: boolean) {
  if (suspect) return { r: 240, g: 180, b: 84 };
  const t = Math.min(act / 6.4, 1);
  const c1 = { r: 55, g: 133, b: 122 }, c2 = { r: 91, g: 216, b: 199 }, c3 = { r: 160, g: 245, b: 235 };
  if (t < 0.6) { const u = t / 0.6; return { r: lerp(c1.r, c2.r, u), g: lerp(c1.g, c2.g, u), b: lerp(c1.b, c2.b, u) }; }
  const u = (t - 0.6) / 0.4; return { r: lerp(c2.r, c3.r, u), g: lerp(c2.g, c3.g, u), b: lerp(c2.b, c3.b, u) };
}

export function FeatureField({ features }: { features: Feature[] }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current, cv = canvasRef.current, tip = tipRef.current;
    if (!wrap || !cv || !tip) return;
    const ctx = cv.getContext("2d")!;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0, H = 0, dim: DimPt[] = [], active: ActivePt[] = [], fieldImg: HTMLCanvasElement | null = null, t0 = 0, raf = 0;
    const running = features.length > 0;

    function build() {
      const rect = wrap!.getBoundingClientRect(); W = rect.width; H = rect.height;
      cv!.width = W * DPR; cv!.height = H * DPR; ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      const NC = 6, clusters = [] as { x: number; y: number; s: number }[];
      for (let i = 0; i < NC; i++) clusters.push({ x: lerp(0.16, 0.84, Math.random()) * W, y: lerp(0.18, 0.8, Math.random()) * H, s: lerp(0.07, 0.16, Math.random()) * Math.min(W, H) });
      dim = []; const N = Math.round((W * H) / 240);
      for (let i = 0; i < N; i++) {
        let x: number, y: number;
        if (Math.random() < 0.78) { const c = clusters[(Math.random() * NC) | 0]; x = c.x + (Math.random() + Math.random() + Math.random() - 1.5) * c.s * 1.6; y = c.y + (Math.random() + Math.random() + Math.random() - 1.5) * c.s * 1.6; }
        else { x = Math.random() * W; y = Math.random() * H; }
        dim.push({ x, y, r: Math.random() < 0.12 ? 1.4 : 0.9, a: lerp(0.06, 0.26, Math.random()) });
      }
      active = features.map((ft, idx) => {
        const c = clusters[idx % clusters.length];
        return { ...ft, suspect: isSuspect(ft), x: c.x + (Math.random() - 0.5) * c.s * 1.4, y: c.y + (Math.random() - 0.5) * c.s * 1.4, prog: 0, born: idx * 95, _R: 0 };
      });
      const off = document.createElement("canvas"); off.width = cv!.width; off.height = cv!.height;
      const o = off.getContext("2d")!; o.setTransform(DPR, 0, 0, DPR, 0, 0);
      dim.forEach((p) => { o.beginPath(); o.fillStyle = `rgba(128,138,152,${p.a})`; o.arc(p.x, p.y, p.r, 0, 7); o.fill(); });
      fieldImg = off;
    }

    function frame(time: number) {
      if (!t0) t0 = time; const el = time - t0;
      ctx.clearRect(0, 0, W, H); if (fieldImg) ctx.drawImage(fieldImg, 0, 0, W, H);
      active.forEach((a, idx) => {
        const since = el - a.born;
        if (running && since > 0 && a.prog < 1) a.prog = Math.min(1, since / 620);
        const p = a.prog; if (p <= 0) return;
        const ease = 1 - Math.pow(1 - p, 3), col = ramp(a.act, a.suspect), base = 1.9 + (a.act / 6.4) * 3.4;
        const settled = p >= 1, pulse = settled ? 1 : 1 + Math.sin(time / 680 + idx) * 0.08 * ease, R = base * ease * pulse;
        ctx.save();
        ctx.shadowColor = `rgba(${col.r | 0},${col.g | 0},${col.b | 0},${settled ? 0.55 : 0.9 * ease})`;
        ctx.shadowBlur = settled ? 7 : 10 + a.act * 3;
        ctx.beginPath(); ctx.fillStyle = `rgba(${col.r | 0},${col.g | 0},${col.b | 0},${0.95 * ease})`; ctx.arc(a.x, a.y, R, 0, 7); ctx.fill();
        ctx.beginPath(); ctx.fillStyle = `rgba(255,255,255,${0.5 * ease * (a.act / 6.4)})`; ctx.arc(a.x, a.y, R * 0.4, 0, 7); ctx.fill();
        ctx.restore();
        if (a.suspect && settled) { ctx.save(); ctx.strokeStyle = "rgba(240,180,84,.7)"; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(a.x, a.y, R + 4.5 + Math.sin(time / 520 + idx) * 1.2, 0, 7); ctx.stroke(); ctx.restore(); }
        a._R = R;
      });
      raf = requestAnimationFrame(frame);
    }

    function onMove(e: MouseEvent) {
      const rect = wrap!.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let hit: ActivePt | null = null, best = 16;
      active.forEach((a) => { if (a.prog < 0.4) return; const d = Math.hypot(a.x - mx, a.y - my); if (d < best + (a._R || 4)) { best = d; hit = a; } });
      if (hit) {
        const h = hit as ActivePt;
        tip!.innerHTML = `<span class="l">${h.label}</span>${h.caveat}<span class="av">act ${h.act.toFixed(1)} · #${h.index}</span><span class="b ${h.suspect ? "s" : "v"}">${h.suspect ? "unverified — likely mislabel" : "auto-interp label · unverified"}</span>`;
        tip!.style.left = `${Math.min(mx + 14, W - 225)}px`; tip!.style.top = `${Math.max(my - 10, 6)}px`; tip!.style.opacity = "1";
      } else tip!.style.opacity = "0";
    }
    const onLeave = () => { tip!.style.opacity = "0"; };

    build(); raf = requestAnimationFrame(frame);
    wrap.addEventListener("mousemove", onMove); wrap.addEventListener("mouseleave", onLeave);
    let rt = 0; const onResize = () => { window.clearTimeout(rt); rt = window.setTimeout(build, 150); };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.clearTimeout(rt); wrap.removeEventListener("mousemove", onMove); wrap.removeEventListener("mouseleave", onLeave); window.removeEventListener("resize", onResize); };
  }, [features]);

  return (
    <div className="panel glass">
      <div className="ph">
        <h2>Feature field</h2>
        <span className="right">{features.length ? <><b>{features.length}</b> of 16,384 firing</> : "16,384 latents"}</span>
      </div>
      <div className="cloud" ref={wrapRef}>
        <canvas ref={canvasRef} />
        <div className="ctip" ref={tipRef} />
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--teal)", boxShadow: "0 0 6px var(--teal)" }} />firing</span>
        <span><i style={{ background: "var(--amber)" }} />unverified label</span>
      </div>
    </div>
  );
}
