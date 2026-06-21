// Family A — the SAE feature field, as a living activation map. Position carries
// meaning: distance from centre ∝ activation (strongest latents pull to the middle;
// size + ink-weight track it too). Layout is seeded by the feature set, so it's
// stable rather than reshuffling-random. The whole constellation drifts on a slow
// coherent flow so it breathes instead of sitting frozen. Labels are collision-
// resolved once against every dot + every other label, then ride their points with
// a leader line — so a label never overlaps a point and association is unambiguous.
import { useEffect, useRef } from "react";

import type { Feature } from "../types";
import { isSuspect } from "../mock";

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// small deterministic PRNG so a given set of features always lays out the same way
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function ramp(act: number, suspect: boolean) {
  if (suspect) return { r: 220, g: 47, b: 44 }; // signal red — unverified label
  // firing latents on a light field: mid-grey -> ink as activation rises
  const t = Math.min(act / 6.4, 1);
  const c1 = { r: 152, g: 152, b: 158 }, c2 = { r: 78, g: 78, b: 84 }, c3 = { r: 24, g: 24, b: 27 };
  if (t < 0.6) { const u = t / 0.6; return { r: lerp(c1.r, c2.r, u), g: lerp(c1.g, c2.g, u), b: lerp(c1.b, c2.b, u) }; }
  const u = (t - 0.6) / 0.4; return { r: lerp(c2.r, c3.r, u), g: lerp(c2.g, c3.g, u), b: lerp(c2.b, c3.b, u) };
}

// coherent slow flow — nearby points move together, so spacing computed at rest stays valid
function flow(x: number, y: number, t: number): [number, number] {
  const dx = Math.sin(x * 0.011 + t * 0.00045) * 3.4 + Math.sin(y * 0.017 - t * 0.0007) * 2.1;
  const dy = Math.cos(y * 0.012 + t * 0.0005) * 3.4 + Math.cos(x * 0.015 + t * 0.0006) * 2.1;
  return [dx, dy];
}

const rectsOverlap = (a: Box, b: Box, m = 0) =>
  a.x < b.x + b.w + m && a.x + a.w + m > b.x && a.y < b.y + b.h + m && a.y + a.h + m > b.y;
function circleHitsRect(cx: number, cy: number, r: number, b: Box) {
  const nx = Math.max(b.x, Math.min(cx, b.x + b.w)), ny = Math.max(b.y, Math.min(cy, b.y + b.h));
  return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r;
}

type Box = { x: number; y: number; w: number; h: number };
type Dim = { x: number; y: number; r: number; a: number };
type Pt = Feature & {
  suspect: boolean; actN: number;
  bx: number; by: number; R: number; born: number; prog: number;
  labeled: boolean; lox: number; loy: number; lw: number; lh: number; flip: boolean;
  x: number; y: number; // current drifted position, refreshed each frame for hit-testing
};

const TOP = 5; // strongest N get a persistent label (suspects are always labelled too)

export function FeatureField({ features, latents = 16384 }: { features: Feature[]; latents?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current, cv = canvasRef.current, tip = tipRef.current;
    if (!wrap || !cv || !tip) return;
    const ctx = cv.getContext("2d")!;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const LABEL_FONT = "11px 'Hanken Grotesk',ui-sans-serif,sans-serif";
    let W = 0, H = 0, dim: Dim[] = [], pts: Pt[] = [], fieldImg: HTMLCanvasElement | null = null;
    let cx = 0, cy = 0, t0 = 0, raf = 0;

    function build() {
      const rect = wrap!.getBoundingClientRect(); W = rect.width; H = rect.height;
      cv!.width = W * DPR; cv!.height = H * DPR; ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      cx = W / 2; cy = H * 0.47;
      const md = Math.min(W, H);
      const seed = features.reduce((s, f) => (s + Math.imul(f.index, 2654435761)) | 0, (features.length * 40503) | 0) >>> 0;
      const rng = mulberry32(seed);

      // dormant backdrop — organic clusters, cached (static texture behind the data)
      const NC = 6, clusters: { x: number; y: number; s: number }[] = [];
      for (let i = 0; i < NC; i++) clusters.push({ x: lerp(0.16, 0.84, rng()) * W, y: lerp(0.18, 0.8, rng()) * H, s: lerp(0.07, 0.16, rng()) * md });
      dim = []; const N = Math.round((W * H) / 300);
      for (let i = 0; i < N; i++) {
        let x: number, y: number;
        if (rng() < 0.78) { const c = clusters[(rng() * NC) | 0]; x = c.x + (rng() + rng() + rng() - 1.5) * c.s * 1.6; y = c.y + (rng() + rng() + rng() - 1.5) * c.s * 1.6; }
        else { x = rng() * W; y = rng() * H; }
        dim.push({ x, y, r: rng() < 0.12 ? 1.4 : 0.9, a: lerp(0.05, 0.18, rng()) });
      }
      const off = document.createElement("canvas"); off.width = cv!.width; off.height = cv!.height;
      const o = off.getContext("2d")!; o.setTransform(DPR, 0, 0, DPR, 0, 0);
      dim.forEach((p) => { o.beginPath(); o.fillStyle = `rgba(24,24,27,${p.a})`; o.arc(p.x, p.y, p.r, 0, 7); o.fill(); });
      fieldImg = off;

      // firing latents — radial: activation -> distance from centre (strong = central)
      const maxAct = Math.max(1e-6, ...features.map((f) => f.act));
      pts = features.map((ft, idx) => {
        const actN = (ft.act / maxAct) * 6.4, strong = ft.act / maxAct;
        const radius = lerp(0.08, 0.4, 1 - strong) * md + (rng() - 0.5) * 0.06 * md;
        const ang = rng() * Math.PI * 2;
        return {
          ...ft, suspect: isSuspect(ft), actN,
          bx: cx + Math.cos(ang) * radius, by: cy + Math.sin(ang) * radius * 0.82,
          R: 2.4 + (actN / 6.4) * 3.8, born: idx * 90, prog: 0,
          labeled: false, lox: 0, loy: 0, lw: 0, lh: 0, flip: false, x: 0, y: 0,
        };
      });

      // relax: push apart any overlapping dots, keep inside bounds (only ~14 points)
      const m = 16;
      for (let it = 0; it < 90; it++) {
        for (let i = 0; i < pts.length; i++) {
          for (let j = i + 1; j < pts.length; j++) {
            const a = pts[i], b = pts[j], dx = b.bx - a.bx, dy = b.by - a.by;
            const d = Math.hypot(dx, dy) || 0.01, need = a.R + b.R + 22;
            if (d < need) { const push = (need - d) / 2, ux = dx / d, uy = dy / d; a.bx -= ux * push; a.by -= uy * push; b.bx += ux * push; b.by += uy * push; }
          }
        }
        for (const p of pts) { p.bx = Math.max(m, Math.min(W - m, p.bx)); p.by = Math.max(m, Math.min(H - m, p.by)); }
      }

      // label placement — strongest first (best spot); avoid all dots + placed labels
      ctx.font = LABEL_FONT;
      const placed: Box[] = [];
      features.forEach((ft, i) => {
        const p = pts[i];
        if (!(i < TOP || p.suspect)) return;
        p.labeled = true;
        const lw = ctx.measureText(ft.label).width + 12, lh = 17, gap = p.R + 9;
        p.flip = p.bx > W * 0.6;
        let chosen: Box | null = null;
        for (const k of [0, -1, 1, -2, 2, -3, 3, -4, 4]) {
          const lx = p.flip ? p.bx - gap - lw : p.bx + gap;
          const ly = p.by + k * (lh + 3) - lh / 2;
          const box = { x: lx, y: ly, w: lw, h: lh };
          if (box.x < 2 || box.x + box.w > W - 2 || box.y < 2 || box.y + box.h > H - 2) continue;
          let bad = placed.some((b) => rectsOverlap(box, b, 3));
          if (!bad) bad = pts.some((q) => circleHitsRect(q.bx, q.by, q.R + 2, box));
          if (!bad) { chosen = box; break; }
        }
        if (!chosen) { const lx = p.flip ? p.bx + gap : p.bx - gap - lw; chosen = { x: lx, y: p.by - lh / 2, w: lw, h: lh }; }
        placed.push(chosen);
        p.lox = chosen.x - p.bx; p.loy = chosen.y - p.by; p.lw = lw; p.lh = lh;
      });
    }

    function frame(time: number) {
      if (!t0) t0 = time; const el = time - t0;
      ctx.clearRect(0, 0, W, H);
      if (fieldImg) { const [px, py] = flow(cx, cy, time); ctx.drawImage(fieldImg, px * 0.5, py * 0.5, W, H); }

      // dots
      pts.forEach((a) => {
        const since = el - a.born;
        if (since > 0 && a.prog < 1) a.prog = Math.min(1, since / 620);
        const p = a.prog; if (p <= 0) { a.x = a.bx; a.y = a.by; return; }
        const ease = 1 - Math.pow(1 - p, 3), settled = p >= 1;
        const [fx, fy] = flow(a.bx, a.by, time);
        a.x = a.bx + fx * ease; a.y = a.by + fy * ease;
        const pulse = settled ? 1 : 1 + Math.sin(time / 680 + a.index) * 0.08 * ease;
        const R = a.R * ease * pulse, col = ramp(a.actN, a.suspect);
        ctx.beginPath(); ctx.fillStyle = `rgba(${col.r | 0},${col.g | 0},${col.b | 0},${0.92 * ease})`; ctx.arc(a.x, a.y, R, 0, 7); ctx.fill();
        if (a.suspect && settled) { ctx.save(); ctx.strokeStyle = "rgba(220,47,44,.8)"; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(a.x, a.y, R + 4.5 + Math.sin(time / 520 + a.index) * 1.1, 0, 7); ctx.stroke(); ctx.restore(); }
      });

      // labels on top — leader line, then pill, then text
      ctx.font = LABEL_FONT; ctx.textBaseline = "middle";
      pts.forEach((a) => {
        if (!a.labeled || a.prog < 0.8) return;
        const ease = Math.min(1, (a.prog - 0.8) / 0.2);
        const lx = a.x + a.lox, ly = a.y + a.loy, nearX = a.flip ? lx + a.lw : lx;
        ctx.save(); ctx.globalAlpha = ease;
        ctx.strokeStyle = "rgba(24,24,27,.22)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(nearX, ly + a.lh / 2); ctx.stroke();
        ctx.beginPath(); (ctx as CanvasRenderingContext2D).roundRect(lx, ly, a.lw, a.lh, 4);
        ctx.fillStyle = "rgba(255,255,255,.94)"; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,.1)"; ctx.stroke();
        ctx.fillStyle = a.suspect ? "rgba(220,47,44,.95)" : "rgba(24,24,27,.9)";
        ctx.fillText(a.label, lx + 6, ly + a.lh / 2 + 0.5);
        ctx.restore();
      });
      raf = requestAnimationFrame(frame);
    }

    function onMove(e: MouseEvent) {
      const rect = wrap!.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let hit: Pt | null = null, best = 15;
      pts.forEach((a) => { if (a.prog < 0.4) return; const d = Math.hypot(a.x - mx, a.y - my); if (d < best + a.R) { best = d; hit = a; } });
      if (hit) {
        const h = hit as Pt;
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
        <span className="right">{features.length ? <><b>{features.length}</b> of {latents.toLocaleString()} firing</> : `${latents.toLocaleString()} latents`}</span>
      </div>
      <div className="cloud" ref={wrapRef}>
        <canvas ref={canvasRef} />
        <div className="ctip" ref={tipRef} />
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--ink)" }} />firing</span>
        <span><i style={{ background: "var(--accent)" }} />unverified label</span>
        <span className="hint">distance from centre ∝ activation</span>
      </div>
    </div>
  );
}
