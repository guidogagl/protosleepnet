import React, { useEffect, useRef } from "react";
import { STAGES, STAGE_COLOR } from "../theme.js";

// ── viridis colormap ──
const VIRIDIS = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ── multi-channel waveform ──
export function Waveform({ channels, height = 46 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, wrap = c.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = height * channels.length;
    c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px";
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    channels.forEach((sig, ci) => {
      const y0 = ci * height, mid = y0 + height / 2;
      let max = 1e-6; for (let i = 0; i < sig.length; i++) max = Math.max(max, Math.abs(sig[i]));
      ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
      ctx.strokeStyle = "#5ac8fa"; ctx.lineWidth = 0.9; ctx.beginPath();
      for (let i = 0; i < sig.length; i++) {
        const x = (i / (sig.length - 1)) * W;
        const y = mid - (sig[i] / max) * (height / 2 - 3);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  }, [channels, height]);
  return <canvas ref={ref} className="plot" />;
}

// ── multi-channel spectrogram heatmap ──
export function Spectrogram({ specs, T, F, chanHeight = 46, maxFreqBin = 100 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, wrap = c.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = chanHeight * specs.length;
    c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px";
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const off = document.createElement("canvas"); off.width = T; off.height = maxFreqBin;
    const octx = off.getContext("2d");
    specs.forEach((spec, ci) => {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < spec.length; i++) { if (spec[i] < lo) lo = spec[i]; if (spec[i] > hi) hi = spec[i]; }
      const rng = hi - lo || 1;
      const img = octx.createImageData(T, maxFreqBin);
      for (let t = 0; t < T; t++) {
        for (let f = 0; f < maxFreqBin; f++) {
          const v = (spec[t * F + f] - lo) / rng;
          const [r, g, b] = viridis(v);
          const px = ((maxFreqBin - 1 - f) * T + t) * 4; // flip freq (low at bottom)
          img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b; img.data[px + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, ci * chanHeight, W, chanHeight);
    });
  }, [specs, T, F, chanHeight, maxFreqBin]);
  return <canvas ref={ref} className="plot" />;
}

// ── prediction probability bars ──
export function ProbBars({ proba, trueLabel, predLabel }) {
  return (
    <div className="prob">
      {STAGES.map((s, i) => {
        const v = proba[i] ?? 0;
        const isTrue = i === trueLabel;
        const isPred = i === predLabel;
        return (
          <div className={"prob row" + (isTrue ? " truth" : "")} key={s}>
            <span className="name" style={{ color: isTrue ? "#fff" : undefined }}>
              {s}{isTrue ? " ●" : ""}
            </span>
            <span className="track">
              <span className="fill" style={{
                width: `${(v * 100).toFixed(1)}%`,
                background: isPred ? STAGE_COLOR[s] : "rgba(255,255,255,0.22)",
              }} />
            </span>
            <span className="val tnum">{(v * 100).toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// ── diverging band-relevance bars ──
export function DivergingBars({ items }) {
  const max = Math.max(1e-6, ...items.map((d) => Math.abs(d.value)));
  return (
    <div className="bars">
      {items.map((d) => {
        const w = (Math.abs(d.value) / max) * 50;
        return (
          <div className="bar" key={d.name}>
            <span className="name">{d.name}</span>
            <span className="track">
              <span className="center-tick" />
              <span className={"fill " + (d.value >= 0 ? "pos" : "neg")} style={{ width: `${w}%` }} />
            </span>
            <span className="val tnum">{d.value >= 0 ? "+" : ""}{d.value.toFixed(1)}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── channel importance (EEG/EOG/EMG) ──
export function ChannelBars({ imp }) {
  const items = ["EEG", "EOG", "EMG"].map((k) => ({ name: k, value: (imp?.[k] ?? 0) * 100 }));
  const max = Math.max(1e-6, ...items.map((d) => d.value));
  return (
    <div className="bars">
      {items.map((d) => (
        <div className="bar" key={d.name}>
          <span className="name">{d.name}</span>
          <span className="track">
            <span className="fill" style={{ left: 0, width: `${(d.value / max) * 100}%`, background: "var(--accent-2)" }} />
          </span>
          <span className="val tnum">{d.value.toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

// ── EEG spectral-envelope line (dB vs frequency) ──
export function EnvelopePlot({ env, fs = 100, nfft = 256, height = 70 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, wrap = c.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = height;
    c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px";
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const db = env.map((v) => 10 * Math.log10(Math.max(v, 1e-6)));
    let lo = Math.min(...db), hi = Math.max(...db); const rng = hi - lo || 1;
    // freq axis: bins map to 0..fs/2; show up to ~40 Hz
    const maxBin = Math.min(env.length - 1, Math.round((40 / (fs / 2)) * (env.length - 1)));
    ctx.strokeStyle = "#5ac8fa"; ctx.lineWidth = 1.4; ctx.beginPath();
    for (let f = 0; f <= maxBin; f++) {
      const x = (f / maxBin) * W;
      const y = H - 4 - ((db[f] - lo) / rng) * (H - 8);
      f === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    // freq gridlines at 10/20/30/40 Hz
    ctx.fillStyle = "#626c7d"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
    [10, 20, 30, 40].forEach((hz) => {
      const x = (hz / 40) * W;
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - 10); ctx.stroke();
      ctx.fillText(hz + "Hz", x, H - 1);
    });
  }, [env, fs, nfft, height]);
  return <canvas ref={ref} className="plot" />;
}
