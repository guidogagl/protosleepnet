import React, { useEffect, useRef } from "react";
import { PROTO_HUES, STAGE_COLOR } from "../theme.js";

// Prototype-gram: the full-night sequence of prototype activations — the paper's
// intermediate representation between the raw PSG and the coarse hypnogram
// (Claim G). One coloured cell per 30-s epoch, hue = matched prototype.
const PAD = { l: 34, r: 10, t: 4, b: 4 }; // left pad matches the Hypnogram axis

export default function PrototypeGram({ data, subjectRanges, subjectOrder, epochRec, onSelectEpoch, onSelectProto }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const geomRef = useRef(null);

  const range = subjectOrder != null && subjectRanges ? subjectRanges[subjectOrder] : null;

  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = wrap.clientWidth, H = wrap.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
      ctx.font = "10px -apple-system, sans-serif";
      ctx.textBaseline = "middle"; ctx.textAlign = "right"; ctx.fillStyle = "#626c7d";
      ctx.fillText("Pk", PAD.l - 8, PAD.t + plotH / 2);

      if (!range || !data) {
        ctx.textAlign = "center";
        ctx.fillText("Select a night to view its prototype-gram", W / 2, H / 2);
        geomRef.current = null;
        return;
      }

      const { start, count } = range;
      const cw = plotW / count;
      for (let i = 0; i < count; i++) {
        const k = data.proto[start + i];
        ctx.fillStyle = PROTO_HUES[k] || "#888";
        ctx.fillRect(PAD.l + i * cw, PAD.t, Math.ceil(cw) + 0.5, plotH);
      }
      geomRef.current = { x0: PAD.l, plotW, count };

      // selected epoch marker
      if (epochRec && epochRec.subjectOrder === subjectOrder) {
        const xx = PAD.l + (epochRec.epochIdx + 0.5) * cw;
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(xx, PAD.t - 1); ctx.lineTo(xx, H - PAD.b + 1); ctx.stroke();
      }
    }

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [data, subjectRanges, subjectOrder, epochRec, range]);

  function onClick(e) {
    const g = geomRef.current;
    if (!g) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const frac = (e.clientX - rect.left - g.x0) / g.plotW;
    const i = Math.floor(frac * g.count);
    if (i >= 0 && i < g.count) onSelectEpoch(subjectOrder, i);
  }

  // legend: which prototypes actually occur in this night
  const present = [];
  if (range && data) {
    const seen = new Set();
    for (let i = 0; i < range.count; i++) seen.add(data.proto[range.start + i]);
    [...seen].sort((a, b) => a - b).forEach((k) => present.push(k));
  }

  return (
    <div className="protogram">
      <div className="hd">
        <h2>Prototype-gram</h2>
        <span className="who faint">a night as a sequence of prototypes — between raw PSG and the hypnogram</span>
        <span className="lg pg-lg">
          {present.map((k) => (
            <button key={k} className="pgchip" title={`Prototype ${k} · ${data.prototypes[k]?.dominant_stage}`}
              onClick={() => onSelectProto(k)}>
              <i style={{ background: PROTO_HUES[k] }} />P{k}
            </button>
          ))}
        </span>
      </div>
      <div ref={wrapRef} style={{ flex: 1, minHeight: 0 }}>
        <canvas ref={canvasRef} onClick={onClick} />
      </div>
    </div>
  );
}
