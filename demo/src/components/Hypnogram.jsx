import React, { useEffect, useRef } from "react";
import { STAGE_COLOR } from "../theme.js";

// hypnogram row order (top→bottom): Wake, REM, N1, N2, N3
const ROW_OF = { 0: 0, 4: 1, 1: 2, 2: 3, 3: 4 }; // stageIdx -> row
const ROW_LABEL = ["W", "REM", "N1", "N2", "N3"];
const PAD = { l: 34, r: 10, t: 6, b: 16 };

export default function Hypnogram({ data, signals, subjectRanges, subjectOrder, epochRec, onSelectEpoch }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const geomRef = useRef(null);

  const hasSubj = subjectOrder != null && subjectRanges && subjectRanges[subjectOrder];
  const subjInfo = hasSubj ? signals.subjects[subjectOrder] : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
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
      const rowY = (r) => PAD.t + (r + 0.5) * (plotH / 5);

      // gridlines + row labels
      ctx.font = "10px -apple-system, sans-serif";
      ctx.textBaseline = "middle";
      for (let r = 0; r < 5; r++) {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.beginPath(); ctx.moveTo(PAD.l, rowY(r)); ctx.lineTo(W - PAD.r, rowY(r)); ctx.stroke();
        ctx.fillStyle = "#626c7d"; ctx.textAlign = "right";
        ctx.fillText(ROW_LABEL[r], PAD.l - 8, rowY(r));
      }

      if (!hasSubj || !data) {
        ctx.fillStyle = "#626c7d"; ctx.textAlign = "center";
        ctx.fillText("Select a night to view its hypnogram", W / 2, H / 2);
        geomRef.current = null;
        return;
      }

      const labels = subjInfo.labels;
      const pred = subjectRanges[subjectOrder].pred;
      const n = labels.length;
      const x = (i) => PAD.l + (i / Math.max(1, n - 1)) * plotW;
      geomRef.current = { x0: PAD.l, plotW, n };

      // predicted (thin, dim) — draw first, underneath
      ctx.strokeStyle = "rgba(90,200,250,0.5)"; ctx.lineWidth = 1;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const r = ROW_OF[pred[i]]; if (r == null) { started = false; continue; }
        const yy = rowY(r), xx = x(i);
        if (!started) { ctx.moveTo(xx, yy); started = true; } else { ctx.lineTo(xx, yy); }
      }
      ctx.stroke();

      // true hypnogram — stepped, coloured by stage
      ctx.lineWidth = 2;
      for (let i = 0; i < n - 1; i++) {
        const s = labels[i]; const r = ROW_OF[s]; if (r == null) continue;
        ctx.strokeStyle = STAGE_COLOR[["W", "N1", "N2", "N3", "REM"][s]] || "#888";
        const yy = rowY(r);
        ctx.beginPath(); ctx.moveTo(x(i), yy); ctx.lineTo(x(i + 1), yy);
        const s2 = labels[i + 1]; const r2 = ROW_OF[s2];
        if (r2 != null) ctx.lineTo(x(i + 1), rowY(r2));
        ctx.stroke();
      }

      // mismatch ticks (bottom)
      ctx.strokeStyle = "rgba(232,102,79,0.55)"; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (labels[i] === -1 || labels[i] === 255) continue;
        if (pred[i] !== labels[i]) { ctx.moveTo(x(i), H - PAD.b + 1); ctx.lineTo(x(i), H - PAD.b + 5); }
      }
      ctx.stroke();

      // selected epoch marker
      if (epochRec && epochRec.subjectOrder === subjectOrder) {
        const xx = x(epochRec.epochIdx);
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(xx, PAD.t - 2); ctx.lineTo(xx, H - PAD.b + 2); ctx.stroke();
      }
    }

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [data, signals, subjectRanges, subjectOrder, epochRec, hasSubj, subjInfo]);

  function onClick(e) {
    const g = geomRef.current;
    if (!g || !hasSubj) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const frac = (px - g.x0) / g.plotW;
    const i = Math.round(frac * (g.n - 1));
    if (i >= 0 && i < g.n) onSelectEpoch(subjectOrder, i);
  }

  return (
    <div className="hypno">
      <div className="hd">
        <h2>Hypnogram</h2>
        <span className="who">{hasSubj ? subjInfo.id : "—"}</span>
        <span className="lg">
          <span><i style={{ borderColor: "#3B82C4" }} />true stage</span>
          <span><i style={{ borderColor: "rgba(90,200,250,0.7)" }} />predicted</span>
          <span><i style={{ borderColor: "rgba(232,102,79,0.7)" }} />error</span>
        </span>
      </div>
      <div ref={wrapRef} style={{ flex: 1, minHeight: 0 }}>
        <canvas ref={canvasRef} onClick={onClick} />
      </div>
    </div>
  );
}
