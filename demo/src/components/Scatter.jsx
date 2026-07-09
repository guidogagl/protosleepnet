import React, { useEffect, useRef, useState, useMemo } from "react";
import createScatterplot from "regl-scatterplot";
import { scaleLinear } from "d3-scale";
import { STAGES, STAGE_HEX, MASK_COLOR, STAGE_COLOR, hexToRGBA } from "../theme.js";

// 12 distinct hues for prototype-colored mode
const PROTO_HUES = Array.from({ length: 12 }, (_, i) => `hsl(${(i * 360) / 12 + 15}, 62%, 62%)`);
const GOOD = "#46d39a", BAD = "#e8664f", NEUTRAL = "#39404E";

function pointSizeFor(n) {
  if (n > 250000) return 1.6;
  if (n > 80000) return 2.2;
  if (n > 20000) return 3.2;
  return 4.2;
}

// returns { valueA:Float32Array, palette:[rgba...] } for the current colorBy
function encodeColor(data, colorBy) {
  const n = data.n;
  const va = new Float32Array(n);
  if (colorBy === "stage") {
    for (let i = 0; i < n; i++) va[i] = data.label[i] === 255 ? 5 : data.label[i];
    return { valueA: va, palette: [...STAGE_HEX, MASK_COLOR].map((h) => hexToRGBA(h)) };
  }
  if (colorBy === "proto") {
    for (let i = 0; i < n; i++) va[i] = data.proto[i];
    return { valueA: va, palette: PROTO_HUES.map((h) => h) };
  }
  // error: correct / incorrect / unscored
  for (let i = 0; i < n; i++) va[i] = data.label[i] === 255 ? 2 : data.pred[i] === data.label[i] ? 0 : 1;
  return { valueA: va, palette: [GOOD, BAD, NEUTRAL] };
}

export default function Scatter({
  data, loading, colorBy, subjectOrder, selection, epochRec, onSelectEpoch, onSelectProto,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const spRef = useRef(null);
  const xScaleRef = useRef(scaleLinear().domain([-1, 1]));
  const yScaleRef = useRef(scaleLinear().domain([-1, 1]));
  const [viewV, setViewV] = useState(0); // bump to reproject overlay
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState(null);
  const rafRef = useRef(0);
  const downRef = useRef(null);
  const dataRef = useRef(null);
  const onSelRef = useRef(null);
  const camRef = useRef(null); // regl camera view matrix (mat4, column-major)
  const sizeRef = useRef({ w: 0, h: 0 });
  dataRef.current = data;
  onSelRef.current = onSelectEpoch;

  // project a data coord (normalized to [-1,1]) to screen px via the camera
  function projectWith(m, px, py, w, h) {
    const tx = m[0] * px + m[4] * py + m[12];
    const ty = m[1] * px + m[5] * py + m[13];
    return [(tx * 0.5 + 0.5) * w, (0.5 - ty * 0.5) * h];
  }

  // create scatterplot once
  useEffect(() => {
    const canvas = canvasRef.current;
    const sp = createScatterplot({
      canvas,
      xScale: xScaleRef.current,
      yScale: yScaleRef.current,
      backgroundColor: [0, 0, 0, 0],
      pointOutlineWidth: 0,
      pointSizeSelected: 4,
      pointColorActive: hexToRGBA("#ffffff"),
      pointColorHover: hexToRGBA("#ffffff"),
      lassoOnLongPress: false,
      deselectOnEscape: true,
    });
    spRef.current = sp;
    camRef.current = sp.get("cameraView");

    sp.subscribe("view", (e) => {
      camRef.current = (e && e.camera && e.camera.view) || sp.get("cameraView") || camRef.current;
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => setViewV((v) => v + 1));
    });
    sp.subscribe("select", ({ points }) => {
      if (points && points.length) onSelectEpoch(points[0]);
    });
    sp.subscribe("pointOver", (idx) => {
      const p = sp.getScreenPosition(idx);
      if (p) setHover({ idx, x: p[0], y: p[1] });
    });
    sp.subscribe("pointOut", () => setHover(null));
    if (import.meta.env.DEV) window.__sp = sp;

    // Reliable click-to-select: regl consumes canvas mouse events and its
    // hit-test radius equals the (tiny) point size, so most clicks select
    // nothing. Attach native listeners on the wrapper and pick the nearest
    // point in screen space on a genuine click (not a drag).
    const wrap = wrapRef.current;
    const onDownN = (e) => { downRef.current = { x: e.clientX, y: e.clientY }; };
    const onUpN = (e) => {
      const d = downRef.current; downRef.current = null;
      const dd = dataRef.current;
      if (!dd || !d) return;
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return; // drag
      const rect = canvasRef.current.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const m = camRef.current || sp.get("cameraView");
      const W = rect.width, H = rect.height;
      let best = -1, bestD = Infinity;
      for (let i = 0; i < dd.n; i++) {
        const [sx, sy] = projectWith(m, dd.x[i], dd.y[i], W, H);
        const dx = sx - cx, dy = sy - cy;
        const q = dx * dx + dy * dy;
        if (q < bestD) { bestD = q; best = i; }
      }
      if (best >= 0 && bestD < 60 * 60 && onSelRef.current) onSelRef.current(best);
    };
    wrap.addEventListener("mousedown", onDownN, true);
    wrap.addEventListener("mouseup", onUpN, true);

    const ro = new ResizeObserver(() => {
      const el = wrapRef.current;
      if (!el) return;
      const w = el.clientWidth, h = el.clientHeight;
      sizeRef.current = { w, h };
      setSize({ w, h });
      try { sp.set({ width: w, height: h }); } catch {}
      camRef.current = sp.get("cameraView") || camRef.current;
      setViewV((v) => v + 1);
    });
    ro.observe(wrapRef.current);
    return () => {
      ro.disconnect(); cancelAnimationFrame(rafRef.current);
      wrap.removeEventListener("mousedown", onDownN, true);
      wrap.removeEventListener("mouseup", onUpN, true);
      sp.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // draw / recolor (also re-runs once the canvas has a real size)
  useEffect(() => {
    const sp = spRef.current;
    if (!sp || !data || !size.w) return;
    const { valueA, palette } = encodeColor(data, colorBy);
    const valueB = new Float32Array(data.n);
    const hasSubj = subjectOrder != null;
    if (hasSubj) {
      for (let i = 0; i < data.n; i++) valueB[i] = data.subj[i] === subjectOrder ? 1 : 0;
    } else {
      // overview: push unscored epochs to the background so stages read clearly
      for (let i = 0; i < data.n; i++) valueB[i] = data.label[i] === 255 ? 0 : 1;
    }
    sp.set({
      pointColor: palette,
      colorBy: "valueA",
      opacityBy: "valueB",
      opacity: hasSubj ? [0.05, 0.95] : [0.14, 0.66],
      pointSize: pointSizeFor(data.n),
    });
    sp.draw(
      { x: data.x, y: data.y, valueA, valueB },
      { zDataType: "categorical", wDataType: "categorical" }
    );
  }, [data, colorBy, subjectOrder, size.w, size.h]);

  // reflect epoch selection in the scatter (active point)
  useEffect(() => {
    const sp = spRef.current;
    if (!sp || !data) return;
    if (selection.type === "epoch") sp.select([selection.gi], { preventEvent: true });
    else sp.deselect({ preventEvent: true });
  }, [selection, data]);

  // project a normalized [-1,1] coord to screen px using the live scales
  const project = (px, py) => [xScaleRef.current(px), yScaleRef.current(py)];

  const protoMarkers = useMemo(() => {
    if (!data || !size.w) return [];
    return data.prototypes.map((p) => {
      const [cx, cy] = project(p.xy[0], p.xy[1]);
      return { k: p.idx, cx, cy, color: STAGE_COLOR[p.dominant_stage] || "#fff", stage: p.dominant_stage };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, size, viewV]);

  const selMarker = useMemo(() => {
    if (!data || !epochRec || !size.w) return null;
    const [cx, cy] = project(data.x[epochRec.gi], data.y[epochRec.gi]);
    const proto = protoMarkers.find((m) => m.k === epochRec.proto);
    return { cx, cy, proto };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, epochRec, size, viewV, protoMarkers]);

  return (
    <div className="scatter-wrap" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
      <svg className="overlay" width={size.w} height={size.h}>
        {/* line from selected epoch to its matched prototype */}
        {selMarker && selMarker.proto && (
          <line
            x1={selMarker.cx} y1={selMarker.cy} x2={selMarker.proto.cx} y2={selMarker.proto.cy}
            stroke="#ffffff" strokeOpacity="0.5" strokeWidth="1.5" strokeDasharray="3 3"
          />
        )}
        {selMarker && (
          <circle cx={selMarker.cx} cy={selMarker.cy} r="7" fill="none" stroke="#fff" strokeWidth="2" />
        )}
        {protoMarkers.map((m) => {
          const active = selection.type === "proto" && selection.k === m.k;
          const r = active ? 11 : 8;
          return (
            <g
              key={m.k}
              className="proto-marker"
              transform={`translate(${m.cx},${m.cy}) rotate(45)`}
              onClick={(e) => { e.stopPropagation(); onSelectProto(m.k); }}
              onMouseDown={(e) => e.stopPropagation()}
              onMouseUp={(e) => e.stopPropagation()}
            >
              <rect x={-r} y={-r} width={r * 2} height={r * 2} rx="3"
                fill={m.color} stroke="#0a0d13" strokeWidth="2" opacity={active ? 1 : 0.95} />
              <rect x={-r} y={-r} width={r * 2} height={r * 2} rx="3"
                fill="none" stroke={active ? "#fff" : "rgba(255,255,255,0.55)"} strokeWidth={active ? 2 : 1.25} />
              <text transform="rotate(-45)" textAnchor="middle" dy="0.35em"
                fontSize="10" fontWeight="700" fill="#0a0d13">{m.k}</text>
            </g>
          );
        })}
      </svg>

      {hover && data && (
        <div className="tooltip" style={{ left: hover.x, top: hover.y }}>
          <div><span className="k">stage </span><span className="v">{STAGES[data.label[hover.idx]] ?? "—"}</span>
            <span className="k"> · pred </span><span className="v">{STAGES[data.pred[hover.idx]]}</span></div>
          <div><span className="k">prototype </span><span className="v">P{data.proto[hover.idx]}</span></div>
        </div>
      )}

      <div className="scatter-label">
        UMAP · 128-D epoch embedding
        <span>{data ? `${data.n.toLocaleString()} epochs` : ""}</span>
      </div>
      <div className="scatter-hint">
        {loading ? "loading embeddings…"
          : "scroll to zoom · drag to pan · click a point or a ◆ prototype"}
      </div>
    </div>
  );
}
