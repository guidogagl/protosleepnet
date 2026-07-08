import React, { useMemo } from "react";
import { STAGES, STAGE_COLOR, STAGE_LABEL, MASK_COLOR } from "../theme.js";

const COLOR_MODES = [
  { id: "stage", label: "Sleep stage" },
  { id: "proto", label: "Prototype" },
  { id: "error", label: "Correctness" },
];

export default function Sidebar({
  data, signals, colorBy, setColorBy, subjectOrder, setSubjectOrder, selection, onSelectProto,
}) {
  const stageCounts = useMemo(() => {
    if (!data) return null;
    const c = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < data.n; i++) c[data.label[i] === 255 ? 5 : data.label[i]]++;
    return c;
  }, [data]);

  return (
    <aside className="left">
      <div className="section">
        <h2>Colour points by</h2>
        <div className="seg" style={{ display: "flex", width: "100%" }}>
          {COLOR_MODES.map((m) => (
            <button key={m.id} style={{ flex: 1 }} aria-pressed={colorBy === m.id}
              onClick={() => setColorBy(m.id)}>{m.label}</button>
          ))}
        </div>

        {colorBy === "stage" && stageCounts && (
          <div className="legend" style={{ marginTop: 13 }}>
            {STAGES.map((s, i) => (
              <div className="row" key={s}>
                <span className="swatch" style={{ background: STAGE_COLOR[s] }} />
                {STAGE_LABEL[s]}
                <span className="count tnum">{stageCounts[i].toLocaleString()}</span>
              </div>
            ))}
            <div className="row">
              <span className="swatch" style={{ background: MASK_COLOR }} />
              unscored
              <span className="count tnum">{stageCounts[5].toLocaleString()}</span>
            </div>
          </div>
        )}
        {colorBy === "error" && (
          <div className="legend" style={{ marginTop: 13 }}>
            <div className="row"><span className="swatch" style={{ background: "#46d39a" }} />correct</div>
            <div className="row"><span className="swatch" style={{ background: "#e8664f" }} />misclassified</div>
            <div className="row"><span className="swatch" style={{ background: MASK_COLOR }} />unscored</div>
          </div>
        )}
        {colorBy === "proto" && (
          <p className="faint" style={{ fontSize: 12, marginTop: 12 }}>
            Each point is coloured by the prototype it is quantised to (nearest in embedding space).
          </p>
        )}
      </div>

      <div className="section">
        <h2>Subject (night)</h2>
        <div className="field">
          <select
            value={subjectOrder ?? ""}
            onChange={(e) => setSubjectOrder(e.target.value === "" ? null : Number(e.target.value))}
          >
            <option value="">All nights ({signals.subjects.length})</option>
            {signals.subjects.map((s) => (
              <option key={s.id} value={s.order}>{s.id} · {s.n_epochs} epochs</option>
            ))}
          </select>
        </div>
        <p className="faint" style={{ fontSize: 11.5, marginTop: 9 }}>
          Selecting a night highlights its epochs in the atlas and loads its hypnogram below.
        </p>
      </div>

      <div className="section">
        <h2>Prototypes · M = 12</h2>
        {data ? (
          <div className="protolist">
            {data.prototypes.map((p) => (
              <div
                key={p.idx}
                className={"protochip" + (selection.type === "proto" && selection.k === p.idx ? " active" : "")}
                onClick={() => onSelectProto(p.idx)}
              >
                <span className="dot" style={{ background: STAGE_COLOR[p.dominant_stage] || "#888" }} />
                <span className="pk">P{p.idx}<small>{p.dominant_stage}</small></span>
              </div>
            ))}
          </div>
        ) : (
          <div className="loader">loading…</div>
        )}
      </div>
    </aside>
  );
}
