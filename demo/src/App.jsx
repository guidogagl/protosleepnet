import React, { useEffect, useMemo, useRef, useState } from "react";
import { loadManifest, loadModel, loadSignalsIndex } from "./data.js";
import Scatter from "./components/Scatter.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Hypnogram from "./components/Hypnogram.jsx";
import RightPanel from "./components/RightPanel.jsx";

const MODEL_LABEL = {
  seq: { name: "ProtoSleepNet", sub: "SeqSleepNet backbone" },
  st: { name: "ProtoSleepTransformer", sub: "SleepTransformer backbone" },
};

export default function App() {
  const [manifest, setManifest] = useState(null);
  const [signals, setSignals] = useState(null);
  const [modelName, setModelName] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [colorBy, setColorBy] = useState("stage"); // stage | proto | error
  const [subjectOrder, setSubjectOrder] = useState(null); // signals order idx or null
  const [selection, setSelection] = useState({ type: null }); // {type:'epoch',gi} | {type:'proto',k}

  // bootstrap
  useEffect(() => {
    (async () => {
      try {
        const [m, s] = await Promise.all([loadManifest(), loadSignalsIndex()]);
        setManifest(m);
        setSignals(s);
        setModelName(Object.keys(m.models)[0]);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  // load model bundle on model change
  useEffect(() => {
    if (!modelName) return;
    let alive = true;
    setLoading(true);
    loadModel(modelName)
      .then((d) => {
        if (!alive) return;
        setData(d);
        setLoading(false);
        setSelection({ type: null });
      })
      .catch((e) => alive && setError(String(e)));
    return () => (alive = false);
  }, [modelName]);

  // per-subject contiguous ranges into the global arrays (emitted in subject order)
  const subjectRanges = useMemo(() => {
    if (!data) return null;
    const ranges = {};
    let off = 0;
    for (const s of data.subjectsMeta) {
      ranges[s.idx] = { id: s.id, start: off, count: s.n_epochs, pred: s.pred };
      off += s.n_epochs;
    }
    return ranges;
  }, [data]);

  const backbone = manifest && modelName ? manifest.models[modelName].backbone : null;

  // resolve the selected epoch into a rich record for the detail panel
  const epochRec = useMemo(() => {
    if (!data || !subjectRanges || selection.type !== "epoch") return null;
    const gi = selection.gi;
    const order = data.subj[gi];
    const r = subjectRanges[order];
    const proba = Array.from(data.proba.subarray(gi * 5, gi * 5 + 5)).map((v) => v / 255);
    return {
      gi,
      subjectOrder: order,
      subjectId: r?.id,
      epochIdx: data.epoch[gi],
      label: data.label[gi],
      pred: data.pred[gi],
      proba,
      proto: data.proto[gi],
      dist: data.dist[gi],
    };
  }, [data, subjectRanges, selection]);

  function selectEpochByGlobal(gi) {
    setSelection({ type: "epoch", gi });
    if (data) setSubjectOrder(data.subj[gi]);
  }
  function selectEpochBySubject(order, epochIdx) {
    const r = subjectRanges?.[order];
    if (!r) return;
    selectEpochByGlobal(r.start + epochIdx);
  }
  function selectProto(k) {
    setSelection({ type: "proto", k });
  }

  // dev-only hooks so the preview harness can drive the app deterministically
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    window.__demo = {
      selectSubject: (o) => { setSubjectOrder(o); setSelection({ type: null }); },
      selectEpoch: (order, ei) => selectEpochBySubject(order, ei),
      selectEpochGlobal: (gi) => selectEpochByGlobal(gi),
      selectProto,
      setModel: setModelName,
      setColorBy,
    };
  });

  if (error)
    return <div className="appmsg">Failed to load demo data — {error}</div>;
  if (!manifest || !signals)
    return <div className="appmsg">Loading atlas…</div>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>ProtoSleepNet</h1>
          <span className="sub">Explainability Atlas</span>
        </div>
        <div className="seg" role="group" aria-label="Model">
          {Object.keys(manifest.models).map((mn) => {
            const b = manifest.models[mn].backbone;
            return (
              <button
                key={mn}
                aria-pressed={mn === modelName}
                onClick={() => setModelName(mn)}
                title={MODEL_LABEL[b]?.sub}
              >
                {MODEL_LABEL[b]?.name || mn}
              </button>
            );
          })}
        </div>
        <span className="spacer" />
        <span className="badge">
          dataset <b>SleepEDF</b>
        </span>
        {data && (
          <span className="badge tnum">
            <b>{data.n.toLocaleString()}</b> epochs · <b>{data.subjectsMeta.length}</b> nights
          </span>
        )}
        {backbone && manifest.models[modelName] && (
          <span className="badge tnum">
            acc <b>{(manifest.models[modelName].accuracy * 100).toFixed(1)}%</b>
          </span>
        )}
      </header>

      <Sidebar
        manifest={manifest}
        data={data}
        signals={signals}
        colorBy={colorBy}
        setColorBy={setColorBy}
        subjectOrder={subjectOrder}
        setSubjectOrder={(o) => {
          setSubjectOrder(o);
          setSelection({ type: null });
        }}
        selection={selection}
        onSelectProto={selectProto}
      />

      <div className="center">
        <Scatter
          data={data}
          loading={loading}
          colorBy={colorBy}
          subjectOrder={subjectOrder}
          selection={selection}
          epochRec={epochRec}
          onSelectEpoch={selectEpochByGlobal}
          onSelectProto={selectProto}
        />
        <Hypnogram
          data={data}
          signals={signals}
          subjectRanges={subjectRanges}
          subjectOrder={subjectOrder}
          epochRec={epochRec}
          onSelectEpoch={selectEpochBySubject}
        />
      </div>

      <RightPanel
        manifest={manifest}
        data={data}
        backbone={backbone}
        selection={selection}
        epochRec={epochRec}
        onSelectProto={selectProto}
      />
    </div>
  );
}
