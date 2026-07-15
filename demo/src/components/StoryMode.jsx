import React, { useEffect, useMemo, useState } from "react";
import { CLAIMS, STAGES } from "../theme.js";

// find a featured recording's order-index by its (anon) label, via subjectRanges
// (the same keyed map the app uses); falls back to the first recording
function orderOf(subjectRanges, id) {
  const k = Object.keys(subjectRanges || {}).find((k) => subjectRanges[k].id === id);
  return k != null ? Number(k) : 0;
}

// first within-subject epoch of `stage` that the model also predicts correctly
// and matches to a same-stage prototype (a clean, representative example)
function findEpoch(data, range, stageIdx) {
  if (!range) return 0;
  const { start, count } = range;
  let fallback = 0;
  for (let i = 0; i < count; i++) {
    const gi = start + i;
    if (data.label[gi] === stageIdx) {
      fallback = fallback || i;
      const dom = data.prototypes[data.proto[gi]]?.dominant_stage;
      if (data.pred[gi] === stageIdx && STAGES.indexOf(dom) === stageIdx) return i;
    }
  }
  return fallback;
}

function argmaxProto(data, key, stage) {
  let best = null, bv = -Infinity;
  data.prototypes.forEach((p) => {
    if (stage && p.dominant_stage !== stage) return;
    const v = p[key] ?? -Infinity;
    if (v > bv) { bv = v; best = p.idx; }
  });
  return best;
}

// The guided tour. Each step names a claim and arranges the app to illustrate it.
const STEPS = [
  {
    ...CLAIMS.A, color: "proto",
    run: (ctx) => { ctx.actions.clear(); },
    body: (
      <>Sleep staging here is <b>not</b> a black box. Every epoch is classified by snapping it to the
      nearest of <b>12 learned prototypes</b> (the ◆). Colour the atlas <b>by prototype</b> — the clusters
      you see <i>are</i> the model's internal vocabulary. Inspecting them is inspecting the decision itself.</>
    ),
  },
  {
    ...CLAIMS.B, color: "stage",
    run: (ctx) => {
      const order = orderOf(ctx.subjectRanges, "Recording C");
      ctx.actions.selectSubject(order);
      const range = ctx.subjectRanges?.[order];
      ctx.actions.selectEpoch(order, findEpoch(ctx.data, range, 2)); // an N2 epoch
    },
    body: (
      <>We open one night and pick an <b>N2</b> epoch. The right panel shows its true stage, the model's
      non-quantised prediction, its nearest prototype, and an <b>Integrated-Gradients</b> map of the exact
      time–frequency evidence — plus an honest badge on whether that evidence matches N2 physiology.</>
    ),
  },
  {
    ...CLAIMS.C, color: "stage",
    run: (ctx) => {
      const k = argmaxProto(ctx.data, "monosemanticity");
      if (k != null) ctx.actions.selectProto(k);
    },
    body: (
      <>Here is the most <b>monosemantic</b> prototype. Its label-purity and monosemanticity scores are high:
      it stands for one specific, recurring sleep pattern rather than a blur of several — a genuine unit of
      meaning the network reuses across the whole cohort.</>
    ),
  },
  {
    ...CLAIMS.D, color: "stage",
    run: (ctx) => {
      const k = argmaxProto(ctx.data, "label_purity", "N3");
      if (k != null) ctx.actions.selectProto(k);
    },
    body: (
      <>This deep-sleep prototype's discovered rule and band relevance concentrate on <b>δ (delta)</b> — exactly
      what AASM associates with N3. N2 prototypes key on <b>spindles (σ)</b>, REM on <b>eye movements</b>. The
      learned vocabulary lines up with textbook physiology, not arbitrary features.</>
    ),
  },
  {
    tag: "Beyond this demo", title: "Two claims need more than one cohort",
    short: "", color: "stage", run: (ctx) => ctx.actions.clear(),
    body: (
      <>The paper also shows ProtoSleepNet keeps black-box accuracy while beating post-hoc explainers, and that
      prototype-occupancy shifts track Parkinson's/Alzheimer's. Those need multiple datasets and clinical cohorts,
      so they live in the paper — this demo focuses on the single-cohort claims A–D.</>
    ),
  },
];

export default function StoryMode({ manifest, data, subjectRanges, signals, actions, onClose }) {
  const [i, setI] = useState(0);
  const ctx = useMemo(() => ({ data, subjectRanges, signals, actions }), [data, subjectRanges, signals, actions]);

  // run the current step's arrangement whenever it changes (and data is ready)
  useEffect(() => {
    if (!data || !subjectRanges) return;
    actions.setColorBy(STEPS[i].color);
    STEPS[i].run(ctx);
  }, [i, data, subjectRanges]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = STEPS[i];
  const last = i === STEPS.length - 1;
  return (
    <div className="story" role="dialog" aria-label="Guided tour">
      <div className="story-top">
        <span className="story-tag">{step.tag}</span>
        <span className="story-title">{step.title}</span>
        <button className="story-x" onClick={onClose} aria-label="Close tour">✕</button>
      </div>
      <p className="story-body">{step.body}</p>
      <div className="story-nav">
        <div className="story-dots">
          {STEPS.map((_, k) => <span key={k} className={"dot" + (k === i ? " on" : "")} onClick={() => setI(k)} />)}
        </div>
        <div className="story-btns">
          <button disabled={i === 0} onClick={() => setI(i - 1)}>Back</button>
          {last
            ? <button className="primary" onClick={onClose}>Explore freely</button>
            : <button className="primary" onClick={() => setI(i + 1)}>Next</button>}
        </div>
      </div>
    </div>
  );
}
