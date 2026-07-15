import React, { useEffect, useState } from "react";
import { STAGES, STAGE_COLOR, STAGE_LABEL, CLAIMS } from "../theme.js";
import { fetchEpochRaw, fetchEpochIG, fetchPlausibility } from "../data.js";
import { spectrogram, STFT_SHAPE } from "../stft.js";
import { Waveform, Spectrogram, ProbBars, DivergingBars, ChannelBars, EnvelopePlot } from "./plots.jsx";

const BAND_GREEK = {
  delta: "δ", theta: "θ", alpha: "α", sigma_low: "σ⁻", sigma_high: "σ⁺",
  beta_low: "β⁻", beta_high: "β⁺", gamma: "γ", mains: "mains",
};

// small caption naming the paper claim a panel demonstrates
function ClaimTag({ id }) {
  const c = CLAIMS[id];
  if (!c) return null;
  return (
    <div className="claimtag" title={c.short}>
      <span className="ct-tag">{c.tag}</span>
      <span className="ct-title">{c.title}</span>
    </div>
  );
}

// honest per-epoch verdict: did IG land on the stage's expected band + channel?
function PlausibilityBadge({ rec, proto }) {
  if (!rec) return null;
  const ok = rec.ok === 1;
  const band = BAND_GREEK[rec.tb] || rec.tb;
  const exp = (rec.ex || []).map((b) => BAND_GREEK[b] || b).join(", ");
  const topCh = rec.ch
    ? Object.entries(rec.ch).sort((a, b) => b[1] - a[1])[0][0]
    : null;
  let msg;
  if (ok) {
    msg = <>IG concentrates on <b>{band}</b> — consistent with {rec.st} physiology{exp && <> (expected {exp})</>}.</>;
  } else if (rec.bok && !rec.cok) {
    msg = <>IG lands on the right band (<b>{band}</b> for {rec.st}) but is weighted to the <b>{topCh}</b> channel — an honest caveat, shown not hidden.</>;
  } else {
    msg = <>IG peaks on <b>{band}</b>, outside {rec.st}'s expected bands{exp && <> ({exp})</>} — an honest miss, shown not hidden.</>;
  }
  return (
    <div className={"plausible " + (ok ? "ok" : "off")}>
      <span className="pl-dot" />
      <span className="pl-txt">{msg}</span>
    </div>
  );
}

function StagePill({ stage }) {
  return <span className="pill" style={{ background: STAGE_COLOR[stage] || "#888" }}>{stage}</span>;
}

// IG relevance is sparse/signed → show |attribution| normalized to its 99th
// percentile (clip outliers) so the driving regions are legible.
function normRelevance(specs) {
  const all = [];
  specs.forEach((ch) => { for (let i = 0; i < ch.length; i++) all.push(Math.abs(ch[i])); });
  all.sort((a, b) => a - b);
  const p99 = all[Math.floor(all.length * 0.99)] || 1;
  return specs.map((ch) => ch.map((v) => Math.min(Math.abs(v) / p99, 1)));
}

function bandItems(card) {
  const names = card.band_names || Object.keys(card.band_relevance_pct || {});
  return names.map((b) => ({ name: BAND_GREEK[b] || b, value: card.band_relevance_pct?.[b] ?? 0 }));
}

function LabelDistribution({ dist }) {
  const total = Object.values(dist || {}).reduce((a, b) => a + b, 0) || 1;
  return (
    <div style={{ display: "flex", height: 10, borderRadius: 5, overflow: "hidden", gap: 1 }}>
      {STAGES.map((s) => {
        const v = (dist?.[s] || 0) / total;
        return v > 0 ? <span key={s} title={`${s}: ${(v * 100).toFixed(0)}%`}
          style={{ width: `${v * 100}%`, background: STAGE_COLOR[s] }} /> : null;
      })}
    </div>
  );
}

export function PrototypeCard({ card, compact }) {
  if (!card) return null;
  return (
    <div>
      <div className="card-head">
        <StagePill stage={card.dominant_stage} />
        <h3>Prototype {card.idx}</h3>
      </div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
        dominant stage <b style={{ color: "#fff" }}>{STAGE_LABEL[card.dominant_stage]}</b>
        {card.label_purity != null && <> · {(card.label_purity * 100).toFixed(0)}% pure</>}
      </div>

      {!compact && <ClaimTag id="C" />}

      {!compact && (
        <>
          <div className="kv"><span className="k">Monosemanticity</span><span className="v tnum">{card.monosemanticity?.toFixed(2) ?? "—"}</span></div>
          <div className="kv"><span className="k">Cohort epochs</span><span className="v tnum">{(card.cohort_cluster_size ?? 0).toLocaleString()}</span></div>
          <div className="kv"><span className="k">Peak EEG band</span><span className="v">{BAND_GREEK[card.peak_band_eeg] || card.peak_band_eeg || "—"}</span></div>
          {card.cross?.plausibility?.mean != null &&
            <div className="kv"><span className="k">Plausibility</span><span className="v tnum">{card.cross.plausibility.mean.toFixed(2)}</span></div>}
          {card.cross?.stability_cv != null &&
            <div className="kv"><span className="k">Cross-dataset CV</span><span className="v tnum">{card.cross.stability_cv.toFixed(2)}</span></div>}
        </>
      )}

      <div className="block">
        <h4>Cohort label mix</h4>
        <LabelDistribution dist={card.cohort_label_distribution} />
      </div>

      {!compact && card.reconSpecs && (
        <div className="block">
          <h4>Hybrid reconstruction · EEG / EOG / EMG</h4>
          <Spectrogram specs={card.reconSpecs} T={STFT_SHAPE.T} F={STFT_SHAPE.F} />
          <p className="faint" style={{ fontSize: 11, margin: "5px 0 0" }}>
            Optimised prototypical input (median of the 256 hybrid reconstructions) —
            the spectrogram the model treats as the essence of this prototype.
          </p>
        </div>
      )}

      {!compact && card.reconWave && (
        <div className="block">
          <h4>Reconstructed signal · phase-estimated</h4>
          <Waveform channels={card.reconWave} />
          <p className="faint" style={{ fontSize: 11, margin: "5px 0 0" }}>
            30 s waveform from the medoid reconstruction via inverse STFT
            (phase is not stored → estimated by Griffin-Lim; shape is indicative).
          </p>
        </div>
      )}

      {!compact && card.igAttr && card.igEpoch && (
        <div className="block">
          <h4>Why this prototype · IG attribution</h4>
          <div className="chan"><h5>representative epoch</h5>
            <Spectrogram specs={card.igEpoch} T={STFT_SHAPE.T} F={STFT_SHAPE.F} /></div>
          <div className="chan" style={{ marginTop: 8 }}><h5>attribution (relevance)</h5>
            <Spectrogram specs={normRelevance(card.igAttr)}
              T={STFT_SHAPE.T} F={STFT_SHAPE.F} cmap="inferno" /></div>
          <p className="faint" style={{ fontSize: 11, margin: "5px 0 0" }}>
            Integrated Gradients on <b>−‖encode(x)−p<sub>{card.idx}</sub>‖²</b>: the bright
            time–frequency regions are what pull an epoch toward this prototype.
          </p>
        </div>
      )}

      <div className="block">
        <h4>Discovered rule</h4>
        {!compact && <ClaimTag id="D" />}
        <div className="rule">
          {["s1", "s2", "s3", "s4"].map((s) => card.rule?.[s] && (
            <div className="s" key={s}>{card.rule[s]}</div>
          ))}
        </div>
      </div>

      {!compact && card.spectral_envelope && (
        <div className="block">
          <h4>EEG spectral signature</h4>
          <EnvelopePlot env={card.spectral_envelope[0]} />
        </div>
      )}

      <div className="block">
        <h4>EEG band relevance (Δ to match)</h4>
        <DivergingBars items={bandItems(card)} />
      </div>

      <div className="block">
        <h4>Channel importance</h4>
        <ChannelBars imp={card.channel_importance_pct} />
      </div>
    </div>
  );
}

function EpochDetail({ manifest, data, epochRec, onSelectProto }) {
  const [raw, setRaw] = useState(null);
  const [specs, setSpecs] = useState(null);
  const [ig, setIg] = useState(null);
  const [plaus, setPlaus] = useState(null);
  const [err, setErr] = useState(null);
  const C = manifest.channels.length, S = manifest.raw.n_samples;

  useEffect(() => {
    let alive = true; setRaw(null); setSpecs(null); setIg(null); setErr(null);
    fetchEpochRaw(epochRec.subjectId, epochRec.epochIdx, C, S)
      .then((chs) => {
        if (!alive) return;
        setRaw(chs);
        setSpecs(chs.map((c) => spectrogram(c)));
      })
      .catch((e) => alive && setErr(String(e)));
    fetchEpochIG(data.model, epochRec.subjectId, epochRec.epochIdx)
      .then((g) => alive && setIg(normRelevance(g)))
      .catch(() => {});
    return () => (alive = false);
  }, [data.model, epochRec.subjectId, epochRec.epochIdx, C, S]);

  // per-recording plausibility audit (cached); index by within-subject epoch
  useEffect(() => {
    let alive = true; setPlaus(null);
    fetchPlausibility(data.model, epochRec.subjectId)
      .then((arr) => alive && setPlaus(arr ? arr[epochRec.epochIdx] : null))
      .catch(() => {});
    return () => (alive = false);
  }, [data.model, epochRec.subjectId, epochRec.epochIdx]);

  const matched = data.prototypes[epochRec.proto];
  const trueStage = epochRec.label === 255 ? null : STAGES[epochRec.label];

  return (
    <div>
      <div className="card-head">
        {trueStage ? <StagePill stage={trueStage} /> : <span className="pill" style={{ background: "#39404E", color: "#e7ebf2" }}>?</span>}
        <h3>Epoch {epochRec.epochIdx}</h3>
      </div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
        {epochRec.subjectId} · true label <b style={{ color: "#fff" }}>{trueStage || "unscored"}</b>
      </div>

      <div className="block" style={{ marginTop: 4 }}>
        <h4>Model prediction (non-quantised)</h4>
        <ProbBars proba={epochRec.proba} trueLabel={epochRec.label === 255 ? -1 : epochRec.label} predLabel={epochRec.pred} />
      </div>

      <div className="block">
        <h4>Prototype match</h4>
        <div className="match-line">
          <span>nearest →</span>
          <span className="pill" style={{ background: STAGE_COLOR[matched?.dominant_stage] || "#888", cursor: "pointer" }}
            onClick={() => onSelectProto(epochRec.proto)}>P{epochRec.proto} · {matched?.dominant_stage}</span>
          <span className="dist-chip tnum">L2 = {epochRec.dist.toFixed(2)}</span>
        </div>
        <p className="faint" style={{ fontSize: 11.5, margin: "2px 0 0" }}>
          Why: this epoch's embedding is closest to P{epochRec.proto}, which encodes —
        </p>
        {matched?.rule?.s2 && <div className="rule" style={{ marginTop: 6 }}><div className="s">{matched.rule.s2}</div></div>}
      </div>

      <div className="block">
        <h4>Input signal · EEG / EOG / EMG</h4>
        {err && <div className="faint">signal unavailable — {err}</div>}
        {!raw && !err && <div className="loader">loading 30 s epoch…</div>}
        {raw && <Waveform channels={raw} />}
      </div>

      <div className="block">
        <h4>Spectrogram — this epoch vs prototype P{epochRec.proto}</h4>
        <div className="chan"><h5>input epoch ({trueStage || "unscored"})</h5>
          {specs && <Spectrogram specs={specs} T={STFT_SHAPE.T} F={STFT_SHAPE.F} />}
          {!specs && !err && <div className="loader">computing STFT…</div>}</div>
        {matched?.reconSpecs && (
          <div className="chan" style={{ marginTop: 8 }}>
            <h5>matched prototype P{epochRec.proto} · {matched.dominant_stage} (reconstruction)</h5>
            <Spectrogram specs={matched.reconSpecs} T={STFT_SHAPE.T} F={STFT_SHAPE.F} />
          </div>
        )}
        <p className="faint" style={{ fontSize: 11, margin: "5px 0 0" }}>
          The match is by L2 in the 128-D embedding — the two spectrograms should share
          the features that define P{epochRec.proto}.
        </p>
      </div>

      <div className="block">
        <h4>Why this epoch → P{epochRec.proto} · IG attribution</h4>
        <ClaimTag id="B" />
        {ig ? <Spectrogram specs={ig} T={STFT_SHAPE.T} F={STFT_SHAPE.F} cmap="inferno" />
          : <div className="loader">loading attribution…</div>}
        <PlausibilityBadge rec={plaus} proto={epochRec.proto} />
        <p className="faint" style={{ fontSize: 11, margin: "5px 0 0" }}>
          Integrated Gradients on <b>−‖encode(x)−p<sub>{epochRec.proto}</sub>‖²</b> for
          <b> this epoch</b>: the bright time–frequency regions are what pull it toward P{epochRec.proto}.
        </p>
      </div>

      <div className="block">
        <button className="input" style={{ cursor: "pointer", textAlign: "center" }}
          onClick={() => onSelectProto(epochRec.proto)}>
          Inspect prototype P{epochRec.proto} →
        </button>
      </div>
    </div>
  );
}

function Overview({ manifest, data, backbone }) {
  const meta = data ? manifest.models[data.model] : null;
  return (
    <div className="rp">
      <div className="block" style={{ marginTop: 0 }}>
        <h4>The atlas</h4>
        <ClaimTag id="A" />
        <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6 }}>
          Each point is one 30-second epoch, placed by a PaCMAP of ProtoSleepNet's
          128-dimensional embedding space. The <b style={{ color: "#fff" }}>◆ diamonds</b> are the
          model's <b style={{ color: "#fff" }}>12 learned prototypes</b>. Every stage decision is made
          by snapping an epoch to its nearest prototype — so this map <i>is</i> the model's reasoning.
          Click any point to see its true stage, the prediction, and why it matches — or click a
          prototype to read the physiological rule it encodes.
        </p>
      </div>
      {meta && (
        <div className="block">
          <h4>Model</h4>
          <div className="kv"><span className="k">Backbone</span><span className="v">{backbone === "seq" ? "SeqSleepNet" : "SleepTransformer"}</span></div>
          <div className="kv"><span className="k">Epochs</span><span className="v tnum">{data.n.toLocaleString()}</span></div>
          <div className="kv"><span className="k">Nights</span><span className="v tnum">{data.subjectsMeta.length}</span></div>
          <div className="kv"><span className="k">Accuracy</span><span className="v tnum">{(meta.accuracy * 100).toFixed(1)}%</span></div>
          <div className="kv"><span className="k">Prototypes</span><span className="v tnum">12</span></div>
          {meta.nn_proto_agreement != null && (
            <div className="kv"><span className="k">PaCMAP L2 faithfulness</span>
              <span className="v tnum">{(meta.nn_proto_agreement * 100).toFixed(0)}%</span></div>
          )}
        </div>
      )}
      {meta?.nn_proto_agreement != null && (
        <p className="faint" style={{ fontSize: 11, marginTop: 4 }}>
          Faithfulness = share of epochs whose <i>nearest prototype on this 2-D map</i> is also
          the true nearest in 128-D. Projections are lossy, so read the actual match from the
          point colour (Prototype mode) and the detail panel — not raw proximity.
        </p>
      )}
    </div>
  );
}

export default function RightPanel({ manifest, data, backbone, selection, epochRec, onSelectProto }) {
  if (!data) return <aside className="right"><div className="rp"><div className="loader">loading…</div></div></aside>;
  let body;
  if (selection.type === "proto") body = <div className="rp"><PrototypeCard card={data.prototypes[selection.k]} /></div>;
  else if (selection.type === "epoch" && epochRec)
    body = <div className="rp"><EpochDetail manifest={manifest} data={data} epochRec={epochRec} onSelectProto={onSelectProto} /></div>;
  else body = <Overview manifest={manifest} data={data} backbone={backbone} />;
  return <aside className="right">{body}</aside>;
}
