// Data-loading layer for the precomputed static bundle.
// All heavy artifacts are little-endian typed arrays; per-epoch raw signals
// are fetched lazily via HTTP Range requests (no full-subject download).

const DATA_URL = (import.meta.env.VITE_DATA_URL || "./data").replace(/\/$/, "");

async function fetchBuffer(path) {
  const r = await fetch(`${DATA_URL}/${path}`);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.arrayBuffer();
}
async function fetchJSON(path) {
  const r = await fetch(`${DATA_URL}/${path}`);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.json();
}

export async function loadManifest() {
  return fetchJSON("manifest.json");
}

export async function loadSignalsIndex() {
  const subjects = await fetchJSON("signals/subjects.json");
  const byId = {};
  subjects.forEach((s, i) => (byId[s.id] = { ...s, order: i }));
  return { subjects, byId };
}

export async function loadSTFTReference() {
  return fetchJSON("signals/stft_reference.json");
}

// Load all per-model arrays + cards + per-subject predictions.
export async function loadModel(model) {
  const [xyB, labelB, predB, protoB, probaB, distB, subjB, epochB, prototypes, subjects] =
    await Promise.all([
      fetchBuffer(`${model}/xy.f32`),
      fetchBuffer(`${model}/label.u8`),
      fetchBuffer(`${model}/pred.u8`),
      fetchBuffer(`${model}/proto.u8`),
      fetchBuffer(`${model}/proba.u8`),
      fetchBuffer(`${model}/dist.f32`),
      fetchBuffer(`${model}/subj.u16`),
      fetchBuffer(`${model}/epoch.u16`),
      fetchJSON(`${model}/prototypes.json`),
      fetchJSON(`${model}/subjects.json`),
    ]);

  // per-prototype hybrid reconstruction (12,3,29,129) dB, optional
  const TF = 29 * 129;
  try {
    const recB = await fetchBuffer(`${model}/reconstructions.f16`);
    const rec = new Uint16Array(recB);
    prototypes.forEach((p, k) => {
      p.reconSpecs = [0, 1, 2].map((c) => {
        const out = new Float32Array(TF);
        const base = (k * 3 + c) * TF;
        for (let i = 0; i < TF; i++) out[i] = halfToFloat(rec[base + i]);
        return out;
      });
    });
  } catch {
    /* reconstruction file absent — card falls back to spectral envelope */
  }
  // Griffin-Lim reconstruction waveform (12,3,3000), optional
  try {
    const S = 3000;
    const tsB = await fetchBuffer(`${model}/recon_timeseries.f16`);
    const ts = new Uint16Array(tsB);
    prototypes.forEach((p, k) => {
      p.reconWave = [0, 1, 2].map((c) => {
        const out = new Float32Array(S);
        const base = (k * 3 + c) * S;
        for (let i = 0; i < S; i++) out[i] = halfToFloat(ts[base + i]);
        return out;
      });
    });
  } catch {
    /* time-series absent */
  }
  // per-prototype Integrated-Gradients attribution + its representative epoch
  // (12,3,29,129), optional → attach igAttr / igEpoch as [3 × Float32Array(TF)]
  const decodeGrid = (buf) => {
    const u = new Uint16Array(buf);
    return (k) => [0, 1, 2].map((c) => {
      const out = new Float32Array(TF);
      const base = (k * 3 + c) * TF;
      for (let i = 0; i < TF; i++) out[i] = halfToFloat(u[base + i]);
      return out;
    });
  };
  try {
    const [aB, eB] = await Promise.all([
      fetchBuffer(`${model}/ig_attr.f16`),
      fetchBuffer(`${model}/ig_epoch.f16`),
    ]);
    const attr = decodeGrid(aB), epoch = decodeGrid(eB);
    prototypes.forEach((p, k) => { p.igAttr = attr(k); p.igEpoch = epoch(k); });
  } catch {
    /* IG arrays absent */
  }

  const xy = new Float32Array(xyB);
  const n = xy.length / 2;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = xy[i * 2];
    y[i] = xy[i * 2 + 1];
  }
  return {
    model,
    n,
    x,
    y,
    label: new Uint8Array(labelB),
    pred: new Uint8Array(predB),
    proto: new Uint8Array(protoB),
    proba: new Uint8Array(probaB), // (n*5)
    dist: new Float32Array(distB),
    subj: new Uint16Array(subjB),
    epoch: new Uint16Array(epochB),
    prototypes,
    subjectsMeta: subjects.subjects || [],
  };
}

// ── float16 (IEEE 754 half) -> float32 ───────────────────────────────
function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// Fetch a single epoch's raw waveform via a Range request.
// Layout: signals/subjects/<id>.raw.bin = float16 (n_epochs, C, S), row-major.
export async function fetchEpochRaw(subjectId, epochIdx, C, S) {
  const bytesPerEpoch = C * S * 2;
  const start = epochIdx * bytesPerEpoch;
  const end = start + bytesPerEpoch - 1;
  const r = await fetch(`${DATA_URL}/signals/subjects/${subjectId}.raw.bin`, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (!r.ok && r.status !== 206) throw new Error(`raw ${subjectId}#${epochIdx}: ${r.status}`);
  const buf = await r.arrayBuffer();
  const u16 = new Uint16Array(buf);
  const out = [];
  for (let c = 0; c < C; c++) {
    const ch = new Float32Array(S);
    for (let i = 0; i < S; i++) ch[i] = halfToFloat(u16[c * S + i]);
    out.push(ch);
  }
  return out; // array of C Float32Array(S)
}

export { DATA_URL };
