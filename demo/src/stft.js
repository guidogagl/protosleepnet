// In-browser STFT reproducing physioex's XSleepNetSpectrogram (which wraps
// scipy.signal.spectrogram): periodic Hamming, nperseg=200, noverlap=100,
// nfft=256, detrend='constant', scaling='density', one-sided, 10*log10(|X|^2).
// Validated against the shipped stft_reference.json fixtures.

const NPERSEG = 200;
const NOVERLAP = 100;
const NFFT = 256;
const FS = 100;
const N_FREQ = NFFT / 2 + 1; // 129
const EPS = 1.1920929e-7; // np.float32 eps

// periodic Hamming window (scipy get_window(..., fftbins=True))
const WIN = (() => {
  const w = new Float32Array(NPERSEG);
  for (let n = 0; n < NPERSEG; n++) w[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / NPERSEG);
  return w;
})();
const WIN_SUMSQ = (() => {
  let s = 0;
  for (let n = 0; n < NPERSEG; n++) s += WIN[n] * WIN[n];
  return s;
})();
const SCALE = 1.0 / (FS * WIN_SUMSQ); // density scaling

// ── iterative radix-2 FFT (size must be power of two = 256) ──
const BITREV = (() => {
  const r = new Uint16Array(NFFT);
  let j = 0;
  for (let i = 1; i < NFFT; i++) {
    let bit = NFFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    r[i] = j;
  }
  return r;
})();
// precomputed twiddles
const COS = new Float32Array(NFFT / 2);
const SIN = new Float32Array(NFFT / 2);
for (let i = 0; i < NFFT / 2; i++) {
  COS[i] = Math.cos((-2 * Math.PI * i) / NFFT);
  SIN[i] = Math.sin((-2 * Math.PI * i) / NFFT);
}

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = BITREV[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= NFFT; len <<= 1) {
    const half = len >> 1;
    const step = NFFT / len;
    for (let i = 0; i < NFFT; i += len) {
      for (let k = 0; k < half; k++) {
        const tw = k * step;
        const c = COS[tw];
        const s = SIN[tw];
        const a = i + k;
        const b = a + half;
        const tr = re[b] * c - im[b] * s;
        const ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// signal: Float32Array(3000) -> Float32Array(T*F) dB, T=29, F=129 (row-major, [t][f])
export function spectrogram(signal) {
  const T = Math.floor((signal.length - NOVERLAP) / (NPERSEG - NOVERLAP)); // 29
  const out = new Float32Array(T * N_FREQ);
  const re = new Float32Array(NFFT);
  const im = new Float32Array(NFFT);
  for (let t = 0; t < T; t++) {
    const start = t * (NPERSEG - NOVERLAP);
    let mean = 0;
    for (let i = 0; i < NPERSEG; i++) mean += signal[start + i];
    mean /= NPERSEG;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < NPERSEG; i++) re[i] = (signal[start + i] - mean) * WIN[i];
    fft(re, im);
    for (let f = 0; f < N_FREQ; f++) {
      let psd = (re[f] * re[f] + im[f] * im[f]) * SCALE;
      if (f !== 0 && f !== N_FREQ - 1) psd *= 2; // one-sided
      out[t * N_FREQ + f] = 10 * Math.log10(psd + EPS);
    }
  }
  return out; // length T*N_FREQ
}

export const STFT_SHAPE = { T: 29, F: N_FREQ };

// Compare against a reference sample from stft_reference.json.
// ref: { raw: [C][S], spec: [C][T][F] }. Returns max abs dB difference.
export function validateAgainstReference(ref) {
  let maxDiff = 0;
  for (let c = 0; c < ref.raw.length; c++) {
    const mine = spectrogram(Float32Array.from(ref.raw[c]));
    const ref2d = ref.spec[c];
    for (let t = 0; t < STFT_SHAPE.T; t++)
      for (let f = 0; f < STFT_SHAPE.F; f++)
        maxDiff = Math.max(maxDiff, Math.abs(mine[t * STFT_SHAPE.F + f] - ref2d[t][f]));
  }
  return maxDiff;
}
