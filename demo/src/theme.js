// Sleep-stage palette — colorblind-safe, ordinal for NREM depth (light→dark),
// REM set apart in vermilion. Used consistently across scatter, hypnogram, cards.
export const STAGES = ["W", "N1", "N2", "N3", "REM"];

export const STAGE_COLOR = {
  W: "#C9CCD6", //  wake — light slate
  N1: "#7FC6F2", //  light blue
  N2: "#3B82C4", //  blue
  N3: "#22346B", //  deep navy (deep sleep)
  REM: "#E8664F", //  vermilion
};
export const STAGE_HEX = STAGES.map((s) => STAGE_COLOR[s]);
export const MASK_COLOR = "#39404E"; // unscored / masked epochs

export const STAGE_LABEL = {
  W: "Wake",
  N1: "N1",
  N2: "N2",
  N3: "N3 (deep)",
  REM: "REM",
};

// hex string -> [r,g,b,a] floats in [0,1] for regl-scatterplot
export function hexToRGBA(hex, alpha = 1) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  return [r, g, b, alpha];
}

export const stageColorList = () => STAGE_HEX.map((h) => hexToRGBA(h, 1));

// 12 distinct hues for prototype-coloured views (scatter + prototype-gram).
// Emitted as HEX so regl-scatterplot can parse them (it rejects hsl()).
function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
    : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
  const to = (v) => Math.round((v + m) * 255).toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}
export const PROTO_HUES = Array.from({ length: 12 }, (_, i) => hslToHex((i * 360) / 12 + 15, 62, 62));

// The paper's explainability claims the demo is built to demonstrate.
// SleepEDF-only, so we carry the claims a single cohort can show (A–D) + the
// prototype-gram (partial G); E/F are cross-dataset/disease → paper link only.
export const CLAIMS = {
  A: { tag: "Claim A", title: "Faithful by construction",
       short: "The model reasons only through similarity to 12 prototypes — inspecting the prototype map IS inspecting the decision, not a surrogate." },
  B: { tag: "Claim B", title: "Every decision is traceable",
       short: "Each 30-s epoch is staged via its nearest prototype; Integrated Gradients exposes the exact time–frequency evidence for the match." },
  C: { tag: "Claim C", title: "Monosemantic prototypes",
       short: "Each of the 12 prototypes is a pure, stage-specialised pattern (high label-purity and monosemanticity)." },
  D: { tag: "Claim D", title: "Clinically meaningful microstructure",
       short: "Prototype signatures follow AASM physiology — spindles for N2, delta for N3, eye movements for REM, alpha/EMG for wake." },
};
