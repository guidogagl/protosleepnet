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
