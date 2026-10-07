"""The coherence checklist of the paper, as code that runs on any prototype record.

The paper assigns each of the 24 codebook entries a verdict (Coherent / Partially coherent /
Non-canonical) against a stage-specific checklist inspired by the AASM criteria. The published
verdicts were computed by `codice/sweep_soglie.py` in the manuscript repository, from the
per-prototype analysis reports. This module is the same rule, written to take a record built
directly from an ablation run, so that the checklist can also be applied to codebooks that are not
the published ones (the random-codebook null in `coherence_null.py`).

A record is a dict with:
    stage        predicted dominant stage of the prototype: "W", "N1", "N2", "N3" or "REM"
    direction    {band: "elevated" | "suppressed"}, sign of the band power minus the training mean
    db           {band: float}, the same difference in dB
    band_rel     {band: float}, band relevance in percent of the total EEG contribution
    ch_pcts      {"EEG": f, "EOG": f, "EMG": f}, channel importance as a fraction
    eog_pct      percentile rank of the EOG power within the codebook, 0-100
    emg_pct      percentile rank of the EMG tone within the codebook, 0-100

Bands are named delta, theta, alpha, sigma_low, sigma_high, beta_low, beta_high, gamma.
"""

from __future__ import annotations

STAGES = ["W", "N1", "N2", "N3", "REM"]
BANDS = ["delta", "theta", "alpha", "sigma_low", "sigma_high", "beta_low", "beta_high", "gamma"]

# Published thresholds. `dir_db` is the minimum |dB| for "elevated"/"suppressed"; 0 reproduces the
# published rule, which uses the sign alone.
PUBLISHED = dict(w_alpha=10.0, n1_alpha=5.0, n2_sigma=15.0, n3_delta=30.0, rem_eog=30.0,
                 perc_high=60.0, perc_low=40.0, dir_db=0.0)


def rank(value: float, values: list[float]) -> float:
    """Percentile rank within the codebook: the share of values strictly smaller, in percent.
    This is `np.searchsorted` of the published code."""
    return 100.0 * sum(1 for x in values if x < value) / len(values)


def criteria(rec: dict, thr: dict | None = None, stage: str | None = None) -> list[bool]:
    """The criteria of a stage, one per element, in the order of the manuscript."""
    s = dict(PUBLISHED, **(thr or {}))
    st = stage or rec["stage"]
    d = s["dir_db"]

    def el(b: str) -> bool:
        return rec["direction"].get(b) == "elevated" and abs(rec["db"].get(b, 0.0)) >= d

    def sup(b: str) -> bool:
        return rec["direction"].get(b) == "suppressed" and abs(rec["db"].get(b, 0.0)) >= d

    br = rec["band_rel"]
    if st == "W":
        return [el("alpha") and br.get("alpha", 0) > s["w_alpha"],
                rec["emg_pct"] > s["perc_high"], rec["eog_pct"] > s["perc_high"]]
    if st == "N1":
        return [el("theta"), sup("alpha") or br.get("alpha", 0) < s["n1_alpha"]]
    if st == "N2":
        return [el("sigma_high") or el("sigma_low"),
                br.get("sigma_high", 0) > s["n2_sigma"] or br.get("sigma_low", 0) > s["n2_sigma"],
                rec["eog_pct"] < s["perc_low"]]
    if st == "N3":
        return [el("delta"), br.get("delta", 0) > s["n3_delta"], rec["emg_pct"] < s["perc_low"]]
    if st == "REM":
        return [rec["ch_pcts"].get("EOG", 0) > s["rem_eog"] / 100.0,
                rec["emg_pct"] < s["perc_low"], el("theta")]
    raise ValueError(f"unknown stage {st!r}")


def verdict(rec: dict, thr: dict | None = None, stage: str | None = None) -> str:
    """C if every criterion holds, P if at least half do, X otherwise."""
    c = criteria(rec, thr, stage)
    n, tot = sum(c), len(c)
    return "C" if n == tot else ("P" if n >= tot / 2 else "X")
