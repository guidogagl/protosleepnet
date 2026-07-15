# Local explanation audit (per-epoch IG vs. sleep physiology)

For each featured recording we test whether the per-epoch Integrated-Gradients
relevance concentrates on the frequency band and channel that the matched
prototype's stage predicts (spindles for N2, delta for N3, EOG for REM, ...).
This is an honest sanity check of the *local* explanations, not a cherry-pick.

## protosleepnet-gagliardi (seq)

| recording | epochs | plausible | band ok | channel ok | N3 pos | REM pos |
|---|---|---|---|---|---|---|
| Recording A | 1910 | 52% | 56% | 81% | 0.20 | 0.23 |
| Recording B | 1583 | 55% | 69% | 68% | 0.52 | - |
| Recording C | 843 | 76% | 82% | 90% | 0.45 | 0.69 |
| Recording D | 1134 | 67% | 76% | 80% | 0.54 | 0.63 |
- Recording A: N3 precedes REM (expected) — N3 mean position 0.20, REM 0.23.
- Recording C: N3 precedes REM (expected) — N3 mean position 0.45, REM 0.69.
- Recording D: N3 precedes REM (expected) — N3 mean position 0.54, REM 0.63.

## protosleeptransformer-gagliardi (st)

| recording | epochs | plausible | band ok | channel ok | N3 pos | REM pos |
|---|---|---|---|---|---|---|
| Recording A | 1910 | 36% | 36% | 95% | 0.20 | 0.33 |
| Recording B | 1583 | 36% | 36% | 90% | 0.52 | 0.73 |
| Recording C | 843 | 67% | 69% | 92% | 0.42 | 0.60 |
| Recording D | 1134 | 64% | 65% | 95% | 0.57 | 0.57 |
- Recording A: N3 precedes REM (expected) — N3 mean position 0.20, REM 0.33.
- Recording B: N3 precedes REM (expected) — N3 mean position 0.52, REM 0.73.
- Recording C: N3 precedes REM (expected) — N3 mean position 0.42, REM 0.60.
- Recording D: REM precedes N3 (atypical) — N3 mean position 0.57, REM 0.57.
