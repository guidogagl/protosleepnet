# Local explanation audit (per-epoch IG vs. sleep physiology)

For each featured recording we test whether the per-epoch Integrated-Gradients
relevance concentrates on the frequency band and channel that the matched
prototype's stage predicts (spindles for N2, delta for N3, EOG for REM, ...).
This is an honest sanity check of the *local* explanations, not a cherry-pick.

## protosleepnet-gagliardi (seq)

| recording | epochs | plausible | band ok | channel ok | N3 pos | REM pos |
|---|---|---|---|---|---|---|
| Recording A | 963 | 68% | 81% | 78% | 0.36 | 0.83 |
| Recording B | 982 | 66% | 83% | 73% | 0.32 | 0.62 |
| Recording C | 1011 | 65% | 73% | 74% | 0.30 | 0.58 |
| Recording D | 940 | 68% | 74% | 81% | 0.27 | 0.65 |
- Recording A: N3 precedes REM (expected) — N3 mean position 0.36, REM 0.83.
- Recording B: N3 precedes REM (expected) — N3 mean position 0.32, REM 0.62.
- Recording C: N3 precedes REM (expected) — N3 mean position 0.30, REM 0.58.
- Recording D: N3 precedes REM (expected) — N3 mean position 0.27, REM 0.65.

## protosleeptransformer-gagliardi (st)

| recording | epochs | plausible | band ok | channel ok | N3 pos | REM pos |
|---|---|---|---|---|---|---|
| Recording A | 848 | 77% | 77% | 100% | 0.28 | 0.59 |
| Recording B | 990 | 63% | 63% | 100% | 0.44 | 0.59 |
| Recording C | 1053 | 73% | 73% | 99% | 0.36 | 0.52 |
| Recording D | 963 | 70% | 70% | 100% | 0.27 | 0.57 |
- Recording A: N3 precedes REM (expected) — N3 mean position 0.28, REM 0.59.
- Recording B: N3 precedes REM (expected) — N3 mean position 0.44, REM 0.59.
- Recording C: N3 precedes REM (expected) — N3 mean position 0.36, REM 0.52.
- Recording D: N3 precedes REM (expected) — N3 mean position 0.27, REM 0.57.
