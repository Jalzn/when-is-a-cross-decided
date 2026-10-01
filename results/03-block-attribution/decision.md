# Decision — Fronts 03+04: block attribution and per-block skill (run 20260930-0410)

## WHAT — blocks at arrival (XGBoost, success, OOF)

| Block | AUC alone | LOO ΔAUC [95% CI] | Unique? |
|---|---|---|---|
| strike (25 feats) | 0.554 | +0.0001 [−0.001, +0.001] | no — redundant |
| fields_flight (90) | 0.797 | +0.0042 [+0.001, +0.007] | yes (small) |
| ball2d (7) | 0.773 | +0.0002 [−0.001, +0.001] | no — subsumed |
| 3D flight technique (13) | 0.761 | +0.0159 [+0.013, +0.019] | yes (largest) |
| arrival geometry (19) | 0.824 | +0.0131 [+0.011, +0.016] | yes |
| union (135) | 0.849 | — | — |

## WHO — temporal split-half stability by block (155 crossers, adaboost/xgboost)

| Block | Spearman |
|---|---|
| strike situation | 0.740 / 0.527 (the only stable one) |
| fields_flight | 0.350 / 0.195 |
| arrival | 0.264 / 0.216 |
| ball2d | 0.081 / 0.136 |
| flight3d | 0.090 / 0.124 |
| union | 0.183 / 0.128 |
| raw rate | −0.026 |

The "delivery technique is skill" hypothesis is refuted: what a crosser repeats is
the situation at the strike. Adding flight information makes player scores less
stable. Raw conversion replicates the MLSA noise finding.

## Shield control (`-shield/`): strike without position features = 0.718 vs a
position-only floor of 0.479 — the situation signal survives beyond role/position
(+0.24 above the floor).

## Decision: supported (s1 complementarity pass; s2/s3 as stated failed — reinterpreted
honestly: redundancy structure and inverted skill hypothesis are the findings)

## Limitations
Stability uses each block's own OOF (not conditional contribution); the chronological
key is the validated surrogate; xcross-lab fronts 11–12 documented convention
sensitivity (0.28–0.66 across setups) — read as an ordering under one ruler.
