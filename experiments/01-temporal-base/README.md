# 01 — Temporal base

Materializes the temporal base for the whole study: for each of the 10,871 crosses, a
sequence of spatial fields `(T, 4, 120, 80)` float32 (attack/defense/general occupancy
+ pitch control) from 2 s before the strike (k = −10, 0.2 s steps) to the end of the
possession window, extracted from raw PFF tracking.

**Verification:** k=0 fields reproduce the 30 spatial sums of the canonical feature
table to float32 precision (max diff 5.9e-06); 100% coverage; zero invalid instants.

**Run:** `python run01.py --run <id> --stages S0,S1,S2,S3,S4,S5,S6` on the GPU host
(uses `xcross` package + the front-05 extraction modules of the companion lab).
Outputs `runs/<id>/metrics.json` with coverage, fidelity, and integrity checks.
