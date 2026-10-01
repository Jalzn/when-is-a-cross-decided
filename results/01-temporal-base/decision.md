# Decision — Front 01: temporal base (run 20260930-0115)

## Evidence

| Check | Value | Criterion | Verdict |
|---|---|---|---|
| Coverage (npz written / sample) | 10,871 / 10,871 (100%) | s1: 100% | pass |
| Instants present | 201,619 / 201,619 (100%) | — | pass |
| k=0 fidelity vs canonical features (max) | 5.89e-06 | s2: ≤ 1e-5 | pass |
| k=0 fidelity (median; frac ≤ 1e-5) | 1.47e-06; 100% | — | pass |
| frac_full_pre | 1.0000 | s3: ≥ 0.95 | pass |
| frac_end_reached / invalid instants | 1.0000 / 0 | — | pass |
| Absent CR / missing raw / K clamp | 0 / 0 / 0 | stops not triggered | pass |

Fidelity by family: max diffs concentrate in `pitch_control_*` (5.9e-06), consistent
with float32 storage of the npz against float64 features; a convention error would
show diffs around 1e-2.

Timings (24 workers): extraction 994 s, fields 293 s. Disk: 32.8 GB of npz.

## Decision: supported — fronts 02+ unblocked

## Limitations
- The anchor is the pair (h100-2 features, h100-2 sequences), both regenerated from
  the same raw data with the same code; the H100-1 cross-platform validation chain of
  the original front 05 was not re-run (that baseline was lost with the machine).
- `elapsed_s.total` in metrics.json sums only S* stages (cosmetic; individual stage
  times are correct).
