# Decision — Front 10: ball height (z) and goalkeeper (run 20261001-0010)

## Curves alone (XGBoost, success, OOF)

| Cutoff | z-only | GK-only | (ref: ball 2D) |
|---|---|---|---|
| k0 | 0.526 | 0.511 | 0.521 |
| f010 | 0.548 | 0.508 | 0.729 |
| f050 | 0.602 | 0.585 | 0.769 |
| f100 | 0.636 | 0.597 | 0.773 |

Contrasts: z adds +0.0009 [−0.0035, +0.0052] at f010 (nothing) and +0.0120
[+0.0078, +0.0160] at f100; z+GK add +0.0066 [+0.0037, +0.0092] to the union.

## Reading
1. The early signal is HORIZONTAL: the z-curve rises slowly and contributes nothing
   in the first tenth; height only refines the arrival contest (+0.012).
2. The goalkeeper is a minor character in flight (≤0.60 alone). Structural caveat:
   visible in only 50.5% of instants (broadcast windows) — conservative reading.
   z present in 90.9% of rows.

## Decision: supported — the physics is complete: 2D direction decides early, height
seasons the arrival, the contest consumes it.
