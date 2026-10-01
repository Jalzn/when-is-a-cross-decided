# Decision — Front 02: the information curve (run 20260930-0245)

## The primary curve (union = fields + ball, XGBoost, success, OOF)

| Cutoff | AUC |
|---|---|
| k−10 (2 s before strike) | 0.548 |
| k−7 / k−4 / k−2 | 0.553 / 0.557 / 0.562 |
| k0 (strike) | 0.578 |
| f25 (quarter of flight) | 0.758 |
| f50 / f75 | 0.781 / 0.797 |
| f100 (arrival) | 0.818 |

Strike-to-arrival gain: +0.240 [0.229, 0.253] (paired bootstrap, 500). The first
quarter of flight carries 75% of the total gain.

## Findings
1. The cross is decided in the air. The approach is flat (0.548–0.578); AUC jumps
   early in the flight and keeps rising. Not a single step at the end.
2. The ball is the earliest signal: ball-only 2D goes from 0.521 at the strike to
   0.751 at quarter-flight — earlier than the players' fields (0.661).
3. The box decides the arrival: fields-only finishes at 0.797 vs 0.772 for the ball.

## Controls
Window length adds at most +0.017 (control arm with length covariates); the fields
gain without any length information is +0.16. a0 ≡ k0 exactly. The k0 value 0.578
reproduces the published xCross anchor.

## Decision: supported (s1 monotonicity, s2 gain with CI, s3 intermediate ≥ half — all pass)

## Limitations
Ball features are 2D (no z in the npz; z enters front 10). f100 includes "out"
crosses (legitimate for the arrival reading; the f25 contrast shows the signal
precedes the end). Pre-touch is weak in this feature family; front 05 tests the
strongest known family.
