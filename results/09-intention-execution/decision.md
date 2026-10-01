# Decision — Front 09: intention vs execution (run 20260930-2320)

## Decomposition of the early ball signal (XGBoost, success, OOF)

| Model | AUC |
|---|---|
| A destination (where the ball ended) | 0.737 |
| B early execution (trajectory at f010 / f020) | 0.729 / 0.746 |
| C destination + execution | 0.777 |
| D pure execution (residuals ⊥ destination) | 0.730 |

Contrasts (paired bootstrap): C−A = +0.040 [+0.034, +0.045]; A−B crosses zero
(indistinguishable); D−A ≈ −0.006 (crosses zero).

## Reading: complementarity, not domination
The strong hypothesis ("the early signal is just the choice of target") is refuted:
destination and strike carry equal-strength, complementary signals. Neither is an
individual trait (front 03: flight technique stability 0.09–0.12) — the delivery is
an event, not a virtue.

## Decision: supported — s1/s2/s3 pass with the honest complementarity reading

## Limitations
Destination is an a-posteriori quantity (descriptive). Residualization is linear OLS
(label-free). No z-axis in these features (front 10 adds it).
