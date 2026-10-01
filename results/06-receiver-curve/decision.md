# Decision — Front 06: the receiver's curve (run 20260930-0720)

## Assignment (R0)
Receiver and primary defender assigned for 10,871/10,871 crosses (nearest attacker to
the ball at arrival, crosser excluded; nearest defender). Median receiver-to-ball
distance at arrival: 3.08 m.

## Whom does the arrival block belong to? (temporal split-half, XGBoost, ≥20 contests)

| Role | stability (arrival block) | stability (raw rate) | n |
|---|---|---|---|
| Receiver | 0.380 | 0.255 | 138 |
| Primary defender | 0.356 | 0.219 | 127 |
| Crosser (control) | 0.216 | −0.026 | 155 |

s1 = True (receiver − crosser = +0.165; criterion ≥ +0.10).

## Reading
1. The arrival block belongs more to whoever disputes the ball than to whoever
   crosses it.
2. It is still the noisiest link: 0.38 is far below the strike situation (0.53–0.74).
3. Being on the end of successful crosses is mildly stable (0.255) — target quality
   has some persistence; raw conversion by crosser stays noise (−0.03).
4. Internal validation: the crosser control reproduces front 03 exactly (0.21551).

## Decision: supported — the WHO now has three roles

## Limitations
Only XGBoost was recorded (a loop bug overwrote adaboost; xgboost is the family
reported across fronts). Proximity-based receiver is a proxy for the intended
target. Moderate power (n=138/127).
