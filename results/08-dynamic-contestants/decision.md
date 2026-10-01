# Decision — Front 08: dynamic contestants (run 20260930-0900)

## Three results (s1/s2/s3 = True)

### 1. The leak is confirmed and removed (s1)
| Cutoff | contaminated sep (front 07) | dynamic sep (clean) |
|---|---|---|
| k−10 | 0.694 | 0.515 |
| k0 | 0.747 | 0.518 |
| f100 | 0.879 | 0.857 |

### 2. The receiver emerges late
P(nearest now = arrival receiver): 5% (f=0.1), 18% (0.25), 28% (0.5), 100% (arrival);
3.1 identity switches per cross on average.

### 3. Separation diverges late
Outcome gap: 0.29 m (f=0.1) → 0.78 (0.5) → 1.26 (0.7) → 2.03 m (arrival).

## Two-phase mechanism (synthesis with fronts 02/05)
Phase 1 (first 10–30% of flight): the ball's trajectory already predicts the outcome
(0.73) while the contest is undifferentiated and the receiver unrecognizable.
Phase 2 (last 30–40%): identity crystallizes, separation diverges, the arrival becomes
the strongest signal (0.857; union+dynamic 0.887). The ball is the fate; the humans
are the resolution.

## Decision: supported — front 07's limitation resolved, mechanism refined

## Limitations
"Nearest to the ball" is one definition of contestant; emergence could be tested with
others. dyn f100 ≈ arrival by construction (compare at intermediate cuts). No
bootstrap CIs in this front (points are clear; full paper adds them).
