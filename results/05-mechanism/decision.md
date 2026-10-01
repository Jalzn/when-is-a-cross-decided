# Decision — Front 05: mechanism (run 20260930-0530)

## 1. Dense curve — the decision is even earlier

Union (XGBoost, success): k0 0.578 → f010 0.735 → f025 0.758 → f050 0.782 → f100 0.818.
Two thirds of the total gain (+0.157) accrue within the first TENTH of the flight
(~0.2 s, ~2 tracking frames). Ball-only at f010: 0.729 — the ball's initial vector is
nearly all of the early signal. Fields: f010 0.603 → f100 0.797 (the box resolves
gradually, later). Robust to cutoff convention (absolute 0.4 s arm: 0.745).

## 2. Who moves: attackers or defenders? — diffuse redundancy

Channels alone: attack 0.787, shared 0.786, defense 0.752 — but removing any channel
leaves the union at 0.848–0.850. The players' information is spread and mutually
substitutable. Channel stability: 0.34 (team level). The only unique blocks remain
flight3d (−0.015) and arrival (−0.012), replicating front 03.

## 3. Where in the box? — "around the ball"

Regions alone: around 0.766, center 0.643, global 0.618, second post 0.606, first
post 0.593; removing any single region leaves 0.797 — redundant, with the area
immediately around the ball the strongest on its own.

## 4. Pre-touch with the strongest family (logistic regression): 0.559 → 0.609 at the
strike (consistent with xcross-lab front 06's 0.604). The flat left side of the curve
is real, not a feature-family artifact.

## Decision: supported — headline sharpened to "two thirds of the decision in the
first tenth of the flight"
