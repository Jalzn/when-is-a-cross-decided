# 07 — Contest separation (SUPERSEDED by 08 — kept for transparency)

Tracks the receiver–defender separation through the flight. Produced the strongest
single feature (0.879 alone at arrival) and the "gap from the start" figure — which
turned out to be an **identity-selection leak**: the tracked players were chosen by
proximity to the ball at arrival (future information). The leak is diagnosed in
`results/07-curva-separacao/decision.md` and eliminated in front 08.

**Run:** `python run06.py --run <id>`.
