# 03 — Block attribution (WHAT) and per-block skill (WHO)

Decomposes the arrival model into five information blocks — strike configuration,
in-flight field evolution, ball 2D, 3D flight technique, arrival geometry — and fits
each block alone, the union, and leave-one-out (paired bootstrap CIs).

Skill audit: per-player means of each block's out-of-fold score (crossers with ≥20
crosses), temporal split-half stability.

**Headline:** only 3D flight technique (+0.016) and arrival geometry (+0.013) carry
unique signal; the only stable player-level signal is the strike situation (0.53–0.74)
— delivery technique is not (0.09–0.12). `run_shield.py` shows the strike block
without position features retains 0.72 vs a 0.48 position-only floor.

**Run:** `python run03.py --run <id> --stages U0,U1,U2,U3`; then `run_shield.py --run <id>`.
