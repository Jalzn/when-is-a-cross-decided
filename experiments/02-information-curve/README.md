# 02 — The information curve (WHEN)

Trains a single fixed classifier at each information cutoff — approach (k = −10…0),
flight quartiles — on only what is knowable by then, and plots AUC vs cutoff.

Three feature families per cutoff: players' field sums (`fields`), ball 2D path
(`ball`), both (`union`); plus a window-length control arm (absolute-time cuts with
length covariates). Protocol: 5-fold StratifiedGroupKFold by match, per-fold sigmoid
calibration, out-of-fold AUC, paired bootstrap (500).

**Headline:** strike 0.578 (reproduces the published xCross) → first flight tenth
0.735 → arrival 0.818 (+0.240 [0.229, 0.253]); two thirds of the gain in the first
tenth. Length adds at most +0.017.

**Run:** `python run02.py --run <id> --stages T0,T1,T2,T3`.
