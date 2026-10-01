# ABSTRACT SSAC27 — FINAL VERSION FOR SUBMISSION

Copy ONLY the block between the --- lines (plain text, no markdown).
Word count verified: see FINAL-wordcount.txt. Rule: fewer than 500 words, title included.

---

When Is a Cross Decided? An Information Decomposition of Crosses from Tracking Data

Introduction. Crosses are common and rarely succeed, so clubs judge them by outcomes. Outcomes, however, are noisy: at the player level they are close to coin flips. We study the cross as a process instead. When does its outcome become predictable? And which information decides it: the strike, the ball's flight, or the players in the box?

Methods. We use broadcast tracking data (25 Hz, with ball height) for 10,871 crosses from 1,041 professional matches in four competitions. Each cross is represented as a time sequence of spatial fields, 120×80 grids of positional entropy and pitch control, from two seconds before the strike to the end of the play. At each point in time we train classifiers that see only information available up to that point: the players' fields, the ball's 2D path, and five information blocks (strike configuration, in-flight field evolution, ball 2D, 3D flight technique, arrival geometry). All models share one protocol: 5-fold cross-validation grouped by match, calibration inside each fold, out-of-fold metrics, and paired bootstrap. We audit individual skill with temporal split-half stability of per-player averages.

Results. Before the strike the outcome is barely predictable: AUC moves from 0.548 to 0.578 over the two-second approach. It then resolves almost at once. One tenth of the way into the ball's flight, about 0.2 s, AUC already reaches 0.735, and at arrival it is 0.818 (total gain +0.240 [0.229, 0.253]). Two thirds of that gain appear in the first tenth. The ball's 2D path alone reaches 0.729 there. The players in the box matter later and dominate the arrival (0.797 vs 0.772). Window length explains at most +0.017 of this. Among the five blocks, only 3D flight technique (+0.016 [0.013, 0.019]) and arrival geometry (+0.013 [0.011, 0.016]) add unique signal. None of it is stable individual skill. For 155 crossers, stability collapses for every flight block (delivery technique 0.09–0.12, box movement 0.22–0.35) while the strike situation is the only stable signal (0.53–0.74; raw conversion −0.03). Position alone gives a floor of 0.48; situation quality beyond position adds +0.24. Adding flight information to player scores makes them less stable (0.13–0.18).

Conclusion. A cross is decided in the air: early by where the ball is going, at arrival by the contest it creates. The crosser decides less of it than we thought. What he repeats is the situation he crosses from, not the balls he delivers. Player evaluation should credit context creation and treat delivery outcomes as collective events. Code, anonymized data, and all predictions are public.

---

## Figures to attach (max 2)
1. `figures/figure1_information_curve.png` — the information curve
2. `figures/figure2_block_attribution.png` — blocks alone + leave-one-out

## Repo (open-source, required)
github.com/Jalzn/when-is-a-cross-decided (+ Jalzn/xcross for the base pipeline)

## Track: Soccer
