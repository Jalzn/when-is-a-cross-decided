# ABSTRACT SSAC27 — VERSÃO FINAL PARA SUBMISSÃO

Quando copiar: use SOMENTE o bloco entre as linhas "---" abaixo (sem markdown).
Contagem verificada: ver FINAL-wordcount.txt. Regra: <500 palavras incluindo título.

---

When Is a Cross Decided? An Information Decomposition of Crosses from Tracking Data

Introduction. A cross is football's most frequent low-conversion action, usually evaluated by its outcome — a metric that behaves as statistical noise at the player level. We ask a different question: when, during the life of a cross, does the outcome become decided, and which parts of it — the delivery technique, the movement of players in the box, the keeper — carry that information? The answer determines where analysts should look and whom it should credit.

Methods. We model 10,871 crosses from 1,041 matches (PFF broadcast tracking, 25 Hz, ball z; Brasileirão, Premier League, Champions League, Bundesliga) as a temporal sequence of spatial fields (120×80 grids of positional entropy and pitch control) from 2 s before the strike to the end of the possession window. At each information cutoff — approach, strike, and deciles of the ball's flight — we train gradient-boosted and bagged classifiers on only what is knowable by then: players' field trajectories, the ball's 2D path, and five information blocks at arrival (strike configuration, in-flight field evolution, ball 2D, 3D flight technique, arrival geometry). All models share one protocol: 5-fold StratifiedGroupKFold by match, per-fold calibration, out-of-fold metrics, paired bootstrap, and a single fixed estimator at every cutoff. Player skill per block is audited by temporal split-half stability of per-player means.

Results. The outcome is nearly flat before the strike (AUC 0.548→0.578 across 2 s of approach; the strike-time value reproduces our published expected-cross model) — then the flight resolves it almost immediately: AUC 0.735 within the first tenth of the ball's flight (~0.2 s) and 0.818 at arrival (+0.240 [0.229, 0.253]); two thirds of the total gain accrue in that first tenth. The ball's own 2D trajectory alone reaches 0.729 there; the box configuration resolves later and dominates the arrival (0.797 vs 0.772). Window length adds at most +0.017: content, not censoring. Decomposing the arrival model into information blocks, only two carry unique signal: 3D flight technique (leave-one-out +0.016 [0.013, 0.019]) and arrival geometry (+0.013 [0.011, 0.016]); strike configuration, the ball's 2D path, and attack- vs defense-channel fields are mutually redundant. Yet none of it is individual skill: among 155 crossers, per-player temporal stability collapses for every flight-time block (delivery technique 0.09–0.12, box movement 0.22–0.35) while the situation at the strike is the only stable signal (0.53–0.74; raw conversion −0.03); position alone explains a 0.48 floor, and situation quality beyond position retains +0.24 above it. Adding flight information to player scores reduces their stability (0.13–0.18).

Conclusion. A cross is decided in the air — early in the flight by where the ball is going, at arrival by the contest it creates — but the crosser is not: the only thing a crosser repeats is the situation from which he crosses. Evaluation should credit context creation as the repeatable individual signal, treating delivery outcomes as collective events. Code, features, and out-of-fold predictions are open; proprietary tracking is anonymized.

---

## Figuras para anexar (máx. 2)
1. `experiments/05-mechanism/runs/1/figure_curve_v2.png` — a curva de informação densa
2. `experiments/03-block-attribution/runs/1/figure_blocks.png` — blocos isolados + leave-one-out

## Repo open-source
- Preparado em `release/` deste lab (instruções no release/README.md): push como
  `Jalzn/when-is-a-cross-decided` (código das frentes 01-10 + métricas + figuras) +
  link para `Jalzn/xcross` (pipeline, features anônimas, OOF públicos do POC I).

## Trilha: Soccer (atribuída pelo conteúdo)
