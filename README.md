# When Is a Cross Decided? — Research artifacts

Code, metrics, and figures for the SSAC27 research paper competition abstract
**"When Is a Cross Decided? An Information Decomposition of Crosses from Tracking Data"**
(Jalmir Ferreira, UFMG).

## What is here

- `experiments/` — the code of the 10 research fronts (Python, runs on any Linux host
  with the `xcross` package; see each `README.md` and `config.json`):
  1. `01-temporal-base` — materializes the temporal base (sequences of spatial fields
     per cross, CR−2 s → end of possession window) with fidelity verification
  2. `02-information-curve` — the information curve (AUC × information cutoff)
  3. `03-block-attribution` — block attribution (alone/union/leave-one-out) +
     per-block player-skill stability + the positional-floor control (`runs/shield`)
  4. `05-mechanism` — dense curve, attack/defense channels, regions, best pre-touch family
  5. `06-receiver-curve` — receiver/defender assignment and arrival-block stability by role
  6. `07-separation-curve` — contest separation trajectory (superseded by 08; kept for
     transparency — its identity-selection leak is documented in its `decision.md`)
  7. `08-dynamic-contestants` — leak-free dynamic contestants; receiver emergence
  8. `09-intention-execution` — destination vs strike decomposition of the early signal
  9. `10-ball-height-goalkeeper` — ball height (z) curve and goalkeeper movement
- `results/` — `metrics.json`, `decision.md`, and result tables of every front
  (the complete scientific record, including negative results and corrections)
- `figures/` — publication figures
- `ABSTRACT.md` — the submitted abstract

## Data

The study uses licensed PFF FC broadcast tracking data (25 Hz, ball z), which **cannot
be redistributed**. In line with SSAC's open-source policy, the companion repository
[github.com/Jalzn/xcross](https://github.com/Jalzn/xcross) publicly provides the
**anonymized per-cross feature tables** (96 columns incl. labels, no player
identifiers), the **out-of-fold predictions** of all models, and the full data
pipeline that regenerates every feature from raw tracking. All personal identifiers
are anonymized; player identities never appear in released data.

## Reproducing

Each front reads the canonical features/sequences produced by the `xcross` pipeline
and writes `runs/<id>/metrics.json`. Validation is always 5-fold StratifiedGroupKFold
grouped by match, with per-fold calibration and out-of-fold metrics; every comparison
uses the same estimator on both sides.

## Citation

Ferreira, J. *When Is a Cross Decided? An Information Decomposition of Crosses from
Tracking Data.* MIT Sloan Sports Analytics Conference Research Papers Competition, 2027
(submitted). Method foundation: *When the Outcome Is Noise: A Calibrated Expected-Cross
Model from Tracking Data* (MLSA @ ECML PKDD 2026).

## Note on languages

Code, READMEs, and data documentation are in English. The per-front
`results/*/decision.md` files are the original lab decision records and are kept in
the working language of the research (Portuguese) for fidelity; every key number in
them also appears in `metrics.json` and in the English experiment READMEs.
