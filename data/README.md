# Data (anonymized derived tables)

These are the **derived, anonymized tables** that support every result of the paper.
They are regenerable from licensed PFF FC tracking by the pipeline in
[github.com/Jalzn/xcross](https://github.com/Jalzn/xcross) plus the front runners in
`experiments/` of this repository. Raw tracking data is **not** redistributed (license);
all player/match identifiers are replaced by deterministic SHA-1 pseudonyms (12 hex
chars), so joins between files still work while no individual can be identified.

## Files

| File | Rows × Cols | What it is |
|---|---|---|
| `crosses_features_anon.parquet` | 10,871 × 96 | One row per cross: labels (`success`, `shot_in_window`), leagues/seasons, the 88 model features (strike-time configuration, spatial field sums, 3D ball flight, arrival geometry), plus anonymized `cross_id`, `match_id`, `crosser_id` |
| `instant_sums_anon.parquet` | 201,619 × 34 | The temporal backbone: per cross and per instant (k = −10…end, 0.2 s steps) the 30 spatial field sums (entropy/pitch control over 6 pitch regions), ball x/y. This is the information curve's raw material |
| `assignment_anon.parquet` | 10,871 × 6 | Primary receiver/defender per cross (nearest to ball at arrival) + distances |
| `contestants_dynamic_anon.parquet` | 201,619 × 11 | Leak-free dynamic contestants per instant: nearest attacker/defender to the ball now, distances, separation, identity switches, convergence to the arrival receiver |

## Reproducibility map

- Information curve (Fig. 1): `instant_sums_anon` + labels from `crosses_features_anon`
- Block attribution and stability (Fig. 2): `crosses_features_anon` (blocks by column families) + `instant_sums_anon`
- Receiver/emergence analyses: `assignment_anon` + `contestants_dynamic_anon`
- Every experiment's exact metrics: `results/<front>/metrics.json` (with `decision.md` documenting criteria, deviations, and corrections)

## License

Code: MIT. Derived tables: released for research reproducibility under the SSAC
open-source policy; the underlying tracking data remains property of PFF FC.
