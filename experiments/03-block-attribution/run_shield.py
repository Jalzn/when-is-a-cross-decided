#!/usr/bin/env python
"""Shielding test for the skill claim (front 04): does the strike block's stability
survive without position features? xcross-lab front 10 showed a positional/contextual
predictor reaches temporal stability ~0.56 — if strike's 0.74 were just role/position,
the claim "situation is skill" would need reframing.

Variants (same ruler, same estimators): strike_full (26), strike_no_pos (23, without
start_x/start_y/distance_from_goal — the key test), pos_only (3 — the role floor).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
E05 = Path("~/xcross-lab/experiments/05-sequencias-de-campos").expanduser()
E01 = E05.parent / "01-campos-e-amostra"
for p in (str(HERE), str(E05), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
from run03 import _temporal_split_half, ESTIMATOR_KEYS, MIN_CROSSES  # noqa: E402

from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402
from sklearn.metrics import roc_auc_score

STRIKE_FULL = [
    "start_x", "start_y", "distance_start_from_goal",
    "attackers_in_box", "defenders_in_box", "attackers_in_zone", "defenders_in_zone", "box_ratio", "zone_ratio",
    "pressure_crosser_nearest_def", "pressure_crosser_def_within_3m",
    "marking_max_attacker_gap_in_box", "marking_free_attackers_in_box", "marking_mean_gap_in_box",
    "pocket_radius_in_box", "pocket_goal_angle_in_box",
    "coverage_kl_attack_defense_in_box", "coverage_kl_attack_defense_in_zone",
    "gk_distance_off_line", "gk_ball_distance", "gk_lateral_speed", "gk_present",
    "shape_line_height", "shape_block_width", "shape_block_area", "shape_last_line_to_gk_gap",
]
POS_COLS = ["start_x", "start_y", "distance_start_from_goal"]
VARIANTS = {
    "strike_full": STRIKE_FULL,
    "strike_no_pos": [c for c in STRIKE_FULL if c not in POS_COLS],
    "pos_only": POS_COLS,
}


def _p(cfg: dict) -> dict[str, Path]:
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = _p(cfg)

    feats = (
        pl.read_parquet(glob.glob(str(p["features_root"] / "*/*/*/features.parquet")))
        .with_columns(pl.col("cross_id").cast(pl.Utf8), pl.col("match_id").cast(pl.Utf8),
                      pl.col("crosser_player_id").cast(pl.Utf8))
        .sort("cross_id")
    )
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    df = (
        feats.select("cross_id", "crosser_player_id", "match_id", "success")
        .join(grid.select("cross_id", "start_frame"), on="cross_id", how="left")
    )
    assert df["start_frame"].null_count() == 0
    y = df["success"].to_numpy()
    groups = df["match_id"].to_numpy().astype(object)
    n_by = df.group_by("crosser_player_id").agg(pl.len().alias("n"))
    qualified = set(n_by.filter(pl.col("n") >= MIN_CROSSES)["crosser_player_id"].to_list())
    dfq = df.with_row_index("i").filter(pl.col("crosser_player_id").is_in(list(qualified)))
    idx = dfq["i"].to_numpy()

    rows = []
    for vname, cols in VARIANTS.items():
        X = feats.select(cols).to_numpy()
        for est in ESTIMATOR_KEYS:
            oof = oof_predict(ESTIMATORS[est], X, y, groups, method="sigmoid")
            auc = float(roc_auc_score(y, oof))
            d = dfq.select("crosser_player_id", "match_id", "start_frame").with_columns(pl.Series("score", oof[idx]))
            stab = _temporal_split_half(d, "score")
            rows.append({"variant": vname, "n_features": len(cols), "estimator": est,
                         "auc": auc, "stability_temporal": stab, "n_players": len(qualified)})
            print(f"  {vname}/{est}: AUC={auc:.4f} stab={stab:.3f} (n={len(qualified)})", flush=True)
    pl.DataFrame(rows).write_csv(run / "shield.csv")

    stab_v = {(r["variant"], r["estimator"]): r["stability_temporal"] for r in rows}
    auc_v = {(r["variant"], r["estimator"]): r["auc"] for r in rows}
    # critérios de blindagem
    shield_holds = bool(
        stab_v[("strike_no_pos", "adaboost")] >= 0.40
        and stab_v[("strike_no_pos", "adaboost")] - stab_v[("pos_only", "adaboost")] >= 0.10
    )
    common.update_metrics(
        run, n_crosses=len(y), n_qualified=len(qualified),
        stability=stab_v, auc=auc_v,
        shield_holds=shield_holds,
        stability_strike_full=stab_v[("strike_full", "adaboost")],
        stability_strike_no_pos=stab_v[("strike_no_pos", "adaboost")],
        stability_pos_only=stab_v[("pos_only", "adaboost")],
        stage_reached="V1", stopped_by=None,
    )
    common.record_elapsed(run, "V1", t0)
    print(f"shield_holds={shield_holds}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
