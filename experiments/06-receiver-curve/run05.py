#!/usr/bin/env python
"""Front 06 — the receiver's curve: whom does the arrival block belong to?

R0: assign primary receiver/defender per cross — players nearest the ball at the last
instant of the window (attackers of the crossing team, crosser excluded).
R1: temporal split-half stability of the arrival-block OOF aggregated by receiver,
defender, and crosser (control); raw conversion rate per role as reference.
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
E02 = Path("~/mit-sloan/02-information-curve").expanduser()
E03 = Path("~/mit-sloan/03-block-attribution").expanduser()
E05 = Path("~/xcross-lab/experiments/05-sequencias-de-campos").expanduser()
E01 = E05.parent / "01-campos-e-amostra"
for p in (str(HERE), str(E02), str(E03), str(E05), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
from run03 import _temporal_split_half, MIN_CROSSES  # noqa: E402

ESTIMATOR_KEYS = ["adaboost", "xgboost"]


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _assign_job(npz_path: str, tr_by_cross: dict, crosser_team: int, crosser_pid: int):
    """Recebedor e defensor primários: mais próximos da bola no último instante."""
    z = np.load(npz_path, allow_pickle=False)
    frames = z["frames"]
    f_arr = int(frames[-1])
    bx, by = float(z["ball_x_t"][-1]), float(z["ball_y_t"][-1])
    cid = str(z["cross_id"])
    tr = tr_by_cross.get((cid,))
    if tr is None or tr.height == 0:
        return {"cross_id": cid, "receiver_id": None, "defender_id": None,
                "receiver_dist": None, "defender_dist": None, "n_at_arrival": 0}
    at_arr = tr.filter(pl.col("frame_num") == f_arr)
    if at_arr.height == 0:
        at_arr = tr.filter(pl.col("frame_num") <= f_arr).sort("frame_num").group_by("player_id").last()
    at_arr = at_arr.with_columns(((pl.col("x") - bx) ** 2 + (pl.col("y") - by) ** 2).sqrt().alias("d"))
    att = at_arr.filter((pl.col("team_id") == crosser_team) & (pl.col("player_id") != crosser_pid)).sort("d")
    deff = at_arr.filter(pl.col("team_id") != crosser_team).sort("d")
    rec = att.row(0, named=True) if att.height else None
    dfn = deff.row(0, named=True) if deff.height else None
    return {"cross_id": cid,
            "receiver_id": int(rec["player_id"]) if rec else None,
            "defender_id": int(dfn["player_id"]) if dfn else None,
            "receiver_dist": float(rec["d"]) if rec else None,
            "defender_dist": float(dfn["d"]) if dfn else None,
            "n_at_arrival": at_arr.height}


def stage_r0(run: Path, cfg: dict) -> pl.DataFrame:
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    feats = (
        pl.read_parquet(sorted(glob.glob(str(p["features_root"] / "*/*/*/features.parquet"))))
        .with_columns(pl.col("cross_id").cast(pl.Utf8), pl.col("crosser_player_id").cast(pl.Int64))
        .select("cross_id", "crosser_player_id")
    )
    grid = grid.join(feats, on="cross_id", how="left").join(
        index.select("cross_id", "abs_path"), on="cross_id", how="left")
    assert grid["crosser_player_id"].null_count() == 0

    by_dir = {}
    for r in grid.iter_rows(named=True):
        by_dir.setdefault(r["league_dir"], {}).setdefault(r["match_id"], []).append(r)

    jobs = []
    for ldir, matches in by_dir.items():
        for mid, rows in matches.items():
            wdir = p["windows_root"] / ldir / mid
            tr = pl.read_parquet(wdir / "tracking_ext.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
            tr_by_cross = tr.partition_by("cross_id", as_dict=True)
            for r in rows:
                jobs.append((r["abs_path"], tr_by_cross, int(r["crosser_team_id"]), int(r["crosser_player_id"])))
    results = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(
        delayed(_assign_job)(*j) for j in jobs
    )
    assign = pl.DataFrame(list(results))
    assign.write_parquet(run / "assignment.parquet")
    n_rec = assign["receiver_id"].is_not_null().sum()
    common.update_metrics(run, n_crosses=assign.height, n_with_receiver=int(n_rec),
                          n_with_defender=int(assign["defender_id"].is_not_null().sum()),
                          receiver_dist_median=float(assign["receiver_dist"].median()),
                          stage_reached="R0", stopped_by=None)
    common.record_elapsed(run, "R0", t0)
    print(f"R0: recebedor atribuido em {n_rec}/{assign.height}; mediana dist recebedor = "
          f"{assign['receiver_dist'].median():.2f} m", flush=True)
    return assign


def stage_r1(run: Path, cfg: dict, assign: pl.DataFrame) -> None:
    t0 = time.time()
    p = _p(cfg)
    oof = np.load(p["front03_run"] / "oof_blocks.npz")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    df = (
        assign.with_columns(pl.col("receiver_id").cast(pl.Utf8), pl.col("defender_id").cast(pl.Utf8))
        .join(grid.select("cross_id", "match_id", "start_frame", "success"), on="cross_id", how="left")
    )
    df = df.sort("cross_id").with_row_index("i")  # ordem = cross_meta (sort por cross_id) = ordem dos arrays OOF
    # oof arrays seguem a ordem de cross_meta da frente 02/03 = sort por cross_id — verificar
    meta03 = pl.read_parquet(p["front03_run"] / "meta03.parquet").sort("cross_id")
    assert df["cross_id"].to_list() == meta03["cross_id"].to_list(), "ordem divergente vs oof"

    rows = []
    for role, col in (("receiver", "receiver_id"), ("defender", "defender_id"), ("crosser_ctrl", None)):
        if role == "crosser_ctrl":
            cf = pl.read_parquet(p["front03_run"] / "meta03.parquet").with_columns(pl.col("crosser_player_id").cast(pl.Utf8))
            df_role = df.join(cf.select("cross_id", "crosser_player_id"), on="cross_id", how="left")
            col = "crosser_player_id"
        else:
            df_role = df
        n_by = df_role.filter(pl.col(col).is_not_null()).group_by(col).agg(pl.len().alias("n"))
        qual = set(n_by.filter(pl.col("n") >= MIN_CROSSES)[col].to_list())
        dfq = df_role.filter(pl.col(col).is_in(list(qual)))
        idx = dfq["i"].to_numpy()
        for tag, key in [("arrival", None), ("raw_rate", None)]:
            if tag == "raw_rate":
                d = dfq.select(pl.col(col).alias("crosser_player_id"), "match_id", "start_frame",
                               pl.col("success").cast(pl.Float64).alias("score"))
                est = "-"
            else:
                for est in ESTIMATOR_KEYS:
                    o = oof[f"alone:arrival|success|{est}"][idx]
                    d = dfq.select(pl.col(col).alias("crosser_player_id"), "match_id", "start_frame").with_columns(pl.Series("score", o))
            rows.append({"role": role, "score": tag, "estimator": est,
                         "stability_temporal": _temporal_split_half(d, "score"), "n_players": len(qual)})
            if tag == "raw_rate":
                break
    pl.DataFrame(rows).write_csv(run / "stability_receiver.csv")
    common.update_metrics(run, stability={f"{r['role']}|{r['score']}|{r['estimator']}": r["stability_temporal"] for r in rows},
                          stage_reached="R1")
    common.record_elapsed(run, "R1", t0)
    for r in rows:
        print(f"  stab[{r['role']}/{r['score']}/{r['estimator']}]: {r['stability_temporal']:.3f} (n={r['n_players']})", flush=True)


def stage_r2(run: Path, cfg: dict) -> None:
    t0 = time.time()
    m = common.load_metrics(run)
    stab = m.get("stability", {})
    recv = max([v for k, v in stab.items() if k.startswith("receiver|arrival")], default=float("nan"))
    cross = max([v for k, v in stab.items() if k.startswith("crosser_ctrl|arrival")], default=float("nan"))
    m.update(
        s1_receiver_more_stable_than_crosser=bool(recv - cross >= 0.10),
        delta_receiver_minus_crosser=recv - cross,
        stage_reached="R2", stopped_by=None,
    )
    common.write_json(run / "metrics.json", m)
    common.record_elapsed(run, "R2", t0)
    print(f"R2: s1={m['s1_receiver_more_stable_than_crosser']} (delta={recv - cross:+.3f})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    assign = stage_r0(run, cfg)
    stage_r1(run, cfg, assign)
    stage_r2(run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
