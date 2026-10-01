#!/usr/bin/env python
"""Front 05 — mechanism: dense curve, channels, regions, pre-touch family.

M1: dense curve (pre-touch every 0.2 s, flight every 10%) for fields/ball/union.
M2: at f100 — attack vs defense vs shared channel decomposition of the in-flight
fields; region-wise leave-one-out (around/first post/second post/center/global).
M3: pre-touch with logistic regression (the strongest known family) — closes the
"flat left side is a feature-family artifact" objection.
M4: submission figure v2.
"""

from __future__ import annotations

import argparse
import glob as _globmod
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
E02 = Path("~/mit-sloan/02-information-curve").expanduser()
E05 = Path("~/xcross-lab/experiments/05-sequencias-de-campos").expanduser()
E01 = E05.parent / "01-campos-e-amostra"
for p in (str(HERE), str(E02), str(E05), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import run02  # noqa: E402  (_prefix_stats, _ball_feats, prefix_end, FEATS30)

from run03 import _temporal_split_half, MIN_CROSSES  # noqa: E402
from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402
from sklearn.metrics import roc_auc_score

DENSE_PRE = list(range(-10, 1))          # cada 0,2 s
DENSE_FRACS = [i / 10 for i in range(1, 11)]  # 10%..100%
CURVES = ["fields", "ball", "union"]
REGIONS = ["around", "in_first_post", "in_second_post", "in_center_box", ("sum", "grad_towards_goal")]


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _build_cuts(run: Path, cfg: dict):
    """Matrizes X para os cortes densos, na ordem de cids do instant_sums."""
    t0 = time.time()
    p = _p(cfg)
    sums = pl.read_parquet(p["front02_run"] / "instant_sums.parquet")
    recs = {}
    for (cid,), df in sums.partition_by("cross_id", as_dict=True).items():
        df = df.sort("k")
        recs[cid] = {"sums": df.select(run02.FEATS30).to_numpy(), "k": df["k"].to_numpy(),
                     "ball_x": df["ball_x"].to_numpy(), "ball_y": df["ball_y"].to_numpy()}
    for cid, rec in recs.items():
        rec["i_cr"] = int(np.where(rec["k"] == 0)[0][0])
    cids = sorted(recs.keys())

    specs = ([("pre", k, f"k{k}") for k in DENSE_PRE] + [("frac", f, f"f{int(f*100):03d}") for f in DENSE_FRACS])
    out = {}
    for mode, param, label in specs:
        mats = {"fields": [], "ball": []}
        for cid in cids:
            rec = recs[cid]
            last_i = run02.prefix_end(cid, recs, mode, param) if False else _prefix_end(rec, mode, param)
            snap, mean, rng = run02._prefix_stats(rec, rec["k"], last_i)
            bf = run02._ball_feats(rec, last_i, rec["i_cr"])
            mats["fields"].append(list(snap) + list(mean) + list(rng))
            mats["ball"].append([bf[v] for v in sorted(bf)])
        Xf, Xb = np.array(mats["fields"]), np.array(mats["ball"])
        out[("fields", label)] = Xf
        out[("ball", label)] = Xb
        out[("union", label)] = np.hstack([Xf, Xb])
    for (curve, label), X in out.items():
        np.savez_compressed(run / f"D_{curve}_{label}.npz", X=X, cids=np.array(cids, dtype=object))
    common.update_metrics(run, n_dense_cuts=len(out), n_crosses=len(cids), stage_reached="M0")
    common.record_elapsed(run, "M0", t0)
    print(f"M0: {len(out)} matrizes densas", flush=True)
    return cids


def _prefix_end(rec, mode, param):
    k = rec["k"]; i_cr = rec["i_cr"]
    if mode == "pre":
        hit = np.where(k == param)[0]
        return int(hit[0]) if len(hit) else i_cr
    n_post = len(k) - 1 - i_cr
    if n_post == 0:
        return i_cr
    j = max(1, int(np.ceil(param * n_post)))
    return min(i_cr + j, len(k) - 1)


def _fit(run, tag, X, y, groups, est):
    oof = oof_predict(ESTIMATORS[est], X, y, groups, method="sigmoid")
    auc = float(roc_auc_score(y, oof))
    print(f"  {tag}/{est}: AUC={auc:.4f}", flush=True)
    return {"tag": tag, "estimator": est, "auc": auc, "oof": oof}


def stage_m1(run: Path, cfg, cids):
    from joblib import Parallel, delayed
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)
    jobs = []
    for curve in CURVES:
        for label in [f"k{k}" for k in DENSE_PRE] + [f"f{int(f*100):03d}" for f in DENSE_FRACS]:
            X = np.load(run / f"D_{curve}_{label}.npz", allow_pickle=True)["X"]
            jobs.append((f"{curve}:{label}", X))
    results = Parallel(n_jobs=cfg["n_fit_jobs"], return_as="generator")(
        delayed(_fit)(run, tag, X, y, groups, "xgboost") for tag, X in jobs
    )
    rows, oof_store = [], {}
    for r in results:
        oof_store[f"{r['tag']}|xgboost"] = r.pop("oof")
        rows.append(r)
    pl.DataFrame(rows).write_csv(run / "curve_dense.csv")
    np.savez_compressed(run / "oof_dense.npz", **oof_store)
    common.update_metrics(run, n_fits_dense=len(rows), stage_reached="M1")
    common.record_elapsed(run, "M1", t0)


def _channel_cols():
    """Índices das colunas do X fields (90 = 30 snap | 30 mean | 30 range, ordem FEATS30)."""
    att = [i for i, c in enumerate(run02.FEATS30) if c.startswith("entropy_attack_") or c.startswith("entropy_diff_")]
    deff = [i for i, c in enumerate(run02.FEATS30) if c.startswith("entropy_defense_")]
    shared = [i for i, c in enumerate(run02.FEATS30) if c.startswith("entropy_general_") or c.startswith("pitch_control_")]
    cols = {"attack": [], "defense": [], "shared": []}
    for base in (0, 30, 60):
        cols["attack"] += [base + i for i in att]
        cols["defense"] += [base + i for i in deff]
        cols["shared"] += [base + i for i in shared]
    return cols


def _region_cols():
    cols = {}
    for name in ["around", "in_first_post", "in_second_post", "in_center_box", "global"]:
        if name == "global":
            idx = [i for i, c in enumerate(run02.FEATS30) if c.endswith("_sum") or c.endswith("_grad_towards_goal")]
        else:
            idx = [i for i, c in enumerate(run02.FEATS30) if c.endswith(name)]
        cols[name] = [base + i for base in (0, 30, 60) for i in idx]
    return cols


def stage_m2(run: Path, cfg):
    from joblib import Parallel, delayed
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)

    Xf = np.load(p["front02_run"] / "X_fields_f100.npz", allow_pickle=True)["X"][:, :-1]
    Xb = np.load(p["front02_run"] / "X_ball_f100.npz", allow_pickle=True)["X"][:, :-1]
    feats = (
        pl.read_parquet(sorted(_globmod.glob(str(p["features_root"] / "*/*/*/features.parquet"))))
        .with_columns(pl.col("cross_id").cast(pl.Utf8)).sort("cross_id")
    )
    from run03 import BLOCKS_CANONICAL
    Xc = {b: feats.select(cols).to_numpy() for b, cols in BLOCKS_CANONICAL.items()}

    ch = _channel_cols()
    Xch = {f"fields_{k}": Xf[:, v] for k, v in ch.items()}
    rg = _region_cols()
    Xrg = {f"region_{k}": Xf[:, v] for k, v in rg.items()}

    mechs = {**Xch, "ball2d": Xb, "flight3d": Xc["flight3d"], "arrival": Xc["arrival"], "strike": Xc["strike"]}
    mech_names = list(mechs.keys())
    Xu = np.hstack([mechs[m] for m in mech_names])

    jobs = [("union_mech", Xu)] + [(f"alone:{m}", mechs[m]) for m in mech_names] + \
           [(f"loo_minus:{m}", np.hstack([mechs[o] for o in mech_names if o != m])) for m in mech_names]
    jobs += [(f"region_alone:{m}", Xrg[m]) for m in Xrg] + \
            [(f"region_loo:{m}", np.hstack([Xf[:, v] for kk, v in rg.items() if kk != m])) for m in Xrg]
    store = {}
    results = Parallel(n_jobs=cfg["n_fit_jobs"], return_as="generator")(
        delayed(_fit)(run, tag, X, y, groups, "xgboost") for tag, X in jobs
    )
    rows = []
    for r in results:
        store[f"{r['tag']}|xgboost"] = r.pop("oof")
        rows.append(r)
    pl.DataFrame(rows).write_csv(run / "mechanism.csv")
    # estabilidade por canal (adaboost)
    stab_rows = []
    feats_pid = feats.select("cross_id", "crosser_player_id", "match_id")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    chrono = feats_pid.join(grid.select("cross_id", "start_frame"), on="cross_id", how="left")
    n_by = chrono.group_by("crosser_player_id").agg(pl.len().alias("n"))
    qual = set(n_by.filter(pl.col("n") >= MIN_CROSSES)["crosser_player_id"].to_list())
    dfq = chrono.with_row_index("i").filter(pl.col("crosser_player_id").is_in(list(qual)))
    idx = dfq["i"].to_numpy()
    for m in ["fields_attack", "fields_defense"]:
        oof = oof_predict(ESTIMATORS["adaboost"], mechs[m], y, groups, method="sigmoid")
        d = dfq.select("crosser_player_id", "match_id", "start_frame").with_columns(pl.Series("score", oof[idx]))
        stab_rows.append({"tag": f"alone:{m}", "estimator": "adaboost", "stability_temporal": _temporal_split_half(d, "score")})
    pl.DataFrame(stab_rows).write_csv(run / "stability_mech.csv")
    np.savez_compressed(run / "oof_mech.npz", **store)
    common.update_metrics(run, n_fits_mech=len(rows) + len(stab_rows), stage_reached="M2")
    common.record_elapsed(run, "M2", t0)


def stage_m3(run: Path, cfg):
    from joblib import Parallel, delayed
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)
    jobs = []
    for label in [f"k{k}" for k in DENSE_PRE]:
        X = np.load(run / f"D_fields_{label}.npz", allow_pickle=True)["X"]
        jobs.append((f"pre_lr:{label}", X))
    store = {}
    results = Parallel(n_jobs=cfg["n_fit_jobs"], return_as="generator")(
        delayed(_fit)(run, tag, X, y, groups, "logreg") for tag, X in jobs
    )
    rows = []
    for r in results:
        store[f"{r['tag']}|logreg"] = r.pop("oof")
        rows.append(r)
    pl.DataFrame(rows).write_csv(run / "pre_lr.csv")
    common.update_metrics(run, n_fits_pre=len(rows), stage_reached="M3")
    common.record_elapsed(run, "M3", t0)


def stage_m4(run: Path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    dense = pl.read_csv(run / "curve_dense.csv")
    pre_lr = pl.read_csv(run / "pre_lr.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"fields": "#1f77b4", "ball": "#d62728", "union": "#222222"}
    for curve in CURVES:
        rows = dense.filter((pl.col("tag").str.starts_with(curve)) & (pl.col("estimator") == "xgboost")).sort("tag")
        xs, ys = [], []
        for r in rows.iter_rows(named=True):
            label = r["tag"].split(":")[1]
            xs.append(int(label[1:]) * 0.2 if label.startswith("k") else int(label[1:]) / 10 * cfg["mean_flight_s"])
            ys.append(r["auc"])
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=colors[curve], label=f"{curve} (XGBoost)")
    rows = pre_lr.sort("tag")
    xs = [int(r["tag"].split(":")[1][1:]) * 0.2 for r in rows.iter_rows(named=True)]
    ys = [r["auc"] for r in rows.iter_rows(named=True)]
    ax.plot(xs, ys, marker="s", ms=4, lw=1.5, ls="--", color="#2ca02c", label="fields pre-touch (LogReg)")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvspan(0, cfg["mean_flight_s"], alpha=0.06, color="blue")
    ax.text(cfg["mean_flight_s"] * 0.5, 0.56, "voo da bola", ha="center", fontsize=9, color="#3355aa")
    ax.set_xlabel("time relative to the strike (s; flight as fraction x mean duration)")
    ax.set_ylabel("AUC (OOF, success)")
    ax.set_title("When is a cross decided — information curve (10,871 crosses)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "figure_curve_v2.png", dpi=150)
    common.update_metrics(run, stage_reached="M4", stopped_by=None)
    common.record_elapsed(run, "M4", t0)


STAGES = {"M0": None, "M1": None, "M2": None, "M3": None, "M4": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    cids = _build_cuts(run, cfg)
    stage_m1(run, cfg, cids)
    stage_m2(run, cfg)
    stage_m3(run, cfg)
    stage_m4(run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
