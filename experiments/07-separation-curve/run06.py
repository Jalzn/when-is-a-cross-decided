#!/usr/bin/env python
"""Front 07 — contest separation through the flight (SUPERSEDED by front 08).

With receiver/defender assigned (front 06), tracks their distance (and each one's
distance to the ball) at every instant. NOTE: the tracked identities are chosen by
proximity at ARRIVAL (future information) — pre-strike values of the separation curve
are therefore identity-contaminated and must not be read as predictive; the leak is
diagnosed in this front's decision.md and eliminated in front 08 (dynamic
contestants). Kept for transparency and for the (post-flight) mechanism analysis.
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
E06 = Path("~/mit-sloan/06-receiver-curve").expanduser()
E05 = Path("~/xcross-lab/experiments/05-sequencias-de-campos").expanduser()
E01 = E05.parent / "01-campos-e-amostra"
for p in (str(HERE), str(E02), str(E03), str(E06), str(E05), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402

from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402
from sklearn.metrics import roc_auc_score

PRE_CUTS = [-10, -7, -4, -2, 0]
POST_FRACS = [0.25, 0.50, 0.75, 1.00]


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _traj_job(npz_path: str, tr_by_cross: dict, receiver_id, defender_id):
    z = np.load(npz_path, allow_pickle=False)
    frames, ks = z["frames"], z["k"]
    bx, by = z["ball_x_t"], z["ball_y_t"]
    cid = str(z["cross_id"])
    tr = tr_by_cross.get((cid,))
    out = []
    if tr is None or receiver_id is None or defender_id is None:
        return []
    # posições por jogador, com forward-fill para frames ausentes
    tr = tr.sort("frame_num")
    piv = tr.pivot(index="frame_num", on="player_id", values=["x", "y"])
    have = piv["frame_num"].to_numpy()
    def pos_at(pid, f):
        j = int(np.searchsorted(have, f, side="right") - 1)
        if j < 0:
            return None
        try:
            x = piv[f"x_{pid}"][j]; y = piv[f"y_{pid}"][j]
            x, y = float(x), float(y)
            return (x, y) if (x == x and y == y) else None  # NaN check
        except Exception:
            return None
    for i, f in enumerate(frames):
        pr = pos_at(receiver_id, int(f))
        pd_ = pos_at(defender_id, int(f))
        if pr is None or pd_ is None:
            continue
        out.append({"cross_id": cid, "k": int(ks[i]),
                    "sep": float(np.hypot(pr[0] - pd_[0], pr[1] - pd_[1])),
                    "rec_ball": float(np.hypot(pr[0] - bx[i], pr[1] - by[i])),
                    "def_ball": float(np.hypot(pd_[0] - bx[i], pd_[1] - by[i]))})
    return out


def stage_s0(run: Path, cfg: dict):
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    assign = pl.read_parquet(p["front06_run"] / "assignment.parquet").with_columns(
        pl.col("receiver_id").cast(pl.Int64), pl.col("defender_id").cast(pl.Int64))
    grid = grid.join(assign.select("cross_id", "receiver_id", "defender_id"), on="cross_id", how="left")
    grid = grid.join(index.select("cross_id", "abs_path"), on="cross_id", how="left")

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
                if r["receiver_id"] is None or r["defender_id"] is None:
                    continue
                jobs.append((r["abs_path"], tr_by_cross, int(r["receiver_id"]), int(r["defender_id"])))
    results = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(
        delayed(_traj_job)(*j) for j in jobs
    )
    rows = []
    for out in results:
        rows.extend(out)
    df = pl.DataFrame(rows)
    df.write_parquet(run / "separation.parquet")
    common.update_metrics(run, n_crosses_with_traj=df["cross_id"].n_unique(), n_rows=df.height,
                          stage_reached="S0", stopped_by=None)
    common.record_elapsed(run, "S0", t0)
    print(f"S0: {df['cross_id'].n_unique()} cruzamentos com trajetoria de separacao ({df.height} instantes)", flush=True)
    return df


def _feats_for_cut(dfp: pl.DataFrame, last_k_index: int, ks: np.ndarray):
    """last_k_index: índice do último k incluído no prefixo (ordem crescente de k)."""
    pre = dfp.slice(0, last_k_index + 1)
    sep_last = pre["sep"][-1]
    sep_cr = pre.filter(pl.col("k") == 0)["sep"]
    sep_cr = float(sep_cr[0]) if sep_cr.len() else float(pre["sep"][0])
    rb_last = pre["rec_ball"][-1]
    db_last = pre["def_ball"][-1]
    return {
        "sep_last": float(sep_last), "sep_mean": float(pre["sep"].mean()), "sep_max": float(pre["sep"].max()),
        "sep_delta": float(sep_last) - sep_cr, "sep_cr": sep_cr,
        "rec_ball_last": float(rb_last), "rec_ball_mean": float(pre["rec_ball"].mean()),
        "def_ball_last": float(db_last), "def_ball_mean": float(pre["def_ball"].mean()),
        "adv_rec": float(db_last) - float(rb_last),  # >0: recebedor mais perto da bola
    }


def stage_s1(run: Path, cfg: dict, df: pl.DataFrame):
    from joblib import Parallel, delayed
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    cids = meta["cross_id"].to_list()
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)

    by_cross = df.partition_by("cross_id", as_dict=True)
    traj = {}
    for (cid,), d in by_cross.items():
        d = d.sort("k")
        traj[cid] = {"ks": d["k"].to_numpy(), "df": d}

    def prefix_i(cid, mode, param):
        ks = traj[cid]["ks"]
        i_cr = int(np.where(ks == 0)[0][0]) if (ks == 0).any() else 0
        if mode == "pre":
            hit = np.where(ks == param)[0]
            return int(hit[0]) if len(hit) else i_cr
        n_post = len(ks) - 1 - i_cr
        if n_post == 0:
            return i_cr
        j = max(1, int(np.ceil(param * n_post)))
        return min(i_cr + j, len(ks) - 1)

    cut_labels = [f"k{k}" for k in PRE_CUTS] + [f"f{int(f*100):03d}" for f in POST_FRACS]
    Xs = {}
    for label, (mode, param) in zip(cut_labels, [("pre", k) for k in PRE_CUTS] + [("frac", f) for f in POST_FRACS]):
        rows = []
        for cid in cids:
            if cid not in traj:
                rows.append({kk: float("nan") for kk in
                             ["sep_last", "sep_mean", "sep_max", "sep_delta", "sep_cr",
                              "rec_ball_last", "rec_ball_mean", "def_ball_last", "def_ball_mean", "adv_rec"]})
                continue
            rows.append(_feats_for_cut(traj[cid]["df"], prefix_i(cid, mode, param), traj[cid]["ks"]))
        Xs[label] = pl.DataFrame(rows).to_numpy()
    # imputa nan com mediana de coluna (poucos casos sem trajetoria)
    for label in Xs:
        X = Xs[label]
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
        Xs[label] = X

    def fit(tag, X):
        oof = oof_predict(ESTIMATORS["xgboost"], X, y, groups, method="sigmoid")
        auc = float(roc_auc_score(y, oof))
        print(f"  {tag}: AUC={auc:.4f}", flush=True)
        return {"tag": tag, "auc": auc}

    rows = [fit(f"sep:{label}", Xs[label]) for label in cut_labels]
    # união + sep no f100: adiciona?
    Xu = np.load(p["front02_run"] / "X_union_f100.npz", allow_pickle=True)["X"][:, :-1]
    Xusep = np.hstack([Xu, Xs["f100"]])
    rows.append(fit("union_f100", Xu))
    rows.append(fit("union+sep_f100", Xusep))
    pl.DataFrame(rows).write_csv(run / "sep_curve.csv")
    common.update_metrics(run, auc_by_cut={r["tag"]: r["auc"] for r in rows}, stage_reached="S1")
    common.record_elapsed(run, "S1", t0)


def stage_s2(run: Path, cfg: dict, df: pl.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    meta = pl.read_parquet(_p(cfg)["front02_run"] / "cross_meta.parquet").sort("cross_id")
    d = df.join(meta.select("cross_id", "success"), on="cross_id", how="left").filter(pl.col("k") > 0).sort(["cross_id", "k"])
    n_post_by = d.group_by("cross_id").agg(pl.len().alias("n_post"))
    d = d.join(n_post_by, on="cross_id")
    d = d.with_columns(((pl.int_range(1, pl.len() + 1).over("cross_id")) / pl.col("n_post")).alias("frac"))
    agg = (d.group_by("frac", "success").agg(pl.col("sep").mean().alias("sep_mean"), pl.len().alias("n"))
           .filter(pl.col("frac").is_in([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])))
    piv = agg.pivot(index="frac", on="success", values="sep_mean").sort("frac")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(piv["frac"], piv["1.0"] if "1.0" in piv.columns else piv["true"], marker="o", label="sucesso", color="#2ca02c")
    ax.plot(piv["frac"], piv["0.0"] if "0.0" in piv.columns else piv["false"], marker="o", label="fracasso", color="#d62728")
    ax.set_xlabel("fração do voo da bola"); ax.set_ylabel("separação recebedor–defensor (m)")
    ax.set_title("O duelo aéreo ao longo do voo — quando os grupos divergem?")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(run / "figure_separation.png", dpi=150)
    piv.write_csv(run / "sep_by_frac_success.csv")
    # divergência: primeira fração com diferença >= 0.3 m
    diffs = (piv.with_columns((pl.col("1.0") - pl.col("0.0")).alias("diff")) if "1.0" in piv.columns
             else piv.with_columns((pl.col("true") - pl.col("false")).alias("diff")))
    div = diffs.filter(pl.col("diff").abs() >= 0.3)
    first_div = float(div["frac"][0]) if div.height else None
    common.update_metrics(run, first_divergence_frac=first_div,
                          sep_mean_by_frac={str(r["frac"]): { "succ": r.get("1.0", r.get("true")), "fail": r.get("0.0", r.get("false")) } for r in piv.iter_rows(named=True)},
                          stage_reached="S2", stopped_by=None)
    common.record_elapsed(run, "S2", t0)
    print(f"S2: primeira divergencia >=0.3m na fracao {first_div}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    df = stage_s0(run, cfg)
    stage_s1(run, cfg, df)
    stage_s2(run, cfg, df)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
