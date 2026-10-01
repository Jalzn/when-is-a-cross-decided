#!/usr/bin/env python
"""Front 08 — dynamic contestants (leak-free) and receiver emergence.

At EVERY instant, the contestant is the player nearest the ball NOW (crosser excluded
from attackers) — identity may switch, no future information.

S0: per cross/instant — dynamic receiver/defender, distances, separation, switches,
convergence to the arrival receiver. S1: clean separation curve. S2: convergence
figure (when does the receiver emerge?). S3: criteria.
"""

from __future__ import annotations

import argparse
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

from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402
from sklearn.metrics import roc_auc_score

PRE_CUTS = [-10, -7, -4, -2, 0]
POST_FRACS = [0.25, 0.50, 0.75, 1.00]


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _dyn_job(npz_path: str, tr_by_cross: dict, crosser_team: int, crosser_pid: int, arrival_receiver, arrival_defender):
    z = np.load(npz_path, allow_pickle=False)
    frames, ks = z["frames"], z["k"]
    bx, by = z["ball_x_t"], z["ball_y_t"]
    cid = str(z["cross_id"])
    tr = tr_by_cross.get((cid,))
    if tr is None:
        return []
    tr = tr.sort("frame_num")
    team_of = dict(zip(tr["player_id"].to_list(), tr["team_id"].to_list()))
    piv = tr.pivot(index="frame_num", on="player_id", values=["x", "y"])
    have = piv["frame_num"].to_numpy()
    att_pids = [pid for pid, t in team_of.items() if t == crosser_team and pid != crosser_pid]
    def_pids = [pid for pid, t in team_of.items() if t != crosser_team]
    out = []
    prev_rec = prev_def = None
    for i, f in enumerate(frames):
        j = int(np.searchsorted(have, int(f), side="right")) - 1
        if j < 0:
            out.append({"cross_id": cid, "k": int(ks[i])})
            continue
        row_x = np.array([piv[f"x_{pid}"][j] for pid in att_pids], dtype=float)
        row_y = np.array([piv[f"y_{pid}"][j] for pid in att_pids], dtype=float)
        dx = np.array([piv[f"x_{pid}"][j] for pid in def_pids], dtype=float)
        dy = np.array([piv[f"y_{pid}"][j] for pid in def_pids], dtype=float)
        rec_i = np.nanargmin(np.hypot(row_x - bx[i], row_y - by[i])) if len(att_pids) else -1
        def_i = np.nanargmin(np.hypot(dx - bx[i], dy - by[i])) if len(def_pids) else -1
        rec_pid = int(att_pids[rec_i]) if rec_i >= 0 else None
        def_pid = int(def_pids[def_i]) if def_i >= 0 else None
        rec_ball = float(np.hypot(row_x[rec_i] - bx[i], row_y[rec_i] - by[i])) if rec_i >= 0 else None
        def_ball = float(np.hypot(dx[def_i] - bx[i], dy[def_i] - by[i])) if def_i >= 0 else None
        if rec_i >= 0 and def_i >= 0:
            sep = float(np.hypot(row_x[rec_i] - dx[def_i], row_y[rec_i] - dy[def_i]))
        else:
            sep = None
        out.append({"cross_id": cid, "k": int(ks[i]), "rec_dyn": rec_pid, "def_dyn": def_pid,
                    "rec_ball": rec_ball, "def_ball": def_ball, "sep_dyn": sep,
                    "adv_dyn": (def_ball - rec_ball) if (rec_ball is not None and def_ball is not None) else None,
                    "rec_switch": int(rec_pid != prev_rec), "def_switch": int(def_pid != prev_def),
                    "is_arrival_receiver": int(rec_pid == arrival_receiver) if arrival_receiver is not None else None,
                    "is_arrival_defender": int(def_pid == arrival_defender) if arrival_defender is not None else None})
        prev_rec, prev_def = rec_pid, def_pid
    return out


def stage_s0(run: Path, cfg: dict):
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    feats = (pl.read_parquet(sorted(__import__("glob").glob(str(p["features_root"] / "*/*/*/features.parquet"))))
             .with_columns(pl.col("cross_id").cast(pl.Utf8), pl.col("crosser_player_id").cast(pl.Int64))
             .select("cross_id", "crosser_player_id"))
    assign = pl.read_parquet(p["front06_run"] / "assignment.parquet").with_columns(
        pl.col("receiver_id").cast(pl.Int64), pl.col("defender_id").cast(pl.Int64))
    grid = (grid.join(feats, on="cross_id", how="left")
            .join(assign.select("cross_id", "receiver_id", "defender_id"), on="cross_id", how="left")
            .join(index.select("cross_id", "abs_path"), on="cross_id", how="left"))
    assert grid["crosser_player_id"].null_count() == 0

    by_dir = {}
    for r in grid.iter_rows(named=True):
        by_dir.setdefault(r["league_dir"], {}).setdefault(r["match_id"], []).append(r)
    jobs = []
    for ldir, matches in by_dir.items():
        for mid, rows in matches.items():
            tr = pl.read_parquet(p["windows_root"] / ldir / mid / "tracking_ext.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
            tr_by_cross = tr.partition_by("cross_id", as_dict=True)
            for r in rows:
                jobs.append((r["abs_path"], tr_by_cross, int(r["crosser_team_id"]), int(r["crosser_player_id"]),
                             r["receiver_id"], r["defender_id"]))
    results = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(delayed(_dyn_job)(*j) for j in jobs)
    rows = []
    for out in results:
        rows.extend(out)
    df = pl.DataFrame(rows)
    df.write_parquet(run / "dyn.parquet")
    common.update_metrics(run, n_crosses=df["cross_id"].n_unique(), n_rows=df.height,
                          mean_rec_switches=float(df.group_by("cross_id").agg(pl.col("rec_switch").sum())["rec_switch"].mean()),
                          stage_reached="S0", stopped_by=None)
    common.record_elapsed(run, "S0", t0)
    print(f"S0: {df['cross_id'].n_unique()} cruzamentos ({df.height} instantes); "
          f"trocas medias de recebedor dinâmico = {df.group_by('cross_id').agg(pl.col('rec_switch').sum())['rec_switch'].mean():.2f}", flush=True)
    return df


def stage_s1(run: Path, cfg: dict, df: pl.DataFrame):
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    cids = meta["cross_id"].to_list()
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)

    by_cross = df.partition_by("cross_id", as_dict=True)
    traj = {}
    for (cid,), d in by_cross.items():
        traj[cid] = d.sort("k")

    def prefix_i(cid, mode, param):
        ks = traj[cid]["k"].to_numpy()
        i_cr = int(np.where(ks == 0)[0][0]) if (ks == 0).any() else 0
        if mode == "pre":
            hit = np.where(ks == param)[0]
            return int(hit[0]) if len(hit) else i_cr
        n_post = len(ks) - 1 - i_cr
        if n_post == 0:
            return i_cr
        j = max(1, int(np.ceil(param * n_post)))
        return min(i_cr + j, len(ks) - 1)

    FEATS = ["sep_last", "sep_mean", "sep_delta", "rec_ball_last", "rec_ball_mean",
             "def_ball_last", "def_ball_mean", "adv_last", "adv_mean", "switches"]

    def feats(cid, mode, param):
        if cid not in traj:
            return {k: float("nan") for k in FEATS}
        d = traj[cid].slice(0, prefix_i(cid, mode, param) + 1)
        sep_cr_row = d.filter(pl.col("k") == 0)
        sep_cr = float(sep_cr_row["sep_dyn"][0]) if sep_cr_row.height and sep_cr_row["sep_dyn"][0] is not None else float("nan")
        sep_last = d["sep_dyn"][-1]
        return {"sep_last": float(sep_last) if sep_last is not None else float("nan"),
                "sep_mean": d["sep_dyn"].mean(), "sep_delta": float(sep_last) - sep_cr if (sep_last is not None and sep_cr == sep_cr) else float("nan"),
                "rec_ball_last": d["rec_ball"][-1], "rec_ball_mean": d["rec_ball"].mean(),
                "def_ball_last": d["def_ball"][-1], "def_ball_mean": d["def_ball"].mean(),
                "adv_last": d["adv_dyn"][-1], "adv_mean": d["adv_dyn"].mean(),
                "switches": float(d["rec_switch"].sum() + d["def_switch"].sum())}

    cut_labels = [f"k{k}" for k in PRE_CUTS] + [f"f{int(f*100):03d}" for f in POST_FRACS]
    specs = [("pre", k) for k in PRE_CUTS] + [("frac", f) for f in POST_FRACS]
    Xs = {}
    for label, (mode, param) in zip(cut_labels, specs):
        Xs[label] = pl.DataFrame([feats(cid, mode, param) for cid in cids]).to_numpy()
    for label in Xs:
        X = Xs[label]
        with np.errstate(all="ignore"):
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

    rows = [fit(f"sepdyn:{label}", Xs[label]) for label in cut_labels]
    Xu = np.load(p["front02_run"] / "X_union_f100.npz", allow_pickle=True)["X"][:, :-1]
    rows.append(fit("union+sepdyn_f100", np.hstack([Xu, Xs["f100"]])))
    pl.DataFrame(rows).write_csv(run / "sepdyn_curve.csv")
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

    # convergence: fraction of crosses where dyn == arrival receiver
    conv = (d.group_by("frac").agg(pl.col("is_arrival_receiver").mean().alias("p_dyn_is_receiver")).sort("frac"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    agg = (d.group_by("frac", "success").agg(pl.col("sep_dyn").mean().alias("sep")).filter(pl.col("frac").is_in([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])))
    piv = agg.pivot(index="frac", on="success", values="sep").sort("frac")
    c1 = "true" if "true" in piv.columns else "1.0"
    c0 = "false" if "false" in piv.columns else "0.0"
    axes[0].plot(piv["frac"], piv[c1], marker="o", color="#2ca02c", label="sucesso")
    axes[0].plot(piv["frac"], piv[c0], marker="o", color="#d62728", label="fracasso")
    axes[0].set_xlabel("fraction of the flight"); axes[0].set_ylabel("sep_dyn (m)")
    axes[0].set_title("Dynamic separation by outcome"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(conv["frac"], conv["p_dyn_is_receiver"], marker="o", color="#1f77b4")
    axes[1].set_xlabel("fraction of the flight"); axes[1].set_ylabel("P(dyn = arrival receiver)")
    axes[1].set_title("Quando o recebedor emerge"); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(run / "figure_dyn.png", dpi=150)
    conv.write_csv(run / "convergence.csv")
    piv.write_csv(run / "sepdyn_by_frac.csv")
    p_at = {str(r["frac"]): float(r["p_dyn_is_receiver"]) for r in conv.iter_rows(named=True) if r["frac"] in (0.1, 0.25, 0.5, 1.0)}
    common.update_metrics(run, p_dyn_is_receiver_at=p_at, stage_reached="S2", stopped_by=None)
    common.record_elapsed(run, "S2", t0)
    print(f"S2: P(dyn=recebedor) em f=0.1/0.25/0.5/1.0: {p_at}", flush=True)


def stage_s3(run: Path, cfg: dict):
    t0 = time.time()
    m = common.load_metrics(run)
    auc = m.get("auc_by_cut", {})
    k0 = auc.get("sepdyn:k0")
    f100 = auc.get("sepdyn:f100")
    union_new = auc.get("union+sepdyn_f100")
    s1_leak_confirm = bool(k0 is not None and k0 <= 0.68)  # pre-touch dropped => diagnosis confirmed
    s2 = bool(f100 is not None and f100 >= 0.75)
    s3 = bool(union_new is not None and union_new >= 0.85)
    m.update(s1_pre_touch_clean=s1_leak_confirm, s2_dyn_f100_strong=s2, s3_union_plus_dyn=s3,
             stage_reached="S3", stopped_by=None)
    common.write_json(run / "metrics.json", m)
    common.record_elapsed(run, "S3", t0)
    print(f"S3: s1={s1_leak_confirm} s2={s2} s3={s3}", flush=True)


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
    stage_s3(run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
