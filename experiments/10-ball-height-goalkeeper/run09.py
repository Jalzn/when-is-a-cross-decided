#!/usr/bin/env python
"""Front 10 — ball height (z) and the goalkeeper in flight.

Z0: per cross/instant — ball z (windows ball_ext) and defending-GK position (roster
position_group == "GK" of the defending team; tracking_ext), forward-filled.
Z1: z-block and gk-block curves; ball2d+z vs ball2d; union+z+gk vs union.
Z2: mechanism figure — mean z and GK-ball distance by outcome through the flight.
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

CUTS = [("pre", 0, "k0"), ("frac", 0.10, "f010"), ("frac", 0.25, "f025"), ("frac", 0.50, "f050"), ("frac", 1.00, "f100")]
_NO_GK = 100.0


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _z_job(npz_path: str, bl_by_cross: dict, tr_by_cross: dict, gk_pid, crosser_team: int, hl: float):
    z_np = np.load(npz_path, allow_pickle=False)
    frames, ks = z_npz = (z_np["frames"], z_np["k"])
    bx, by = z_np["ball_x_t"], z_np["ball_y_t"]
    cid = str(z_np["cross_id"])
    bl = bl_by_cross.get((cid,))
    tr = tr_by_cross.get((cid,))
    out = []
    z_series = {}
    if bl is not None:
        bl = bl.sort("frame_num")
        bf = bl["frame_num"].to_numpy()
        bz = bl["z"].to_numpy()
        last = np.nan
        for i, f in enumerate(frames):
            j = int(np.searchsorted(bf, int(f), side="right")) - 1
            if j >= 0:
                last = bz[j]
            z_series[i] = last
    gk_xy = {}
    if tr is not None and gk_pid is not None:
        tr = tr.sort("frame_num")
        piv = tr.pivot(index="frame_num", on="player_id", values=["x", "y"])
        have = piv["frame_num"].to_numpy()
        for i, f in enumerate(frames):
            j = int(np.searchsorted(have, int(f), side="right")) - 1
            if j >= 0:
                try:
                    gk_xy[i] = (float(piv[f"x_{gk_pid}"][j]), float(piv[f"y_{gk_pid}"][j]))
                except Exception:
                    pass
    for i in range(len(frames)):
        z = z_series.get(i, np.nan)
        gk = gk_xy.get(i)
        out.append({
            "cross_id": cid, "k": int(ks[i]), "z": None if z != z else float(z),
            "gk_ball": None if gk is None else float(np.hypot(gk[0] - bx[i], gk[1] - by[i])),
            "gk_off_line": None if gk is None else float(hl - gk[0]),
            "gk_y": None if gk is None else gk[1],
        })
    return out


def stage_z0(run: Path, cfg: dict):
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    grid = grid.join(index.select("cross_id", "abs_path"), on="cross_id", how="left")

    by_dir = {}
    for r in grid.iter_rows(named=True):
        by_dir.setdefault(r["league_dir"], {}).setdefault(r["match_id"], []).append(r)
    jobs = []
    for ldir, matches in by_dir.items():
        for mid, rows in matches.items():
            wdir = p["windows_root"] / ldir / mid
            bl = pl.read_parquet(wdir / "ball_ext.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
            tr = pl.read_parquet(wdir / "tracking_ext.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
            roster = pl.read_parquet(Path(rows[0]["processed_dir"]) / "roster.parquet")
            gk_def = roster.filter((pl.col("position_group") == "GK") & (pl.col("team_id") != rows[0]["crosser_team_id"]))
            gk_pid = int(gk_def["player_id"][0]) if gk_def.height else None
            bl_by = bl.partition_by("cross_id", as_dict=True)
            tr_by = tr.partition_by("cross_id", as_dict=True)
            for r in rows:
                jobs.append((r["abs_path"], bl_by, tr_by, gk_pid, int(r["crosser_team_id"]), float(r["half_length"])))
    results = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(delayed(_z_job)(*j) for j in jobs)
    rows = []
    for out in results:
        rows.extend(out)
    df = pl.DataFrame(rows)
    df.write_parquet(run / "zgk.parquet")
    common.update_metrics(run, n_crosses=df["cross_id"].n_unique(),
                          frac_z_present=float(df["z"].is_not_null().mean()),
                          frac_gk_present=float(df["gk_ball"].is_not_null().mean()),
                          stage_reached="Z0", stopped_by=None)
    common.record_elapsed(run, "Z0", t0)
    print(f"Z0: {df['cross_id'].n_unique()} cruzamentos; z presente {df['z'].is_not_null().mean():.3f}; "
          f"gk presente {df['gk_ball'].is_not_null().mean():.3f}", flush=True)
    return df


def stage_z1(run: Path, cfg: dict, df: pl.DataFrame):
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

    ZF = ["z_last", "z_cr", "z_delta", "z_max_so_far", "k_since_apex", "z_slope", "z_mean"]
    GF = ["gk_ball_last", "gk_ball_mean", "gk_ball_delta", "gk_off_last", "gk_off_delta", "gk_y_last", "gk_disp"]

    def feats_z(cid, mode, param):
        if cid not in traj:
            return {k: float("nan") for k in ZF}
        d = traj[cid].slice(0, prefix_i(cid, mode, param) + 1)
        zs = d["z"].to_numpy()
        zs0 = np.where(zs == zs, zs, np.nan)
        if np.all(np.isnan(zs0)):
            return {k: float("nan") for k in ZF}
        z_cr_row = d.filter(pl.col("k") == 0)["z"]
        z_cr = float(z_cr_row[0]) if z_cr_row.len() and z_cr_row[0] is not None else float("nan")
        valid = zs0[~np.isnan(zs0)]
        apex_i = int(np.nanargmax(zs0))
        return {"z_last": float(zs0[-1]), "z_cr": z_cr,
                "z_delta": float(zs0[-1] - z_cr) if z_cr == z_cr else float("nan"),
                "z_max_so_far": float(valid.max()), "k_since_apex": float(len(zs0) - 1 - apex_i),
                "z_slope": float(zs0[-1] - zs0[-2]) if len(valid) > 1 and zs0[-2] == zs0[-2] else 0.0,
                "z_mean": float(np.nanmean(zs0))}

    def feats_gk(cid, mode, param):
        if cid not in traj:
            return {k: float("nan") for k in GF}
        d = traj[cid].slice(0, prefix_i(cid, mode, param) + 1)
        gb = d["gk_ball"].fill_null(_NO_GK).to_numpy()
        go = d["gk_off_line"].to_numpy()
        gy = d["gk_y"].to_numpy()
        go0 = np.where(go == go, go, np.nan)
        gy0 = np.where(gy == gy, gy, np.nan)
        go_cr = go0[0] if len(go0) and go0[0] == go0[0] else np.nan
        gy_cr = gy0[0] if len(gy0) and gy0[0] == gy0[0] else np.nan
        return {"gk_ball_last": float(gb[-1]), "gk_ball_mean": float(gb.mean()),
                "gk_ball_delta": float(gb[-1] - gb[0]),
                "gk_off_last": float(go0[-1]) if go0[-1] == go0[-1] else _NO_GK,
                "gk_off_delta": float(go0[-1] - go_cr) if go0[-1] == go0[-1] and go_cr == go_cr else 0.0,
                "gk_y_last": float(gy0[-1]) if gy0[-1] == gy0[-1] else 0.0,
                "gk_disp": float(abs(gy0[-1] - gy_cr)) if gy0[-1] == gy0[-1] and gy_cr == gy_cr else 0.0}

    Xz, Xg = {}, {}
    for label, (mode, param, _) in zip([c[2] for c in CUTS], CUTS):
        Xz[label] = pl.DataFrame([feats_z(cid, mode, param) for cid in cids]).to_numpy()
        Xg[label] = pl.DataFrame([feats_gk(cid, mode, param) for cid in cids]).to_numpy()
    for D in (Xz, Xg):
        for label in D:
            X = D[label]
            with np.errstate(all="ignore"):
                med = np.nanmedian(X, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            idx = np.where(np.isnan(X))
            X[idx] = np.take(med, idx[1])
            D[label] = X

    def fit(tag, X):
        oof = oof_predict(ESTIMATORS["xgboost"], X, y, groups, method="sigmoid")
        auc = float(roc_auc_score(y, oof))
        print(f"  {tag}: AUC={auc:.4f}", flush=True)
        return {"tag": tag, "auc": auc, "oof": oof}

    jobs = []
    for label in Xz:
        jobs.append((f"z:{label}", Xz[label]))
        jobs.append((f"gk:{label}", Xg[label]))
    b010 = np.load(p["front05_run"] / "D_ball_f010.npz", allow_pickle=True)["X"]
    b100 = np.load(p["front02_run"] / "X_ball_f100.npz", allow_pickle=True)["X"][:, :-1]
    Xu = np.load(p["front02_run"] / "X_union_f100.npz", allow_pickle=True)["X"][:, :-1]
    jobs += [("ball2d_f010", b010), ("ball2d+z_f010", np.hstack([b010, Xz["f010"]])),
             ("ball2d_f100", b100), ("ball2d+z_f100", np.hstack([b100, Xz["f100"]])),
             ("union_f100", Xu), ("union+z+gk_f100", np.hstack([Xu, Xz["f100"], Xg["f100"]]))]
    results = []
    for tag, X in jobs:
        results.append(fit(tag, X))
    oofs = {r["tag"]: r.pop("oof") for r in results}
    pl.DataFrame(results).write_csv(run / "zgk_auc.csv")
    np.savez_compressed(run / "oof_zgk.npz", **oofs)

    rng = np.random.default_rng(0)
    contrasts = []
    def paired(ta, tb, label):
        pa, pb = oofs[ta], oofs[tb]
        diffs = []
        for _ in range(cfg["bootstrap_n"]):
            idx = rng.integers(0, len(y), len(y))
            diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
        contrasts.append({"contrast": label, "delta": float(np.mean(diffs)),
                          "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))})
    paired("ball2d+z_f010", "ball2d_f010", "z_adiciona_f010")
    paired("ball2d+z_f100", "ball2d_f100", "z_adiciona_f100")
    paired("union+z+gk_f100", "union_f100", "z_gk_adicionam_f100")
    pl.DataFrame(contrasts).write_csv(run / "zgk_contrasts.csv")
    common.update_metrics(run, auc={r["tag"]: r["auc"] for r in results}, contrasts=contrasts, stage_reached="Z1")
    common.record_elapsed(run, "Z1", t0)


def stage_z2(run: Path, cfg: dict, df: pl.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    meta = pl.read_parquet(_p(cfg)["front02_run"] / "cross_meta.parquet").sort("cross_id")
    d = df.join(meta.select("cross_id", "success"), on="cross_id", how="left").filter(pl.col("k") > 0).sort(["cross_id", "k"])
    n_post_by = d.group_by("cross_id").agg(pl.len().alias("n_post"))
    d = d.join(n_post_by, on="cross_id")
    d = d.with_columns(((pl.int_range(1, pl.len() + 1).over("cross_id")) / pl.col("n_post")).alias("frac"))
    fr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in [(axes[0], "z", "altura da bola (m)"), (axes[1], "gk_ball", "GK–bola (m)")]:
        agg = d.group_by("frac", "success").agg(pl.col(col).mean().alias("v")).filter(pl.col("frac").is_in(fr))
        piv = agg.pivot(index="frac", on="success", values="v").sort("frac")
        c1 = "true" if "true" in piv.columns else "1.0"
        c0 = "false" if "false" in piv.columns else "0.0"
        ax.plot(piv["frac"], piv[c1], marker="o", color="#2ca02c", label="sucesso")
        ax.plot(piv["frac"], piv[c0], marker="o", color="#d62728", label="fracasso")
        ax.set_xlabel("fraction of the flight"); ax.set_ylabel(ttl); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Eixo z e goleiro ao longo do voo")
    fig.tight_layout(); fig.savefig(run / "figure_zgk.png", dpi=150)
    common.update_metrics(run, stage_reached="Z2", stopped_by=None)
    common.record_elapsed(run, "Z2", t0)
    print("Z2: figura salva", flush=True)


def stage_z3(run: Path, cfg: dict):
    t0 = time.time()
    m = common.load_metrics(run)
    auc, cont = m["auc"], {c["contrast"]: c for c in m["contrasts"]}
    m.update(
        s1_z_soma_f100=bool(cont["z_adiciona_f100"]["lo"] > 0),
        s2_z_gk_somam_uniao=bool(cont["z_gk_adicionam_f100"]["lo"] > 0),
        auc_z_f100=auc.get("z:f100"), auc_gk_f100=auc.get("gk:f100"),
        stage_reached="Z3", stopped_by=None,
    )
    common.write_json(run / "metrics.json", m)
    common.record_elapsed(run, "Z3", t0)
    print(f"Z3: s1={m['s1_z_soma_f100']} s2={m['s2_z_gk_somam_uniao']} | z:f100={auc.get('z:f100'):.4f} gk:f100={auc.get('gk:f100'):.4f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    df = stage_z0(run, cfg)
    stage_z1(run, cfg, df)
    stage_z2(run, cfg, df)
    stage_z3(run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
