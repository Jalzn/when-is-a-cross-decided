#!/usr/bin/env python
"""Front 09 — intention vs execution in the early ball signal.

Decomposition of the first-tenth signal (ball alone reaches 0.729):
  A "destination"  realized landing point + derived geometry
  B "execution"    trajectory features at f010/f020
  C = A + B        does execution add over destination?
  D "pure exec."   residuals of B on A (OLS, label-free)

Status: descriptive decomposition (destination is an a-posteriori quantity),
consistent with the right-hand side of the curve.
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

CUTS = ["f010", "f020"]


def _p(cfg):
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def stage_t0(run: Path, cfg: dict):
    t0 = time.time()
    p = _p(cfg)
    feats = (pl.read_parquet(sorted(__import__("glob").glob(str(p["features_root"] / "*/*/*/features.parquet"))))
             .with_columns(pl.col("cross_id").cast(pl.Utf8)).sort("cross_id"))
    # A: destino + geometria
    A = feats.select(
        pl.col("end_x").alias("end_x"),
        pl.col("end_y").alias("end_y"),
        (((52.5 - pl.col("end_x")) ** 2 + pl.col("end_y") ** 2) ** 0.5).alias("end_dist_goal"),
        (pl.col("end_x").abs() <= 52.5).cast(pl.Float64).alias("end_in_field"),
        (pl.col("end_x") >= 52.5 - 16.5).cast(pl.Float64).alias("end_in_final_third"),
        pl.arctan2(pl.col("end_y"), (52.5 - pl.col("end_x"))).alias("end_angle_goal"),
    ).to_numpy()
    mats = {"A_destino": A}
    for cut in CUTS:
        B = np.load(p["front05_run"] / f"D_ball_{cut}.npz", allow_pickle=True)["X"]
        mats[f"B_execucao_{cut}"] = B
        mats[f"C_destino+execucao_{cut}"] = np.hstack([A, B])
        # D: residuals of B ~ A (OLS, label-free — no outcome leak)
        Xd = np.hstack([A, np.ones((len(A), 1))])
        beta, *_ = np.linalg.lstsq(Xd, B, rcond=None)
        mats[f"D_execucao_pura_{cut}"] = B - Xd @ beta
    for name, X in mats.items():
        np.savez_compressed(run / f"IE_{name}.npz", X=X)
    common.update_metrics(run, n_features={k: int(v.shape[1]) for k, v in mats.items()},
                          n_crosses=len(A), stage_reached="T0", stopped_by=None)
    common.record_elapsed(run, "T0", t0)
    print(f"T0: {len(mats)} matrizes", flush=True)
    return mats


def stage_t1(run: Path, cfg: dict, mats: dict):
    from joblib import Parallel, delayed
    t0 = time.time()
    p = _p(cfg)
    meta = pl.read_parquet(p["front02_run"] / "cross_meta.parquet").sort("cross_id")
    y = meta["success"].to_numpy()
    groups = meta["match_id"].cast(pl.Utf8).to_numpy().astype(object)

    def fit(tag, X):
        oof = oof_predict(ESTIMATORS["xgboost"], X, y, groups, method="sigmoid")
        auc = float(roc_auc_score(y, oof))
        print(f"  {tag}: AUC={auc:.4f}", flush=True)
        return {"tag": tag, "auc": auc, "oof": oof}

    jobs = [(k, v) for k, v in mats.items()]
    results = Parallel(n_jobs=4, return_as="generator")(delayed(fit)(k, X) for k, X in jobs)
    oofs, rows = {}, []
    for r in results:
        oofs[r["tag"]] = r.pop("oof")
        rows.append(r)
    pl.DataFrame(rows).write_csv(run / "ie_auc.csv")
    np.savez_compressed(run / "oof_ie.npz", **oofs)

    # contrastes pareados por bootstrap de cruzamentos
    rng = np.random.default_rng(0)
    contrasts = []
    def paired(tag_a, tag_b, label):
        pa, pb = oofs[tag_a], oofs[tag_b]
        diffs = []
        for _ in range(cfg["bootstrap_n"]):
            idx = rng.integers(0, len(y), len(y))
            diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
        contrasts.append({"contrast": label, "delta": float(np.mean(diffs)),
                          "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))})
    for cut in CUTS:
        paired(f"C_destino+execucao_{cut}", "A_destino", f"C_menos_A|{cut}")
        paired("A_destino", f"B_execucao_{cut}", f"A_menos_B|{cut}")
        paired(f"D_execucao_pura_{cut}", "A_destino", f"D_menos_A|{cut}")
    pl.DataFrame(contrasts).write_csv(run / "ie_contrasts.csv")
    common.update_metrics(run, auc={r["tag"]: r["auc"] for r in rows}, contrasts=contrasts, stage_reached="T1")
    common.record_elapsed(run, "T1", t0)


def stage_t2(run: Path, cfg: dict):
    t0 = time.time()
    m = common.load_metrics(run)
    auc = m["auc"]
    cont = {c["contrast"]: c for c in m["contrasts"]}
    a_dest = auc["A_destino"]
    b010 = auc["B_execucao_f010"]
    c_ca = cont["C_menos_A|f010"]
    s1_destino_forte = bool(a_dest >= 0.65)
    s2_alvo_domina = bool(a_dest >= b010 - 0.02)
    s3_execucao_soma = bool(c_ca["lo"] > 0.005)
    m.update(s1_destino_forte=s1_destino_forte, s2_alvo_domina=s2_alvo_domina,
             s3_execucao_soma_significativo=s3_execucao_soma, stage_reached="T2", stopped_by=None)
    common.write_json(run / "metrics.json", m)
    common.record_elapsed(run, "T2", t0)
    print(f"T2: s1={s1_destino_forte} s2={s2_alvo_domina} s3={s3_execucao_soma}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    mats = stage_t0(run, cfg)
    stage_t1(run, cfg, mats)
    stage_t2(run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
