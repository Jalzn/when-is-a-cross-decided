#!/usr/bin/env python
"""Front 02 — the information curve (orchestrator T0..T3).

Question: WHEN does the available information resolve the outcome of a cross,
measured along its lifecycle (approach -> strike -> flight -> arrival) as the
out-of-fold AUC of models restricted to information up to a cutoff.

Three curves per cutoff: A `fields` (players' spatial field sums), B `ball` (2D ball
trajectory), C `union`; plus control arm D (absolute-time cuts with window-length
covariates — the leakage test of xcross-lab front 07). One fixed estimator at every
cutoff (adaboost + xgboost, sigmoid, xcross's oof_predict: StratifiedGroupKFold by
match, per-fold calibration). Targets: success + shot. Paired bootstrap (500).
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

import common  # noqa: E402  (frente 01 do xcross-lab: update_metrics etc.)
import recompute  # noqa: E402  (frente 01: reduce_features)

from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402

FEATS30 = [
    "entropy_attack_around", "entropy_defense_around", "entropy_general_around", "entropy_diff_around", "pitch_control_around",
    "entropy_attack_in_first_post", "entropy_defense_in_first_post", "entropy_general_in_first_post", "entropy_diff_in_first_post", "pitch_control_in_first_post",
    "entropy_attack_in_second_post", "entropy_defense_in_second_post", "entropy_general_in_second_post", "entropy_diff_in_second_post", "pitch_control_in_second_post",
    "entropy_attack_in_center_box", "entropy_defense_in_center_box", "entropy_general_in_center_box", "entropy_diff_in_center_box", "pitch_control_in_center_box",
    "entropy_attack_sum", "entropy_defense_sum", "entropy_general_sum", "entropy_diff_sum", "pitch_control_sum",
    "entropy_attack_grad_towards_goal", "entropy_defense_grad_towards_goal", "entropy_general_grad_towards_goal", "entropy_diff_grad_towards_goal", "pitch_control_grad_towards_goal",
]
PRE_CUTS = [-10, -7, -4, -2, 0]
POST_FRACS = [0.25, 0.50, 0.75, 1.00]
ABS_POST_CUTS = [2, 4, 6]  # arm D: absolute post-strike instants (0.4/0.8/1.2 s)
TARGETS = ["success", "shot_in_window"]
ESTIMATOR_KEYS = ["adaboost", "xgboost"]
CURVES = ["fields", "ball", "union"]


def _p(cfg: dict) -> dict[str, Path]:
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


# ---------------------------------------------------------------- T0: somas por instante

def _cross_record(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    T = int(z["T"]); i_cr = int(z["t_cr"]); k = z["k"]
    hl, hw = float(z["half_length"]), float(z["half_width"])
    sums = np.zeros((T, 30), dtype=np.float64)
    for i in range(T):
        if not bool(z["present"][i]):
            continue
        m = z["fields"][i].astype(np.float64)
        f = recompute.reduce_features(m[0], m[1], m[2], m[3], hl, hw, (float(z["ball_x_t"][i]), float(z["ball_y_t"][i])))
        sums[i] = [f[c] for c in FEATS30]
    return {
        "cross_id": str(z["cross_id"]), "k": k, "T": T, "n_post": int(z["n_post"]),
        "sums": sums, "ball_x": z["ball_x_t"].astype(np.float64), "ball_y": z["ball_y_t"].astype(np.float64),
        "start_x": float(z["start_x"]), "start_y": float(z["start_y"]),
    }


def stage_t0(run: Path, cfg: dict) -> None:
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    feats = (
        pl.read_parquet(glob.glob(str(p["features_root"] / "*/*/*/features.parquet")))
        .with_columns(pl.col("cross_id").cast(pl.Utf8))
        .select("cross_id", "success", "shot_in_window")
    )
    index = index.join(feats, on="cross_id", how="left")
    assert index["success"].null_count() == 0, "rotulos ausentes para cruzamentos do indice"
    recs = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(
        delayed(_cross_record)(r) for r in index["abs_path"].to_list()
    )
    rows = []
    for rec in recs:
        for i in range(rec["T"]):
            rows.append({"cross_id": rec["cross_id"], "k": int(rec["k"][i]),
                         "ball_x": rec["ball_x"][i], "ball_y": rec["ball_y"][i],
                         **{c: rec["sums"][i, j] for j, c in enumerate(FEATS30)}})
    pl.DataFrame(rows).write_parquet(run / "instant_sums.parquet")
    meta = index.select("cross_id", "match_id", "league", "success", "shot_in_window", "end_reason", "T", "n_post")
    meta.write_parquet(run / "cross_meta.parquet")
    common.update_metrics(run, n_crosses=index.height, n_instants=len(rows), stage_reached="T0", stopped_by=None)
    common.record_elapsed(run, "T0", t0)
    print(f"T0: {len(rows)} instantes x 30 somas calculados", flush=True)


# ---------------------------------------------------------------- T1: datasets por corte

def _prefix_stats(rec, ks, last_i):
    """Snapshot at the last instant + prefix statistics (instants 0..last_i)."""
    s = rec["sums"][: last_i + 1]
    snap = s[-1]
    mean = s.mean(axis=0)
    rng = s.max(axis=0) - s.min(axis=0)
    return snap, mean, rng


def _ball_feats(rec, last_i, i_cr):
    bx, by = rec["ball_x"][: last_i + 1], rec["ball_y"][: last_i + 1]
    x0, y0 = rec["ball_x"][i_cr], rec["ball_y"][i_cr]
    dx, dy = bx[-1] - x0, by[-1] - y0
    path = float(np.hypot(np.diff(bx), np.diff(by)).sum())
    eucl = float(np.hypot(dx, dy)) + 1e-9
    n_steps = max(int(last_i - i_cr), 0)
    return {
        "ball_x_last": float(bx[-1]), "ball_y_last": float(by[-1]),
        "ball_dx": float(dx), "ball_dy": float(dy),
        "ball_path_len": path, "ball_path_ratio": path / eucl,
        "ball_speed_proxy": path / (0.2 * max(n_steps, 1)),
        "ball_disp_from_start": float(np.hypot(bx[-1] - bx[0], by[-1] - by[0])),
    }


def stage_t1(run: Path, cfg: dict) -> None:
    t0 = time.time()
    sums = pl.read_parquet(run / "instant_sums.parquet")
    meta = pl.read_parquet(run / "cross_meta.parquet")

    # rebuild per-cross records (sorted by k)
    recs = {}
    for (cid,), df in sums.partition_by("cross_id", as_dict=True).items():
        df = df.sort("k")
        recs[cid] = {
            "sums": df.select(FEATS30).to_numpy(), "k": df["k"].to_numpy(),
            "ball_x": df["ball_x"].to_numpy(), "ball_y": df["ball_y"].to_numpy(),
        }
    starts = {}  # (unused; the CR ball reference comes from the npz)
    for cid, rec in recs.items():
        k = rec["k"]
        i_cr = int(np.where(k == 0)[0][0])
        rec["i_cr"] = i_cr

    def prefix_end(cid: str, mode: str, param) -> int:
        rec = recs[cid]; k = rec["k"]; i_cr = rec["i_cr"]
        if mode == "pre":
            hit = np.where(k == param)[0]
            return int(hit[0]) if len(hit) else i_cr
        if mode == "frac":
            n_post = len(k) - 1 - i_cr
            if n_post == 0:
                return i_cr
            j = max(1, int(np.ceil(param * n_post)))
            return min(i_cr + j, len(k) - 1)
        if mode == "abs":
            target_k = param
            post = k[k > 0]
            if len(post) == 0 or target_k > post.max():
                return len(k) - 1  # janela acabou antes do corte
            return int(np.where(k == target_k)[0][0])
        raise ValueError(mode)

    datasets = {}  # (curve, cut_label) -> feature matrix builder outputs
    cut_specs = []
    for kc in PRE_CUTS:
        cut_specs.append(("pre", kc, f"k{kc}"))
    for f in POST_FRACS:
        cut_specs.append(("frac", f, f"f{int(f*100):03d}"))
    for ka in [0] + ABS_POST_CUTS:
        cut_specs.append(("abs", ka, f"a{ka}"))

    rows_out = {c: {} for c in CURVES + ["fields+meta"]}
    for cid in recs:
        rec = recs[cid]
        for mode, param, label in cut_specs:
            last_i = prefix_end(cid, mode, param)
            snap, mean, rng = _prefix_stats(rec, rec["k"], last_i)
            bf = _ball_feats(rec, last_i, rec["i_cr"])
            is_post = last_i > rec["i_cr"]
            base_fields = {"f_snap": snap, "f_mean": mean, "f_rng": rng}
            rows_out["fields"].setdefault(label, {})[cid] = (base_fields, None, is_post)
            rows_out["ball"].setdefault(label, {})[cid] = (None, bf, is_post)
            rows_out["union"].setdefault(label, {})[cid] = (base_fields, bf, is_post)
            if mode == "abs":
                n_post_total = int(meta.filter(pl.col("cross_id") == cid)["n_post"][0])
                meta_feats = {"n_post": float(n_post_total), "ended_before": float(last_i < len(rec["k"]) - 1)}
                rows_out["fields+meta"].setdefault(label, {})[cid] = (base_fields, meta_feats, is_post)

    # materializa matrizes
    cids = sorted(recs.keys())
    matrices = {}
    for curve in rows_out:
        for label in rows_out[curve]:
            cols = []
            d = rows_out[curve][label]
            first = next(iter(d.values()))
            has_f, has_b = first[0] is not None, first[1] is not None
            X = np.zeros((len(cids), (90 if has_f else 0) + (len(first[1]) if has_b else 0) + 1), dtype=np.float64)
            for i, cid in enumerate(cids):
                basef, bf, is_post = d[cid]
                row = []
                if has_f:
                    row = list(basef["f_snap"]) + list(basef["f_mean"]) + list(basef["f_rng"])
                if has_b:
                    row += [bf[v] for v in sorted(bf)]
                row.append(1.0 if is_post else 0.0)
                X[i] = row
            matrices[(curve, label)] = X
    # salva compacto por corte (npz separado por curva)
    for (curve, label), X in matrices.items():
        np.savez_compressed(run / f"X_{curve}_{label}.npz", X=X, cids=np.array(cids, dtype=object))
    common.update_metrics(run, n_cut_datasets=len(matrices), n_features_by_curve={
        "fields": int(matrices[("fields", "k0")].shape[1]),
        "ball": int(matrices[("ball", "k0")].shape[1]),
        "union": int(matrices[("union", "k0")].shape[1]),
    }, stage_reached="T1")
    common.record_elapsed(run, "T1", t0)
    print(f"T1: {len(matrices)} cutoff datasets built", flush=True)


# ---------------------------------------------------------------- T2: OOF

def _one_job(key, X, y, groups, cfg):
    curve, label, target, est = key
    oof = oof_predict(ESTIMATORS[est], X, y, groups, method="sigmoid")
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, oof))
    return {"curve": curve, "cut": label, "target": target, "estimator": est, "auc": auc, "oof": oof}


def stage_t2(run: Path, cfg: dict) -> None:
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    meta = pl.read_parquet(run / "cross_meta.parquet")
    lab_by_cid_s = dict(zip(meta["cross_id"].to_list(), meta["success"].to_list()))
    lab_by_cid_h = dict(zip(meta["cross_id"].to_list(), meta["shot_in_window"].to_list()))
    grp_by_cid = dict(zip(meta["cross_id"].to_list(), meta["match_id"].cast(pl.Utf8).to_list()))

    jobs = []
    for fname in sorted(glob.glob(str(run / "X_*.npz"))):
        parts = Path(fname).stem.split("_", 1)[1].rsplit("_", 1)
        curve, label = parts[0], parts[1]
        z = np.load(fname, allow_pickle=True)
        X = z["X"]
        cids = [str(c) for c in z["cids"]]
        for target in TARGETS:
            y = np.array([(lab_by_cid_s if target == "success" else lab_by_cid_h)[c] for c in cids], dtype=int)
            groups = np.array([grp_by_cid[c] for c in cids], dtype=object)
            for est in ESTIMATOR_KEYS:
                jobs.append(((curve, label, target, est), X, y, groups))
    print(f"T2: {len(jobs)} ajustes OOF (5 folds calibrados cada)", flush=True)
    results = Parallel(n_jobs=cfg["n_fit_jobs"], return_as="generator")(
        delayed(_one_job)(k, X, y, g, cfg) for k, X, y, g in jobs
    )
    rows, oof_store = [], {}
    for r in results:
        oof_store["|".join(r[k] for k in ("curve", "cut", "target", "estimator"))] = r["oof"]
        rows.append({k: v for k, v in r.items() if k != "oof"})
        print(f"  {r['curve']}/{r['cut']}/{r['target']}/{r['estimator']}: AUC={r['auc']:.4f}", flush=True)
    pl.DataFrame(rows).write_csv(run / "curve_raw.csv")
    np.savez_compressed(run / "oof_all.npz", **oof_store)
    common.update_metrics(run, n_fits=len(rows), stage_reached="T2")
    common.record_elapsed(run, "T2", t0)


# ---------------------------------------------------------------- T3: metrics + criteria + figure

def _auc_ci(y, p, rng, n_boot):
    from sklearn.metrics import roc_auc_score
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def stage_t3(run: Path, cfg: dict) -> None:
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    meta = pl.read_parquet(run / "cross_meta.parquet").sort("cross_id")
    cids_sorted = meta["cross_id"].to_list()
    y_s = meta["success"].to_numpy()
    y_shot = meta["shot_in_window"].to_numpy()
    oof = np.load(run / "oof_all.npz")
    raw = pl.read_csv(run / "curve_raw.csv")
    n_boot = cfg["bootstrap_n"]
    rng = np.random.default_rng(0)

    cut_order = [f"k{k}" for k in PRE_CUTS] + [f"f{int(f*100):03d}" for f in POST_FRACS]
    curve_rows = []
    for r in raw.iter_rows(named=True):
        if r["curve"] == "fields+meta":
            continue
        key = "|".join([r["curve"], r["cut"], r["target"], r["estimator"]])
        p = oof[key]
        y = y_s if r["target"] == "success" else y_shot
        lo, hi = _auc_ci(y, p, rng, n_boot)
        curve_rows.append({**r, "auc_lo": lo, "auc_hi": hi})
    pl.DataFrame(curve_rows).write_csv(run / "curve.csv")

    # paired contrasts (same bootstrap indices)
    contrasts = []
    for target in TARGETS:
        for est in ESTIMATOR_KEYS:
            for curve in CURVES:
                y = y_s if target == "success" else y_shot
                pk0 = oof[f"{curve}|k0|{target}|{est}"]
                pf100 = oof[f"{curve}|f100|{target}|{est}"]
                diffs = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(y), len(y))
                    diffs.append(roc_auc_score(y[idx], pf100[idx]) - roc_auc_score(y[idx], pk0[idx]))
                lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
                contrasts.append({"curve": curve, "target": target, "estimator": est,
                                  "contrast": "k0_vs_f100", "delta": float(np.mean(diffs)), "lo": lo, "hi": hi})
    # arm D: comprimento explica?
    for target in TARGETS:
        y = y_s if target == "success" else y_shot
        pa = oof[f"fields+meta|a6|{target}|xgboost"]
        lo, hi = _auc_ci(y, pa, rng, n_boot)
        contrasts.append({"curve": "fields+meta", "target": target, "estimator": "xgboost",
                          "contrast": "abs_1.2s_with_length", "delta": float(roc_auc_score(y, pa)), "lo": lo, "hi": hi})
    pl.DataFrame(contrasts).write_csv(run / "contrasts.csv")

    # criteria (union curve, xgboost, success — primary)
    prim = {(r["cut"]): r for r in curve_rows if r["curve"] == "union" and r["target"] == "success" and r["estimator"] == "xgboost"}
    aucs_seq = [prim[c]["auc"] for c in cut_order if c in prim]
    mono = all(aucs_seq[i + 1] >= aucs_seq[i] - cfg["s1_mono_tol"] for i in range(len(aucs_seq) - 1))
    c_k0 = next((c for c in contrasts if c["curve"] == "union" and c["target"] == "success" and c["estimator"] == "xgboost" and c["contrast"] == "k0_vs_f100"), None)
    jump = c_k0["delta"] if c_k0 else None
    jump_ci_excl_0 = bool(c_k0 and c_k0["lo"] > 0) if c_k0 else False
    s2 = bool(jump is not None and jump >= cfg["s2_min_jump"] and jump_ci_excl_0)
    mid_cuts = ["f025", "f050", "f075"]
    s3 = bool(jump and any(prim[c]["auc"] - prim["k0"]["auc"] >= 0.5 * jump for c in mid_cuts if c in prim))

    # figura-âncora
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x_pre = np.arange(-2.0, 0.01, 0.2)
    for ax, target in zip(axes, TARGETS):
        for curve, color in zip(CURVES, ["#1f77b4", "#d62728", "#222222"]):
            rows = [r for r in curve_rows if r["curve"] == curve and r["target"] == target and r["estimator"] == "xgboost" and r["cut"] in cut_order]
            rows = sorted(rows, key=lambda r: cut_order.index(r["cut"]))
            xs = [PRE_CUTS[i] * 0.2 if r["cut"].startswith("k") else None for i, r in enumerate(rows)]
            # x-axis: time relative to strike (pre: k*0.2; post: fraction * mean length)
            xs = []
            for r in rows:
                if r["cut"].startswith("k"):
                    xs.append(int(r["cut"][1:]) * 0.2)
                else:
                    xs.append(int(r["cut"][1:]) / 100 * cfg["mean_flight_s"])
            ax.errorbar(xs, [r["auc"] for r in rows], yerr=[[r["auc"] - r["auc_lo"] for r in rows], [r["auc_hi"] - r["auc"] for r in rows]],
                        label=curve, color=color, marker="o", capsize=3, lw=1.8)
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("time relative to the strike (s; flight as fraction x mean)")
        ax.set_title(target)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUC (OOF)")
    axes[0].legend()
    fig.suptitle("Information curve of the cross — when is the outcome decided")
    fig.tight_layout()
    fig.savefig(run / "figure_curve.png", dpi=150)

    common.update_metrics(
        run,
        curve_primary={c: prim[c]["auc"] for c in cut_order if c in prim},
        s1_monotonic=bool(mono), s2_jump_ge_010_with_ci=s2, jump_k0_f100=jump,
        jump_ci=[c_k0["lo"], c_k0["hi"]] if c_k0 else None,
        s3_intermediate_half_jump=s3,
        auc_anchor_k0_fields=next((r["auc"] for r in curve_rows if r["curve"] == "fields" and r["cut"] == "k0" and r["target"] == "success" and r["estimator"] == "xgboost"), None),
        auc_f100_fields=next((r["auc"] for r in curve_rows if r["curve"] == "fields" and r["cut"] == "f100" and r["target"] == "success" and r["estimator"] == "xgboost"), None),
        auc_f100_ball=next((r["auc"] for r in curve_rows if r["curve"] == "ball" and r["cut"] == "f100" and r["target"] == "success" and r["estimator"] == "xgboost"), None),
        stage_reached="T3", stopped_by=None,
    )
    common.record_elapsed(run, "T3", t0)
    print(f"T3: s1={mono} s2={s2} (jump={jump}) s3={s3}", flush=True)


STAGES = {"T0": stage_t0, "T1": stage_t1, "T2": stage_t2, "T3": stage_t3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--stages", default="T0,T1,T2,T3")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    cfg["run_id"] = a.run
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    for s in a.stages.split(","):
        print(f"=== {s} ===", flush=True)
        STAGES[s.strip()](run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
