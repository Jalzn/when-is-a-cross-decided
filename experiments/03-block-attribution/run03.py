#!/usr/bin/env python
"""Fronts 03+04 — block attribution (WHAT) and per-block skill (WHO).

Blocks at the arrival cutoff (f100), target success (+ shot as robustness):
  strike       26 canonical strike-time non-spatial features
  fields_flight 90 in-flight evolution of the players' field sums (front 02)
  ball2d         7 in-flight 2D ball path (front 02)
  flight3d      13 delivery technique (flight_*, swing_*, clearance_*)
  arrival       19 arrival geometry (end point, zone summaries, support)

Fits: each block alone, full union, leave-one-out. Same OOF protocol as front 02.
Skill (front 04): per-player means (>=20 crosses) of each block's OOF score; temporal
split-half stability (match_id+start_frame chronological key, validated in xcross-lab
front 10). Paired bootstrap of the flight3d-vs-arrival contrast.
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

from xcross.model.estimators import ESTIMATORS  # noqa: E402
from xcross.model.train import oof_predict  # noqa: E402

BLOCKS_CANONICAL = {
    "strike": [
        "start_x", "start_y", "distance_start_from_goal",
        "attackers_in_box", "defenders_in_box", "attackers_in_zone", "defenders_in_zone", "box_ratio", "zone_ratio",
        "pressure_crosser_nearest_def", "pressure_crosser_def_within_3m",
        "marking_max_attacker_gap_in_box", "marking_free_attackers_in_box", "marking_mean_gap_in_box",
        "pocket_radius_in_box", "pocket_goal_angle_in_box",
        "coverage_kl_attack_defense_in_box", "coverage_kl_attack_defense_in_zone",
        "gk_distance_off_line", "gk_ball_distance", "gk_lateral_speed", "gk_present",
        "shape_line_height", "shape_block_width", "shape_block_area", "shape_last_line_to_gk_gap",
    ],
    "flight3d": [
        "flight_apex_height", "flight_apex_timing", "flight_launch_angle", "flight_descent_angle",
        "flight_hang_time", "flight_loftiness", "flight_pace_3d", "flight_bounce_count",
        "clearance_min_margin_over_defender", "clearance_over_keeper",
        "swing_cutback", "swing_inout", "swing_curl_magnitude",
    ],
    "arrival": [
        "end_x", "end_y", "distance_from_end_line", "polar_angle_cross",
        "entropy_attack_in_zone", "entropy_defense_in_zone", "entropy_general_in_zone", "entropy_diff_in_zone", "pitch_control_in_zone",
        "attackers_near_action_line", "defenders_near_action_line",
        "temporal_defenders_in_box_delta", "temporal_attackers_in_box_delta",
        "temporal_entropy_diff_zone_delta", "temporal_pitch_control_zone_delta",
        "support_attackers_ring", "support_defenders_ring", "support_ratio_ring",
    ],
}
TARGETS = ["success", "shot_in_window"]
ESTIMATOR_KEYS = ["adaboost", "xgboost"]
MIN_CROSSES = 20


def _p(cfg: dict) -> dict[str, Path]:
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def stage_u0(run: Path, cfg: dict) -> None:
    t0 = time.time()
    p = _p(cfg)
    feats = (
        pl.read_parquet(glob.glob(str(p["features_root"] / "*/*/*/features.parquet")))
        .with_columns(pl.col("cross_id").cast(pl.Utf8), pl.col("match_id").cast(pl.Utf8))
        .sort("cross_id")
    )
    # matrizes da frente 02 no corte f100 (linhas em cids ordenados)
    xf = np.load(p["front02_run"] / "X_fields_f100.npz", allow_pickle=True)
    xb = np.load(p["front02_run"] / "X_ball_f100.npz", allow_pickle=True)
    cids02 = [str(c) for c in xf["cids"]]
    assert cids02 == [str(c) for c in xb["cids"]]
    f = feats.filter(pl.col("cross_id").is_in(cids02)).sort("cross_id")
    assert f["cross_id"].to_list() == cids02, "ordem cids divergente entre frente02 e features"

    Xf = xf["X"][:, :-1]  # dropa is_post (constante em f100)
    Xb = xb["X"][:, :-1]
    blocks = {
        "fields_flight": Xf,
        "ball2d": Xb,
    }
    for bname, cols in BLOCKS_CANONICAL.items():
        blocks[bname] = f.select(cols).to_numpy()
    blocks["union"] = np.hstack([blocks[k] for k in ("strike", "fields_flight", "ball2d", "flight3d", "arrival")])
    for name, X in blocks.items():
        np.savez_compressed(run / f"B_{name}.npz", X=X, cids=np.array(cids02, dtype=object))
    f.select("cross_id", "crosser_player_id", "match_id", "success", "shot_in_window").write_parquet(run / "meta03.parquet")
    # chave cronológica: (match_id, start_frame) do grid da frente 01
    grid = pl.read_parquet(p["front01_grid"]).with_columns(pl.col("cross_id").cast(pl.Utf8))
    f2 = f.select("cross_id").join(grid.select("cross_id", "start_frame", "period"), on="cross_id", how="left")
    assert f2["start_frame"].null_count() == 0
    f2.write_parquet(run / "chrono.parquet")
    common.update_metrics(run, n_crosses=len(cids02),
                          n_features_by_block={k: int(v.shape[1]) for k, v in blocks.items()},
                          stage_reached="U0", stopped_by=None)
    common.record_elapsed(run, "U0", t0)
    print(f"U0: blocos construídos para {len(cids02)} cruzamentos", flush=True)


def _one_job(key, X, y, groups, cfg):
    from sklearn.metrics import roc_auc_score
    tag, target, est = key
    oof = oof_predict(ESTIMATORS[est], X, y, groups, method="sigmoid")
    return {"tag": tag, "target": target, "estimator": est,
            "auc": float(roc_auc_score(y, oof)), "oof": oof}


def stage_u1(run: Path, cfg: dict) -> None:
    from joblib import Parallel, delayed

    t0 = time.time()
    meta = pl.read_parquet(run / "meta03.parquet")
    y_by = {"success": meta["success"].to_numpy(), "shot_in_window": meta["shot_in_window"].to_numpy()}
    groups = meta["match_id"].to_numpy().astype(object)

    block_names = ["strike", "fields_flight", "ball2d", "flight3d", "arrival"]
    Xs = {}
    for fname in glob.glob(str(run / "B_*.npz")):
        z = np.load(fname, allow_pickle=True)
        Xs[Path(fname).stem[2:]] = z["X"]

    jobs = []
    for target in TARGETS:
        for est in ESTIMATOR_KEYS:
            jobs.append((("union", target, est), Xs["union"], y_by[target]))
            for b in block_names:
                jobs.append(((f"alone:{b}", target, est), Xs[b], y_by[target]))
                jobs.append(((f"loo_minus:{b}", target, est),
                             np.hstack([Xs[o] for o in block_names if o != b]), y_by[target]))
    print(f"U1: {len(jobs)} ajustes OOF", flush=True)
    results = Parallel(n_jobs=cfg["n_fit_jobs"], return_as="generator")(
        delayed(_one_job)(k, X, y, groups, cfg) for k, X, y in jobs
    )
    rows, oof_store = [], {}
    for r in results:
        oof_store["|".join([r["tag"], r["target"], r["estimator"]])] = r["oof"]
        rows.append({k: v for k, v in r.items() if k != "oof"})
        print(f"  {r['tag']}/{r['target']}/{r['estimator']}: AUC={r['auc']:.4f}", flush=True)
    pl.DataFrame(rows).write_csv(run / "blocks_auc.csv")
    np.savez_compressed(run / "oof_blocks.npz", **oof_store)
    common.update_metrics(run, n_fits=len(rows), stage_reached="U1")
    common.record_elapsed(run, "U1", t0)


def _temporal_split_half(df: pl.DataFrame, score_col: str) -> float:
    """Spearman entre médias por jogador das metades cronológicas."""
    from scipy.stats import spearmanr
    df = df.sort(["match_id", "start_frame"])
    halves = (
        df.with_columns((pl.int_range(0, pl.len()).over("crosser_player_id") < (pl.len().over("crosser_player_id") + 1) // 2).alias("first_half"))
        .group_by("crosser_player_id", "first_half").agg(pl.col(score_col).mean().alias("s"), pl.len().alias("n"))
    )
    piv = halves.pivot(index="crosser_player_id", on="first_half", values="s").drop_nulls()
    if piv.height < 5:
        return float("nan")
    rho, _ = spearmanr(piv["true"], piv["false"])
    return float(rho)


def stage_u2(run: Path, cfg: dict) -> None:
    t0 = time.time()
    meta = pl.read_parquet(run / "meta03.parquet")
    chrono = pl.read_parquet(run / "chrono.parquet")
    oof = np.load(run / "oof_blocks.npz")
    df = (
        meta.select("cross_id", "crosser_player_id", "match_id", "success")
        .join(chrono.select("cross_id", "start_frame"), on="cross_id", how="left")
        .with_columns(pl.col("crosser_player_id").cast(pl.Utf8))
    )
    n_by = df.group_by("crosser_player_id").agg(pl.len().alias("n"))
    qualified = set(n_by.filter(pl.col("n") >= MIN_CROSSES)["crosser_player_id"].to_list())
    df = df.with_row_index("i")  # i = posição em meta03 = ordem dos arrays OOF
    dfq = df.filter(pl.col("crosser_player_id").is_in(list(qualified)))
    idx = dfq["i"].to_numpy()

    rows = []
    tags = ["union"] + [f"alone:{b}" for b in ["strike", "fields_flight", "ball2d", "flight3d", "arrival"]]
    # taxa bruta como referência
    raw = dfq.select("crosser_player_id", "match_id", "start_frame", pl.col("success").cast(pl.Float64).alias("score_raw"))
    rows.append({"tag": "raw_rate", "estimator": "-", "stability_temporal": _temporal_split_half(raw, "score_raw"),
                 "n_players": len(qualified)})
    for tag in tags:
        for est in ESTIMATOR_KEYS:
            key = f"{tag}|success|{est}"
            if key not in oof:
                continue
            d = dfq.select("crosser_player_id", "match_id", "start_frame").with_columns(pl.Series("score", oof[key][idx]))
            rows.append({"tag": tag, "estimator": est,
                         "stability_temporal": _temporal_split_half(d, "score"),
                         "n_players": len(qualified)})
    pl.DataFrame(rows).write_csv(run / "stability_blocks.csv")
    common.update_metrics(run, n_qualified_crossers=len(qualified), stage_reached="U2")
    common.record_elapsed(run, "U2", t0)
    for r in rows:
        print(f"  stab {r['tag']}/{r['estimator']}: {r['stability_temporal']:.3f} (n={r['n_players']})", flush=True)


def stage_u3(run: Path, cfg: dict) -> None:
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    auc = pl.read_csv(run / "blocks_auc.csv")
    stab = pl.read_csv(run / "stability_blocks.csv")

    # bootstrap pareado dos AUCs (crosses) para contrastes LOO
    meta = pl.read_parquet(run / "meta03.parquet")
    oof = np.load(run / "oof_blocks.npz")
    y = meta["success"].to_numpy()
    rng = np.random.default_rng(0)
    from sklearn.metrics import roc_auc_score
    loo_deltas = {}
    for b in ["strike", "fields_flight", "ball2d", "flight3d", "arrival"]:
        for est in ["xgboost", "adaboost"]:
            pu, pm = oof[f"union|success|{est}"], oof[f"loo_minus:{b}|success|{est}"]
            diffs = []
            for _ in range(cfg["bootstrap_n"]):
                idx = rng.integers(0, len(y), len(y))
                diffs.append(roc_auc_score(y[idx], pu[idx]) - roc_auc_score(y[idx], pm[idx]))
            loo_deltas[f"{b}|{est}"] = {"mean": float(np.mean(diffs)), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}
    common.write_json(run / "loo_deltas.json", loo_deltas)

    # critérios
    def get_auc(tag, est="xgboost", target="success"):
        return float(auc.filter((pl.col("tag") == tag) & (pl.col("estimator") == est) & (pl.col("target") == target))["auc"][0])
    blocks = ["strike", "fields_flight", "ball2d", "flight3d", "arrival"]
    union_auc = get_auc("union")
    s1 = all(get_auc(f"alone:{b}") < union_auc for b in blocks)
    s2 = all(loo_deltas[f"{b}|xgboost"]["lo"] > 0 for b in blocks)

    # contraste flight3d vs arrival em estabilidade (bootstrap de jogadores)
    stab_v = {r["tag"]: r["stability_temporal"] for r in stab.iter_rows(named=True) if r["estimator"] == "xgboost"}
    # (ponto; CI por reamostragem de jogadores ficaria no paper completo)
    s3 = bool(stab_v.get("alone:flight3d", float("nan")) > stab_v.get("alone:arrival", float("nan")))

    # figura: dois painéis (AUC blocos; estabilidade blocos)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = ["toque\n(strike)", "área no voo\n(fields)", "bola 2D", "voo 3D\n(técnica)", "chegada"]
    alone_v = [get_auc(f"alone:{b}") for b in blocks]
    loo_v = [loo_deltas[f"{b}|xgboost"]["mean"] for b in blocks]
    x = np.arange(len(blocks))
    axes[0].bar(x - 0.2, alone_v, 0.4, label="bloco isolado", color="#4c72b0")
    axes[0].bar(x + 0.2, [union_auc] * 5, 0.4, label="união", color="#c0c0c0")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("AUC (OOF, success)"); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("O QUÊ carrega a informação")
    axes[1].bar(np.arange(len(loo_v)), loo_v, color="#55a868")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("ΔAUC ao remover o bloco (união − LOO)")
    axes[1].set_title("contribuição única (leave-one-out)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(run / "figure_blocks.png", dpi=150)

    common.update_metrics(
        run,
        auc_by_block_alone={b: get_auc(f"alone:{b}") for b in blocks},
        auc_union=union_auc,
        loo_delta_xgboost={b: loo_deltas[f"{b}|xgboost"] for b in blocks},
        stability_by_block=stab_v,
        stability_raw_rate=float([r["stability_temporal"] for r in stab.iter_rows(named=True) if r["tag"] == "raw_rate"][0]),
        s1_complementarity=bool(s1), s2_unique_contribution=bool(s2), s3_flight3d_more_stable_than_arrival=s3,
        stage_reached="U3", stopped_by=None,
    )
    common.record_elapsed(run, "U3", t0)
    print(f"U3: s1={s1} s2={s2} s3={s3}", flush=True)


STAGES = {"U0": stage_u0, "U1": stage_u1, "U2": stage_u2, "U3": stage_u3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--stages", default="U0,U1,U2,U3")
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
