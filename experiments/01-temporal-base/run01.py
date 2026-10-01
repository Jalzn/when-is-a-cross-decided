#!/usr/bin/env python
"""Front 01 — temporal base (orchestrator S0..S6, runs on the GPU host).

Reuses the extraction/field modules of the companion xcross-lab (extract_seq,
fields_seq, evaluate) over the FULL consolidated sample (10,871 crosses / 1,041
matches of the canonical feature tables). Documented deviations vs the original
front-05 pipeline: (A) post-strike K clamped at 24 (T<=35, k-key limit; never
triggered in practice); (B) versions computed at runtime as the new baseline (the
H100-1 hashes were lost with that machine); (C) NEW fidelity check — k=0 of each
sequence vs the 30 spatial sums of the canonical features via reduce_features
(threshold 1e-5: float32 storage vs float64 features).

Outputs runs/<id>/{metrics.json, grid.parquet, index.parquet, fidelity_by_cross.csv,
coverage_by_cross.parquet, environment.json}. The npz sequences stay in the canonical
root ~/xcross-lab/data/fields_seq/{cr,windows}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import socket
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

import common  # noqa: E402  (frente 01)
import evaluate  # noqa: E402  (frente 05)
import recompute  # noqa: E402  (frente 01)
import extract_seq  # noqa: E402  (frente 05)
import fields_seq  # noqa: E402  (frente 05)
import importlib.util  # noqa: E402

# front-05 orchestrator by explicit path (avoids `run` name clash with front 01)
_spec = importlib.util.spec_from_file_location("run05", E05 / "run.py")
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)

K_MAX_POST = 24  # desvio A: T = 11 + K <= 35


def _p(cfg: dict) -> dict[str, Path]:
    return {k: Path(os.path.expanduser(v)) for k, v in cfg["paths"][cfg["site"]].items()}


def _stop(run: Path, stage: str, crit: str, msg: str) -> None:
    common.update_metrics(run, stage_reached=stage, stopped_by=crit)
    print(f"[{crit}] {msg}", flush=True)
    sys.exit(3)


def stage_s0(run: Path, cfg: dict) -> None:
    """Environment + full sample from the canonical features + runtime versions."""
    t0 = time.time()
    p = _p(cfg)
    from xcross.data.build import _build_version
    from xcross.features.build import _features_version

    bv, fv = _build_version(), _features_version()
    cfg["expected"]["build_version"] = bv
    cfg["expected"]["features_version"] = fv
    cfg["expected"]["fields_version_01"] = "h2-rebuild-20260930"

    feats_files = sorted(glob.glob(str(p["features_root"] / "*/*/*/features.parquet")))
    sample = (
        pl.scan_parquet(feats_files)
        .select(
            pl.col("cross_id").cast(pl.Utf8),
            pl.col("match_id").cast(pl.Utf8),
            pl.col("league").cast(pl.Utf8),
        )
        .collect()
    )
    assert sample["cross_id"].n_unique() == sample.height, "cross_id duplicado nos features"
    sample.write_parquet(run / "sample_full.parquet")

    proc = {q.name: q for q in p["processed_root"].glob("*/*/*") if (q / "crosses.parquet").exists()}
    raw_missing = [m for m in sample["match_id"].unique().to_list() if common.raw_triple_dir(p["raw_root"], m) is None]
    proc_missing = [m for m in sample["match_id"].unique().to_list() if m not in proc]

    m = dict(
        run_id=cfg["run_id"], host=socket.gethostname(), platform=platform.platform(),
        n_cpus=os.cpu_count(), build_version=bv, features_version=fv,
        fields_version_01=cfg["expected"]["fields_version_01"],
        n_crosses_sample=sample.height, n_matches_sample=sample["match_id"].n_unique(),
        n_features_files=len(feats_files),
        n_matches_raw_missing=len(raw_missing), raw_missing_ids=sorted(raw_missing),
        n_matches_processed_missing=len(proc_missing),
        sample_sha256=common.sha256_file(run / "sample_full.parquet"),
        stage_reached="S0", stopped_by=None,
    )
    common.update_metrics(run, **m)
    common.record_elapsed(run, "S0", t0)
    print(f"S0: {sample.height} cruzamentos / {sample['match_id'].n_unique()} partidas; "
          f"raw_missing={len(raw_missing)} proc_missing={len(proc_missing)}", flush=True)
    if raw_missing or proc_missing:
        _stop(run, "S0", "p1", f"raw_missing={len(raw_missing)} proc_missing={len(proc_missing)}")


def stage_s1(run: Path, cfg: dict) -> None:
    """Grade de frames planejados (adaptado da frente 05 E1; desvio A: clamp K<=24)."""
    t0 = time.time()
    p = _p(cfg)
    sample = pl.read_parquet(run / "sample_full.parquet")
    proc = {q.name: q for q in p["processed_root"].glob("*/*/*") if (q / "crosses.parquet").exists()}
    g = cfg["grid"]
    rows, n_clamped, n_ge_end = [], 0, 0
    for key, df in sample.partition_by("match_id", as_dict=True).items():
        mid = key[0] if isinstance(key, tuple) else key
        pd_ = proc[mid]
        cr = pl.read_parquet(pd_ / "crosses.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
        meta = pl.read_parquet(pd_ / "meta.parquet").row(0, named=True)
        fps = float(meta["fps"])
        hl, hw = (meta["pitch_length_m"] or 105.0) / 2, (meta["pitch_width_m"] or 68.0) / 2
        pre_frames = int(round(g["pre_window_s"] * fps))
        rd = common.raw_triple_dir(p["raw_root"], mid)
        league_dir = str(pd_.relative_to(p["processed_root"]).parent)
        j = df.select("cross_id", "league").join(cr, on="cross_id", how="left")
        assert j["start_frame"].null_count() == 0, f"cruzamentos ausentes no build de {mid}"
        trmax = (
            pl.read_parquet(pd_ / "tracking.parquet").with_columns(pl.col("cross_id").cast(pl.Utf8))
            .group_by("cross_id").agg(pl.col("frame_num").max().alias("tr_max"))
        )
        n_ge_end += j.join(trmax, on="cross_id", how="left").filter(pl.col("tr_max") >= pl.col("end_frame")).height
        for r in j.iter_rows(named=True):
            start, end = int(r["start_frame"]), int(r["end_frame"])
            frames = [start + int(round(g["step_s"] * k * fps)) for k in range(g["k_min"], 0)]
            frames.append(start)
            K = 0
            while start + int(round(g["step_s"] * (K + 1) * fps)) <= end - 1:
                K += 1
                if K > K_MAX_POST:
                    n_clamped += 1
                    K = K_MAX_POST
                    break
                frames.append(start + int(round(g["step_s"] * K * fps)))
            if K == K_MAX_POST and start + int(round(g["step_s"] * (K + 1) * fps)) <= end - 1:
                pass  # already counted in the clamp
            et = r["end_event_type"]
            reason = "out" if et == "OUT" else "possession_event" if et is not None else ("cap" if end - start >= int(5 * fps) else "period_end")
            rows.append({"cross_id": r["cross_id"], "match_id": mid, "league": r["league"], "period": int(r["period"]),
                         "start_frame": start, "end_frame": end, "end_event_type": et, "end_reason": reason,
                         "prev_pe_type": r["prev_pe_type"], "success": r["success"], "crosser_team_id": int(r["crosser_team_id"]),
                         "start_x": float(r["start_x"]), "start_y": float(r["start_y"]), "fps": fps,
                         "half_length": hl, "half_width": hw, "pre_frames": pre_frames,
                         "window_first_frame": start - pre_frames, "K": K, "T": len(frames), "frames": frames,
                         "raw_dir": str(rd) if rd else None, "processed_dir": str(pd_), "league_dir": league_dir})
    grid = pl.DataFrame(rows)
    grid.write_parquet(run / "grid.parquet")
    ids = sorted(grid["cross_id"].to_list())
    zv = np.random.default_rng(cfg["zero_vel"]["seed"]).choice(ids, min(cfg["zero_vel"]["n"], len(ids)), replace=False)
    (run / "zero_vel_ids.txt").write_text("".join(sorted(zv.tolist())) + "\n")
    common.update_metrics(
        run, fps_values=sorted(grid["fps"].unique().to_list()),
        T_planned_mean=float(grid["T"].mean()), T_planned_max=int(grid["T"].max()),
        n_instants_planned_total=int(grid["T"].sum()), n_crosses_k_clamped=n_clamped,
        n_crosses_end_le_start=int((grid["end_frame"] <= grid["start_frame"]).sum()),
        n_crosses_build_tracking_ge_end_frame=n_ge_end, stage_reached="S1",
    )
    common.record_elapsed(run, "S1", t0)
    print(f"S1: T mean {grid['T'].mean():.2f} max {int(grid['T'].max())}; clamped={n_clamped}", flush=True)


def _fid_job(cross_id: str, npz_path: str, feat_row: dict, feats30: list[str]):
    z = np.load(npz_path, allow_pickle=False)
    i_cr = int(z["t_cr"])
    if not bool(z["present"][i_cr]):
        return {"cross_id": cross_id, "cr_absent": True, "max_abs_diff": None}
    m = z["fields"][i_cr].astype(np.float64)
    hl, hw = float(z["half_length"]), float(z["half_width"])
    calc = recompute.reduce_features(m[0], m[1], m[2], m[3], hl, hw, (float(z["start_x"]), float(z["start_y"])))
    diffs = {f: abs(float(calc[f]) - float(feat_row[f])) for f in feats30}
    return {"cross_id": cross_id, "cr_absent": False, "max_abs_diff": max(diffs.values()), **{f"d_{f}": diffs[f] for f in feats30}}


def stage_s4(run: Path, cfg: dict) -> None:
    """Fidelidade k=0: campos da sequência vs 30 somas das features canônicas."""
    from joblib import Parallel, delayed

    t0 = time.time()
    p = _p(cfg)
    index = pl.read_parquet(p["fields_seq_root"] / "index.parquet")
    feats30 = cfg["features_30"]
    feats = (
        pl.read_parquet(glob.glob(str(p["features_root"] / "*/*/*/features.parquet")))
        .with_columns(pl.col("cross_id").cast(pl.Utf8))
        .select(["cross_id"] + feats30)
    )
    frows = {r["cross_id"]: r for r in feats.iter_rows(named=True)}
    jobs = [(r["cross_id"], r["abs_path"], frows[r["cross_id"]], feats30) for r in index.iter_rows(named=True) if r["cross_id"] in frows]
    n_missing_feat = index.height - len(jobs)
    out = Parallel(n_jobs=cfg["n_workers"], return_as="generator")(delayed(_fid_job)(*j) for j in jobs)
    rows = []
    for r in out:
        rows.append(r)
    df = pl.DataFrame(rows)
    df.write_csv(run / "fidelity_by_cross.csv")
    dcols = [f"d_{f}" for f in feats30]
    per_feat = {f: float(df[f"d_{f}"].drop_nans().max()) for f in feats30}
    diffs_all = df["max_abs_diff"].drop_nulls()
    m = dict(
        n_crosses_checked=diffs_all.len(), n_crosses_cr_absent=int(df["cr_absent"].sum()),
        n_crosses_missing_features=n_missing_feat,
        fid_max_abs_diff_cr=float(diffs_all.max()), fid_median_abs_diff_cr=float(diffs_all.median()),
        fid_frac_le_1e6=float((diffs_all <= 1e-6).mean()), fid_frac_le_1e5=float((diffs_all <= 1e-5).mean()),
        fid_frac_le_1e4=float((diffs_all <= 1e-4).mean()),
        fid_max_abs_diff_by_feature=per_feat,
        stage_reached="S4",
    )
    common.update_metrics(run, **m)
    common.record_elapsed(run, "S4", t0)
    print(f"S4: max={m['fid_max_abs_diff_cr']:.3e} median={m['fid_median_abs_diff_cr']:.3e} "
          f"frac<=1e-6={m['fid_frac_le_1e6']:.4f} cr_absent={m['n_crosses_cr_absent']}", flush=True)
    if m["fid_max_abs_diff_cr"] > cfg["criteria"]["s2_fid_max_le"]:
        _stop(run, "S4", "p2", f"fid_max_abs_diff_cr={m['fid_max_abs_diff_cr']}")


def stage_s5(run: Path, cfg: dict) -> None:
    t0 = time.time()
    p = _p(cfg)
    m = evaluate.coverage(run, cfg, pl.read_parquet(run / "grid.parquet"),
                          pl.read_parquet(p["fields_seq_root"] / "index.parquet"),
                          pl.read_parquet(run / "sample_full.parquet"),
                          p["windows_root"], cfg["n_workers"])
    common.update_metrics(run, **m)
    common.record_elapsed(run, "S5", t0)
    if m["frac_full_pre"] < cfg["stop"]["p3_min_frac_full_pre"]:
        _stop(run, "S5", "p3", f"frac_full_pre={m['frac_full_pre']}")
    common.update_metrics(run, stage_reached="S5")
    print(f"S5: full_pre={m['frac_full_pre']:.4f} end_reached={m['frac_end_reached']:.4f} "
          f"invalid={m['n_instants_invalid']}", flush=True)


def stage_s6(run: Path, cfg: dict) -> None:
    t0 = time.time()
    p = _p(cfg)
    from xcross.data.build import _build_version
    from xcross.features.build import _features_version

    def du(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0

    env = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
           "numpy": np.__version__, "polars": pl.__version__,
           "build_version": _build_version(), "features_version": _features_version(),
           "modules": {"run01_sha256": common.sha256_file(HERE / "run01.py")},
           "disk_bytes": {"fields_seq_cr": du(p["fields_seq_root"]), "windows": du(p["windows_root"])}}
    common.write_json(run / "environment.json", env)
    mt = common.load_metrics(run)
    mt.update(
        s1_coverage_100=bool(mt.get("n_npz_written") == mt.get("n_crosses_sample")),
        s2_fid_max_le_criteria=bool(mt.get("fid_max_abs_diff_cr", 1.0) <= cfg["criteria"]["s2_fid_max_le"]),
        s3_frac_full_pre_ge_095=bool(mt.get("frac_full_pre", 0.0) >= cfg["criteria"]["s3_min_frac_full_pre"]),
        stage_reached="S6", stopped_by=None,
    )
    el = mt.get("elapsed_s", {})
    el["total"] = round(sum(v for k, v in el.items() if k[0] in "S"), 1)
    mt["elapsed_s"] = el
    common.write_json(run / "metrics.json", mt)
    common.record_elapsed(run, "S6", t0)
    print(f"S6: criteria s1={mt['s1_coverage_100']} s2={mt['s2_fid_max_le_criteria']} s3={mt['s3_frac_full_pre_ge_095']}", flush=True)


STAGES = {"S0": stage_s0, "S1": stage_s1, "S2": lambda r, c: run05.stage_e2(r, c),
          "S3": lambda r, c: run05.stage_e3(r, c), "S4": stage_s4, "S5": stage_s5, "S6": stage_s6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--stages", default="S0,S1,S2,S3,S4,S5,S6")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    cfg["run_id"] = a.run
    cfg["paths"][cfg["site"]]["run_root"] = str(HERE / "runs" / a.run)
    run = Path(cfg["paths"][cfg["site"]]["run_root"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps(cfg, indent=1))
    for s in a.stages.split(","):
        s = s.strip()
        print(f"=== {s} ===", flush=True)
        STAGES[s](run, cfg)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
