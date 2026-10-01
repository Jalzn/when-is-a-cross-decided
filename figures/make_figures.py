#!/usr/bin/env python
"""Regenerate all publication figures in English from the released result tables.

Run from the repository root:  python figures/make_figures.py
Inputs: results/*/(csv, metrics.json) and data/ball_height_goalkeeper_anon.parquet
Outputs: figures/figure{1..4}_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
MEAN_FLIGHT_S = 1.51

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})


def auc_row(df: pl.DataFrame, tag: str) -> float:
    return float(df.filter(pl.col("tag") == tag)["auc"][0])


def fig1_information_curve() -> None:
    dense = pl.read_csv(ROOT / "results/05-mechanism/curve_dense.csv").filter(pl.col("estimator") == "xgboost")
    pre_lr = pl.read_csv(ROOT / "results/05-mechanism/pre_lr.csv")

    def xy(prefix: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for r in dense.filter(pl.col("tag").str.starts_with(prefix)).iter_rows(named=True):
            label = r["tag"].split(":")[1]
            t = int(label[1:]) * 0.2 if label.startswith("k") else int(label[1:]) / 10 * MEAN_FLIGHT_S
            xs.append(t)
            ys.append(r["auc"])
        order = np.argsort(xs)
        return np.array(xs)[order].tolist(), np.array(ys)[order].tolist()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"fields": ("#4c72b0", "players' fields"), "ball": ("#c44e52", "ball trajectory (2D)"),
              "union": ("#222222", "both combined")}
    for key, (color, label) in styles.items():
        xs, ys = xy(key)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=color, label=label)

    xs = [int(r["tag"].split(":")[1][1:]) * 0.2 for r in pre_lr.iter_rows(named=True)]
    ys = [r["auc"] for r in pre_lr.iter_rows(named=True)]
    ax.plot(xs, ys, ls="--", lw=1.5, color="#55a868", label="pre-touch, logistic regression")

    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvspan(0, MEAN_FLIGHT_S, alpha=0.06, color="#4c72b0")
    ax.text(MEAN_FLIGHT_S * 0.5, 0.545, "ball in flight", ha="center", fontsize=9, color="#4c72b0")
    for x, y, note in [(-1.9, 0.548, "approach\n(flat)"), (0.1, 0.735, "first tenth:\n+0.157"), (1.51, 0.818, "arrival\n0.818")]:
        ax.annotate(note, xy=(x, y), xytext=(x + 0.08, y - 0.055), fontsize=8.5, color="#333333")
    ax.set_xlabel("time relative to the strike (s; flight shown as fraction × mean duration)")
    ax.set_ylabel("AUC (out-of-fold, success)")
    ax.set_title("When is a cross decided? Predictability over the life of 10,871 crosses", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figure1_information_curve.png")
    plt.close(fig)


def fig2_block_attribution() -> None:
    auc = pl.read_csv(ROOT / "results/03-block-attribution/blocks_auc.csv")
    m = json.loads((ROOT / "results/03-block-attribution/metrics.json").read_text())
    loo = m["loo_delta_xgboost"]
    blocks = ["strike", "fields_flight", "ball2d", "flight3d", "arrival"]
    labels = ["strike-time\nconfiguration", "box fields\nin flight", "ball 2D\npath", "3D flight\ntechnique", "arrival\ngeometry"]
    union = float(auc.filter((pl.col("tag") == "union") & (pl.col("target") == "success") & (pl.col("estimator") == "xgboost"))["auc"][0])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    x = np.arange(len(blocks))
    alone = [auc_row(auc, f"alone:{b}") for b in blocks]
    axes[0].bar(x, alone, 0.62, color="#4c72b0", label="block alone")
    axes[0].axhline(union, color="#c44e52", ls="--", lw=1.6, label=f"all blocks combined ({union:.3f})")
    axes[0].set_xticks(x, labels, fontsize=8.5)
    axes[0].set_ylim(0.5, 0.9)
    axes[0].set_ylabel("AUC (out-of-fold, success)")
    axes[0].set_title("What carries the information", fontsize=11)
    axes[0].legend(fontsize=8.5, frameon=False)

    deltas = [loo[b]["mean"] for b in blocks]
    errs = [[loo[b]["mean"] - loo[b]["lo"] for b in blocks], [loo[b]["hi"] - loo[b]["mean"] for b in blocks]]
    colors = ["#c44e52" if loo[b]["lo"] > 0 else "#a6a6a6" for b in blocks]
    axes[1].bar(x, deltas, 0.62, yerr=errs, capsize=3, color=colors)
    axes[1].set_xticks(x, labels, fontsize=8.5)
    axes[1].set_ylabel("ΔAUC when the block is removed (combined − leave-one-out)")
    axes[1].set_title("Unique contribution (red: CI excludes zero)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "figure2_block_attribution.png")
    plt.close(fig)


def fig3_two_phases() -> None:
    sep = pl.read_csv(ROOT / "results/08-dynamic-contestants/sepdyn_by_frac.csv")
    conv = pl.read_csv(ROOT / "results/08-dynamic-contestants/convergence.csv")
    c1 = "true" if "true" in sep.columns else "1.0"
    c0 = "false" if "false" in sep.columns else "0.0"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(sep["frac"], sep[c1], marker="o", ms=4, color="#55a868", label="retained by attack (success)")
    axes[0].plot(sep["frac"], sep[c0], marker="o", ms=4, color="#c44e52", label="lost (failure)")
    axes[0].set_xlabel("fraction of the ball's flight")
    axes[0].set_ylabel("receiver–defender separation (m)")
    axes[0].set_title("The contest resolves late", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)

    axes[1].plot(conv["frac"], conv["p_dyn_is_receiver"], marker="o", ms=4, color="#4c72b0")
    axes[1].set_xlabel("fraction of the ball's flight")
    axes[1].set_ylabel("P(nearest attacker now = final receiver)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("The receiver emerges late", fontsize=11)
    fig.suptitle("Two phases: the ball decides early, the humans resolve late", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "figure3_two_phases_emergence.png")
    plt.close(fig)


def fig4_height_goalkeeper() -> None:
    df = pl.read_parquet(ROOT / "data/ball_height_goalkeeper_anon.parquet")
    labels = pl.read_parquet(ROOT / "data/crosses_features_anon.parquet").select("cross_id", "success")
    df = df.join(labels, on="cross_id", how="left")
    d = df.filter(pl.col("k") > 0).sort(["cross_id", "k"])
    n_post = d.group_by("cross_id").agg(pl.len().alias("n"))
    d = d.join(n_post, on="cross_id").with_columns(
        (pl.int_range(1, pl.len() + 1).over("cross_id") / pl.col("n")).alias("frac"))
    fr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, col, ylab in [(axes[0], "z", "ball height (m)"), (axes[1], "gk_ball", "goalkeeper–ball distance (m)")]:
        agg = (d.group_by("frac", "success").agg(pl.col(col).mean().alias("v")).filter(pl.col("frac").is_in(fr))
               .pivot(index="frac", on="success", values="v").sort("frac"))
        c1 = "true" if "true" in agg.columns else "1.0"
        c0 = "false" if "false" in agg.columns else "0.0"
        ax.plot(agg["frac"], agg[c1], marker="o", ms=4, color="#55a868", label="success")
        ax.plot(agg["frac"], agg[c0], marker="o", ms=4, color="#c44e52", label="failure")
        ax.set_xlabel("fraction of the ball's flight")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle("Height matters late; the keeper is a minor character in flight", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "figure4_height_goalkeeper.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_information_curve()
    fig2_block_attribution()
    fig3_two_phases()
    fig4_height_goalkeeper()
    print("figures written to", FIG)
