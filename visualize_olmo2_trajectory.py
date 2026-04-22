#!/usr/bin/env python3
"""Visualize the OLMo-2 training trajectory from aggregate_olmo2_trajectory.py output.

Produces 4 figures saved to outputs/:
  - olmo2_pass_rate.{png,svg}      — pass rate (%) vs tokens
  - olmo2_d50_boxplot.{png,svg}    — d50 distribution boxplots across checkpoints
  - olmo2_d50_median.{png,svg}     — median d50 with IQR band vs tokens
  - olmo2_n_analyzed.{png,svg}     — n_analyzed and n_passing vs tokens
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _save(fig, base: str) -> None:
    for ext in ("png", "svg"):
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def figure_pass_rate(rows: list, out_dir: str) -> None:
    tokens = [r["tokens_b"] for r in rows]
    rates = [r["pass_rate"] * 100 for r in rows]
    labels = [r["checkpoint"].replace("stage1-", "") for r in rows]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tokens, rates, "o-", color="#2563eb", lw=2, ms=6)
        for t, r, lbl in zip(tokens, rates, labels):
            ax.annotate(lbl, (t, r), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="#374151")
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (B)", fontsize=11)
        ax.set_ylabel("Pass rate (%)", fontsize=11)
        ax.set_title("OLMo-2 7B — semantic cluster pass rate vs training tokens", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0)
        fig.tight_layout()
    _save(fig, os.path.join(out_dir, "olmo2_pass_rate"))


def figure_d50_median(rows: list, out_dir: str) -> None:
    tokens = np.array([r["tokens_b"] for r in rows], dtype=float)
    medians = np.array([r["d50_median"] if r["d50_median"] is not None else np.nan for r in rows])
    p25 = np.array([r["d50_p25"] if r["d50_p25"] is not None else np.nan for r in rows])
    p75 = np.array([r["d50_p75"] if r["d50_p75"] is not None else np.nan for r in rows])
    p5 = np.array([r["d50_p5"] if r["d50_p5"] is not None else np.nan for r in rows])
    p95 = np.array([r["d50_p95"] if r["d50_p95"] is not None else np.nan for r in rows])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(tokens, p5, p95, alpha=0.12, color="#2563eb", label="p5–p95")
        ax.fill_between(tokens, p25, p75, alpha=0.25, color="#2563eb", label="IQR (p25–p75)")
        ax.plot(tokens, medians, "o-", color="#2563eb", lw=2, ms=6, label="Median d50")
        ax.axhspan(5, 22, color="#16a34a", alpha=0.06, label="Passing range [5–22]")
        ax.axhline(5, color="#16a34a", lw=0.8, ls="--")
        ax.axhline(22, color="#16a34a", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (B)", fontsize=11)
        ax.set_ylabel("d50 (dims for 50% variance)", fontsize=11)
        ax.set_title("OLMo-2 7B — median d50 trajectory with IQR band", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
    _save(fig, os.path.join(out_dir, "olmo2_d50_median"))


def figure_d50_boxplot(rows: list, out_dir: str) -> None:
    """Approximate boxplot using p5/p25/median/p75/p95 summary stats."""
    labels = [r["checkpoint"].replace("stage1-step", "").split("-")[0] + "s" for r in rows]
    positions = list(range(len(rows)))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, r in enumerate(rows):
            med = r["d50_median"]
            if med is None:
                continue
            p25 = r["d50_p25"] or med
            p75 = r["d50_p75"] or med
            p5  = r["d50_p5"]  or med
            p95 = r["d50_p95"] or med
            # whiskers
            ax.vlines(i, p5, p95, color="#6b7280", lw=1.5)
            ax.hlines([p5, p95], i - 0.15, i + 0.15, color="#6b7280", lw=1.5)
            # IQR box
            rect = plt.Rectangle((i - 0.3, p25), 0.6, p75 - p25,
                                  facecolor="#bfdbfe", edgecolor="#2563eb", lw=1.5)
            ax.add_patch(rect)
            # median line
            ax.hlines(med, i - 0.3, i + 0.3, color="#1d4ed8", lw=2)

        ax.axhspan(5, 22, color="#16a34a", alpha=0.06, label="Passing range")
        ax.axhline(5,  color="#16a34a", lw=0.8, ls="--")
        ax.axhline(22, color="#16a34a", lw=0.8, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("d50", fontsize=11)
        ax.set_title("OLMo-2 7B — d50 distribution per checkpoint (p5/IQR/p95)", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
    _save(fig, os.path.join(out_dir, "olmo2_d50_boxplot"))


def figure_n_analyzed(rows: list, out_dir: str) -> None:
    tokens = [r["tokens_b"] for r in rows]
    n_ana = [r["n_analyzed"] for r in rows]
    n_pass = [r["n_passing"] for r in rows]
    n_skip = [r.get("n_skipped", 0) for r in rows]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tokens, n_ana,  "s--", color="#6b7280", lw=1.5, ms=5, label="Analyzed")
        ax.plot(tokens, n_pass, "o-",  color="#2563eb", lw=2,   ms=6, label="Passing")
        ax.plot(tokens, n_skip, "^:",  color="#dc2626", lw=1.5, ms=5, label="Skipped (<30 vectors)")
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (B)", fontsize=11)
        ax.set_ylabel("Cluster count", fontsize=11)
        ax.set_title("OLMo-2 7B — clusters analyzed vs training tokens", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
    _save(fig, os.path.join(out_dir, "olmo2_n_analyzed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="outputs/olmo2_trajectory.json")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.trajectory):
        print(f"Trajectory file not found: {args.trajectory}")
        print("Run aggregate_olmo2_trajectory.py first.")
        return

    with open(args.trajectory) as f:
        data = json.load(f)

    rows = data["checkpoints"]
    if not rows:
        print("No checkpoints in trajectory file.")
        return

    print(f"Plotting {len(rows)} checkpoints ...")
    os.makedirs(args.outputs_dir, exist_ok=True)

    figure_pass_rate(rows, args.outputs_dir)
    figure_d50_median(rows, args.outputs_dir)
    figure_d50_boxplot(rows, args.outputs_dir)
    figure_n_analyzed(rows, args.outputs_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
