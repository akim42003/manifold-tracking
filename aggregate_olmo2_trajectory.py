#!/usr/bin/env python3
"""Aggregate per-cluster SVD results across OLMo-2 checkpoints into a trajectory.

Usage:
    python aggregate_olmo2_trajectory.py
"""

import argparse
import csv
import json
import os
import re

import numpy as np

_CHECKPOINTS = [
    ("stage1-step150-tokens1B",       1),
    ("stage1-step600-tokens3B",       3),
    ("stage1-step3000-tokens13B",     13),
    ("stage1-step12000-tokens51B",    51),
    ("stage1-step51000-tokens214B",   214),
    ("stage1-step217000-tokens911B",  911),
    ("stage1-step928646-tokens3896B", 3896),
]


def load_checkpoint(out_dir: str, name: str, suffix: str = "") -> dict | None:
    path = os.path.join(out_dir, f"olmo2-{name}.vindex", f"per_cluster_svd{suffix}.json")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def summarise(data: dict, name: str, tokens_b: int) -> dict:
    clusters = data.get("clusters", [])
    d50s = np.array([r["d50"] for r in clusters]) if clusters else np.array([])
    passing = [r for r in clusters if 5 <= r["d50"] <= 22]
    n_analyzed = len(clusters)
    n_passing = len(passing)
    pass_rate = n_passing / n_analyzed if n_analyzed > 0 else 0.0

    sample_pairs = []
    for r in passing[:10]:
        pairs = r.get("top_pairs", [])
        if pairs:
            p = pairs[0]
            sample_pairs.append({
                "cluster": r["cluster"],
                "d50": r["d50"],
                "n": r["n"],
                "top_pair": f"{p['in']}→{p['out']}",
            })

    skipped = data.get("skipped_cluster_ids", [])
    n_skipped = len(skipped) if isinstance(skipped, list) else data.get("summary", {}).get("n_skipped", 0)

    return {
        "checkpoint": name,
        "tokens_b": tokens_b,
        "n_analyzed": n_analyzed,
        "n_skipped": n_skipped,
        "n_passing": n_passing,
        "pass_rate": round(pass_rate, 4),
        "d50_median": float(np.median(d50s))        if len(d50s) else None,
        "d50_p5":     float(np.percentile(d50s, 5)) if len(d50s) else None,
        "d50_p25":    float(np.percentile(d50s, 25)) if len(d50s) else None,
        "d50_p75":    float(np.percentile(d50s, 75)) if len(d50s) else None,
        "d50_p95":    float(np.percentile(d50s, 95)) if len(d50s) else None,
        "passing_cluster_ids": [r["cluster"] for r in passing],
        "sample_passing_top_pairs": sample_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--suffix", default="", help="filename suffix e.g. '_spherical'")
    args = parser.parse_args()

    rows = []
    missing = []

    for name, tokens_b in _CHECKPOINTS:
        print(f"Loading {name} ...")
        data = load_checkpoint(args.outputs_dir, name, args.suffix)
        if data is None:
            missing.append(name)
            continue
        rows.append(summarise(data, name, tokens_b))

    if not rows:
        print("No checkpoints found. Run the sweep first.")
        return

    if missing:
        print(f"\nWarning: {len(missing)} missing: {missing}")

    print(f"\n{'checkpoint':<40} {'tokens_B':>9} {'n_ana':>6} {'n_pass':>7} "
          f"{'pass%':>6} {'d50_med':>8} {'d50 IQR':>12}")
    print("─" * 92)
    for r in rows:
        p25 = f"{r['d50_p25']:.0f}" if r['d50_p25'] is not None else "—"
        p75 = f"{r['d50_p75']:.0f}" if r['d50_p75'] is not None else "—"
        med = f"{r['d50_median']:.1f}" if r['d50_median'] is not None else "—"
        print(f"  {r['checkpoint']:<38} {r['tokens_b']:>9} {r['n_analyzed']:>6} "
              f"{r['n_passing']:>7} {r['pass_rate']*100:>5.1f}% {med:>8}  [{p25}–{p75}]")

    result = {
        "model": "allenai/OLMo-2-1124-7B",
        "analysis_params": {
            "k": 128,
            "gate_cos_percentile": 85,
            "n_bootstrap": 100,
            "min_cluster_size": 30,
            "d50_passing_range": [5, 22],
        },
        "checkpoints": rows,
    }

    os.makedirs(args.outputs_dir, exist_ok=True)
    json_path = os.path.join(args.outputs_dir, f"olmo2_trajectory{args.suffix}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    csv_path = os.path.join(args.outputs_dir, f"olmo2_trajectory{args.suffix}.csv")
    fields = ["checkpoint", "tokens_b", "n_analyzed", "n_skipped", "n_passing",
              "pass_rate", "d50_median", "d50_p5", "d50_p25", "d50_p75", "d50_p95"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
