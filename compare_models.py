#!/usr/bin/env python3
"""Side-by-side comparison of dimensionality results across models.

Reads results.json files produced by replicate_relation_offsets.py and
raw_analysis.txt files (or JSON equivalents) produced by analyze_raw_directions.py,
then prints a comparison table and saves a CSV.

Usage:
    python compare_models.py \
        --results outputs/tinyllama/results.json \
        --results outputs/gemma3-4b/results.json \
        --out outputs/comparison.csv

    # Automatically discovers all results.json under outputs/
    python compare_models.py --auto outputs/
"""

import argparse
import csv
import json
import os
import sys


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def short_model_name(result: dict, path: str) -> str:
    model = result.get("model", "")
    if model:
        return model.split("/")[-1]
    vindex = result.get("vindex", path)
    return os.path.basename(vindex.rstrip("/")).replace(".vindex", "")


def format_row(result: dict, path: str) -> dict:
    name = short_model_name(result, path)
    dims = result.get("dimensionality", {})
    rc = result.get("replication_check", {})
    return {
        "model": name,
        "n_vectors": result.get("n_vectors", "?"),
        "d_model": result.get("d_model", "?"),
        "50pct_dim": dims.get("50pct", "?"),
        "90pct_dim": dims.get("90pct", "?"),
        "95pct_dim": dims.get("95pct", "?"),
        "spectral_gap_idx": result.get("spectral_gap_idx", "?"),
        "spectral_gap_ratio": round(result.get("spectral_gap_ratio", 0), 2),
        "top_sv_1": round(result["top_10_singular_values"][0], 3) if result.get("top_10_singular_values") else "?",
        "top_sv_2": round(result["top_10_singular_values"][1], 3) if result.get("top_10_singular_values") else "?",
        "passes": "✓" if rc.get("passes") else "✗",
        "source": path,
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results to compare.")
        return

    col_widths = {
        "model": max(len(r["model"]) for r in rows) + 2,
        "50pct_dim": 6,
        "90pct_dim": 6,
        "95pct_dim": 6,
        "spectral_gap_idx": 4,
        "spectral_gap_ratio": 6,
        "top_sv_1": 7,
        "passes": 5,
    }

    w = col_widths
    header = (
        f"{'Model':<{w['model']}} "
        f"{'50%D':>{w['50pct_dim']}} "
        f"{'90%D':>{w['90pct_dim']}} "
        f"{'95%D':>{w['95pct_dim']}} "
        f"{'Gap@':>{w['spectral_gap_idx']}} "
        f"{'Ratio':>{w['spectral_gap_ratio']}} "
        f"{'S[1]':>{w['top_sv_1']}} "
        f"{'Pass':>{w['passes']}}"
    )
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)
    for r in rows:
        gap_component = r['spectral_gap_idx']
        gap_display = f"PC{gap_component + 1}" if isinstance(gap_component, int) else str(gap_component)
        print(
            f"{r['model']:<{w['model']}} "
            f"{str(r['50pct_dim']):>{w['50pct_dim']}} "
            f"{str(r['90pct_dim']):>{w['90pct_dim']}} "
            f"{str(r['95pct_dim']):>{w['95pct_dim']}} "
            f"{gap_display:>{w['spectral_gap_idx']}} "
            f"{str(r['spectral_gap_ratio']):>{w['spectral_gap_ratio']}} "
            f"{str(r['top_sv_1']):>{w['top_sv_1']}} "
            f"{r['passes']:>{w['passes']}}"
        )
    print(sep)
    print("Target: 50%D in 5–22D, spectral gap ratio > 2×")


def save_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")


def discover_results(root: str) -> list[str]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "results.json":
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Compare dimensionality results across models."
    )
    parser.add_argument("--results", action="append", default=[],
                        metavar="PATH",
                        help="Path to a results.json file (repeat for multiple models)")
    parser.add_argument("--auto", metavar="DIR",
                        help="Discover all results.json files under this directory")
    parser.add_argument("--out", default="outputs/comparison.csv",
                        help="Output CSV path (default: outputs/comparison.csv)")
    args = parser.parse_args()

    paths = list(args.results)
    if args.auto:
        paths += discover_results(args.auto)

    if not paths:
        sys.exit("Provide --results paths or --auto <dir>.\n"
                 "Example: python compare_models.py --auto outputs/")

    rows = []
    for path in paths:
        try:
            result = load_result(path)
            rows.append(format_row(result, path))
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No valid results.json files found.")

    print_table(rows)
    save_csv(rows, args.out)


if __name__ == "__main__":
    main()
