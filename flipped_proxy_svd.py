#!/usr/bin/env python3
"""Flipped-proxy SVD — test output-side vs input-side feature organization.

Extends the per-cluster SVD pipeline with three filter regimes:

  gate   (default) — gate cosine filter only, identical to per_cluster_svd.py
  margin           — down-logit margin filter (output-anchored proxy)
  both             — intersection of gate + margin filters

Use this to test whether FFN features are organized more cleanly on the output
side than the input side (e.g. Mistral's noisy-content-but-passing clusters).

Usage:
    # Reproduce existing pipeline (regression test)
    python flipped_proxy_svd.py --vindex gemma3-4b.vindex --k 128 --filter-regime gate

    # Test the flipped (output-anchored) proxy on Mistral
    python flipped_proxy_svd.py --vindex mistral-7b.vindex --k 128 --filter-regime margin

    # Strict intersection
    python flipped_proxy_svd.py --vindex mistral-7b.vindex --k 128 --filter-regime both
"""

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
from sklearn.preprocessing import normalize

ASCII_WORD = re.compile(r"^▁[A-Za-z]{3,}$")

_VERY_LARGE_MARGIN = 1e9   # stand-in when top-2 logit is absent / zero


# ── data loading ──────────────────────────────────────────────────────────────

def read_index(p):
    return json.load(open(os.path.join(p, "index.json")))


def read_embeddings(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden, vocab = idx["hidden_size"], idx.get("vocab_size")
    raw = np.fromfile(os.path.join(p, "embeddings.bin"), dtype=dtype)
    return raw.reshape(vocab if vocab else raw.size // hidden, hidden).astype(np.float32)


def read_down_meta_full(p):
    """Read down_meta.bin, returning per-feature (layer, feat, top_token_id, top1_logit, top2_logit).

    Binary layout per feature entry (entry_size = 8 + top_k * 8):
        offset  0: top_token_id  (u32)
        offset  4: c_score       (f32)  — header score, not used here
        offset  8: token_id_1    (u32)  — first top-k pair
        offset 12: logit_1       (f32)
        offset 16: token_id_2    (u32)  — second top-k pair
        offset 20: logit_2       (f32)
        ...
    """
    MAGIC = 0x444D4554
    rows = []
    with open(os.path.join(p, "down_meta.bin"), "rb") as f:
        magic, _, num_layers, top_k = struct.unpack("<IIII", f.read(16))
        if magic != MAGIC:
            raise ValueError(f"Bad magic in down_meta.bin: {magic:#x}")
        entry_size = 4 + 4 + top_k * 8
        has_top2 = top_k >= 2
        for layer in range(num_layers):
            (nf,) = struct.unpack("<I", f.read(4))
            raw = f.read(nf * entry_size)
            for i in range(nf):
                base = i * entry_size
                (top_tid,) = struct.unpack_from("<I", raw, base)
                if top_k >= 1:
                    (logit1,) = struct.unpack_from("<f", raw, base + 12)
                else:
                    logit1 = 0.0
                if has_top2:
                    (logit2,) = struct.unpack_from("<f", raw, base + 20)
                    # Guard: missing / zero top-2 → treat as decisive
                    if logit2 == 0.0 and logit1 != 0.0:
                        logit2 = logit1 - _VERY_LARGE_MARGIN
                else:
                    logit2 = logit1 - _VERY_LARGE_MARGIN
                rows.append((layer, i, top_tid, logit1, logit2))
    return rows, top_k


def build_ww_vocab(tok_path, embed):
    tok_json = json.load(open(tok_path))
    special_ids = {t["id"] for t in tok_json.get("added_tokens", [])}
    ids = np.array([tid for t, tid in tok_json["model"]["vocab"].items()
                    if ASCII_WORD.match(t) and tid < embed.shape[0] and tid not in special_ids],
                   dtype=np.int32)
    return ids, embed[ids], special_ids


def read_gate_vectors(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden = idx["hidden_size"]
    raw = np.fromfile(os.path.join(p, "gate_vectors.bin"), dtype=dtype)
    return raw.reshape(raw.size // hidden, hidden).astype(np.float32)


# ── direction computation with regime-aware filtering ─────────────────────────

def compute_directions(vindex_path, regime, gate_cos_pct, down_margin_pct,
                       same_concept_threshold):
    idx = read_index(vindex_path)
    bands = idx.get("layer_bands", {})
    kl_min = bands["knowledge"][0] if "knowledge" in bands else min(14, idx["num_layers"])
    kl_max = bands["knowledge"][1] + 1 if "knowledge" in bands else min(28, idx["num_layers"])
    print(f"  Knowledge layers: {kl_min}–{kl_max - 1}")

    embed = read_embeddings(vindex_path, idx)
    ww_ids, ww_embed, special_ids = build_ww_vocab(
        os.path.join(vindex_path, "tokenizer.json"), embed)
    ww_norm = ww_embed / (np.linalg.norm(ww_embed, axis=1, keepdims=True) + 1e-8)
    print(f"  Whole-word vocab: {len(ww_ids):,}  special excluded: {len(special_ids):,}")

    rows, top_k = read_down_meta_full(vindex_path)
    print(f"  down_meta: {len(rows):,} total features  top_k={top_k}")

    layers_info = idx.get("layers", [])
    layer_of = np.empty(len(rows), dtype=np.int32)
    fg = 0
    for li in layers_info:
        n = li["num_features"]
        layer_of[fg:fg + n] = li["layer"]
        fg += n

    kl_idx = np.where((layer_of >= kl_min) & (layer_of < kl_max))[0]
    kl_out_ids  = np.array([rows[i][2] for i in kl_idx], dtype=np.int32)
    kl_logit1   = np.array([rows[i][3] for i in kl_idx], dtype=np.float32)
    kl_logit2   = np.array([rows[i][4] for i in kl_idx], dtype=np.float32)
    kl_margins  = kl_logit1 - kl_logit2
    print(f"  Knowledge-layer features: {len(kl_idx):,}")

    # ── down-logit margin distribution (always printed for diagnostics) ──
    print(f"  Down-logit margin distribution (all {len(kl_idx):,} KL features):")
    for pv in [50, 75, 85, 90, 95, 99]:
        print(f"    p{pv:2d}: {np.percentile(kl_margins, pv):.4f}")

    # ── gate NN search ──
    all_gates = read_gate_vectors(vindex_path, idx)
    kl_gates = all_gates[kl_idx]

    print(f"  Gate NN search ({len(kl_idx):,} × {len(ww_ids):,}) ...")
    batch, n_kl = 2048, len(kl_idx)
    in_ids   = np.zeros(n_kl, dtype=np.int32)
    gate_cos = np.zeros(n_kl, dtype=np.float32)
    n_batches = (n_kl + batch - 1) // batch
    t0 = time.perf_counter()
    for bi, s in enumerate(range(0, n_kl, batch)):
        e = min(s + batch, n_kl)
        g = kl_gates[s:e]
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        scores = g_norm @ ww_norm.T
        best = scores.argmax(axis=1)
        in_ids[s:e]   = ww_ids[best]
        gate_cos[s:e] = scores[np.arange(len(best)), best]
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.perf_counter() - t0
            print(f"    {bi+1}/{n_batches}  {elapsed:.0f}s  "
                  f"eta {elapsed/(bi+1)*(n_batches-bi-1):.0f}s",
                  end="\r", flush=True)
    print()

    print(f"  Gate cosine distribution (all {n_kl:,} features):")
    for pv in [50, 75, 85, 90, 95, 99]:
        print(f"    p{pv:2d}: {np.percentile(gate_cos, pv):.4f}")

    # ── resolve per-regime thresholds ──
    if regime == "gate":
        gate_thresh   = float(np.percentile(gate_cos, gate_cos_pct))
        margin_thresh = -np.inf
        print(f"\n  Regime: gate  (p{gate_cos_pct} gate cos → {gate_thresh:.4f})")
    elif regime == "margin":
        gate_thresh   = float(np.percentile(gate_cos, 50))   # permissive
        margin_thresh = float(np.percentile(kl_margins, down_margin_pct))
        print(f"\n  Regime: margin  (gate relaxed to p50 → {gate_thresh:.4f}; "
              f"p{down_margin_pct} margin → {margin_thresh:.4f})")
    else:  # both
        gate_thresh   = float(np.percentile(gate_cos, gate_cos_pct))
        margin_thresh = float(np.percentile(kl_margins, down_margin_pct))
        print(f"\n  Regime: both  (p{gate_cos_pct} gate → {gate_thresh:.4f}; "
              f"p{down_margin_pct} margin → {margin_thresh:.4f})")

    # ── filter yield breakdown ──
    gate_mask   = gate_cos >= gate_thresh
    margin_mask = kl_margins >= margin_thresh
    both_mask   = gate_mask & margin_mask
    print(f"  Filter yield:")
    print(f"    gate filter   : {gate_mask.sum():>7,} / {n_kl:,} "
          f"({gate_mask.mean()*100:.1f}%)")
    print(f"    margin filter : {margin_mask.sum():>7,} / {n_kl:,} "
          f"({margin_mask.mean()*100:.1f}%)")
    print(f"    intersection  : {both_mask.sum():>7,} / {n_kl:,} "
          f"({both_mask.mean()*100:.1f}%)")

    if regime == "gate":
        quality_mask = gate_mask
    elif regime == "margin":
        quality_mask = margin_mask & gate_mask   # gate at p50, margin at target pct
    else:
        quality_mask = both_mask

    print(f"    → using       : {quality_mask.sum():>7,} features for this regime")

    # ── same validity guards as per_cluster_svd ──
    valid = (quality_mask &
             (kl_out_ids > 0) &
             (~np.isin(kl_out_ids, list(special_ids))) &
             (kl_out_ids < embed.shape[0]) &
             (in_ids != kl_out_ids))

    out_vecs = embed[kl_out_ids[valid]]
    in_vecs  = embed[in_ids[valid]]
    dot = np.einsum("ij,ij->i",
                    out_vecs / (np.linalg.norm(out_vecs, axis=1, keepdims=True) + 1e-8),
                    in_vecs  / (np.linalg.norm(in_vecs,  axis=1, keepdims=True) + 1e-8))
    diff_mask = dot <= same_concept_threshold
    out_vecs, in_vecs = out_vecs[diff_mask], in_vecs[diff_mask]
    valid_idx = np.where(valid)[0][diff_mask]

    dirs = normalize((out_vecs - in_vecs).astype(np.float32), norm="l2")
    keep = np.linalg.norm(dirs, axis=1) > 0.1
    print(f"  Directions after all filters: {keep.sum():,}  "
          f"(quality filter: {valid.sum():,}  "
          f"cosine≤{same_concept_threshold}: {diff_mask.sum():,})")

    return dirs[keep], kl_out_ids[valid_idx][keep], in_ids[valid_idx][keep], embed


# ── per-cluster SVD (identical to per_cluster_svd.py) ────────────────────────

def build_vocab_inv(vindex_path):
    tok_path = os.path.join(vindex_path, "tokenizer.json")
    try:
        vocab = json.load(open(tok_path))["model"]["vocab"]
        return {tid: tok.replace("▁", " ").strip() for tok, tid in vocab.items()}
    except Exception:
        return {}


def top_pairs(out_ids_c, in_ids_c, vocab_inv, n=8):
    from collections import Counter
    pairs = list(zip(in_ids_c.tolist(), out_ids_c.tolist()))
    common = Counter(pairs).most_common(n)
    def dec(tid): return vocab_inv.get(tid, f"#{tid}")
    return [{"in": dec(inp), "out": dec(out), "count": cnt} for (inp, out), cnt in common]


def per_cluster_svd(dirs, assignments, out_ids, in_ids, vocab_inv, min_size=30):
    k = int(assignments.max()) + 1
    results = []
    skipped = 0
    for c in range(k):
        mask = assignments == c
        n = mask.sum()
        if n < min_size:
            skipped += 1
            continue
        X = dirs[mask].astype(np.float64)
        X -= X.mean(axis=0)
        S = np.linalg.svd(X, compute_uv=False)
        var = S ** 2
        total = var.sum()
        if total < 1e-12:
            skipped += 1
            continue
        cum = np.cumsum(var) / total
        d50 = int(np.searchsorted(cum, 0.50)) + 1
        d90 = int(np.searchsorted(cum, 0.90)) + 1
        gap_idx = int(np.argmax(S[:-1] / (S[1:] + 1e-12)))
        gap_ratio = float(S[gap_idx] / (S[gap_idx + 1] + 1e-12)) if len(S) > 1 else 1.0
        pairs = top_pairs(out_ids[mask], in_ids[mask], vocab_inv)
        results.append({"cluster": c, "n": int(n), "d50": d50, "d90": d90,
                        "spectral_gap_idx": gap_idx, "spectral_gap_ratio": gap_ratio,
                        "top_pairs": pairs})
    return results, skipped


def report(results, skipped, k, regime, gate_pass_rate=None):
    if not results:
        print(f"  No clusters passed the min-size filter (skipped {skipped}/{k}).")
        return None

    d50s = np.array([r["d50"] for r in results])
    d90s = np.array([r["d90"] for r in results])
    gaps = np.array([r["spectral_gap_ratio"] for r in results])

    in_range = ((d50s >= 5) & (d50s <= 22)).sum()
    pct = in_range / len(d50s) * 100

    print(f"\n── Per-cluster 50%-variance dimensionality ({len(results)} clusters, {skipped} skipped) ──")
    print(f"  {'Stat':<12} {'d50':>6} {'d90':>6} {'gap_ratio':>10}")
    print(f"  {'─'*38}")
    for label, fn in [("median", np.median), ("mean", np.mean),
                      ("p5",  lambda x: np.percentile(x, 5)),
                      ("p25", lambda x: np.percentile(x, 25)),
                      ("p75", lambda x: np.percentile(x, 75)),
                      ("p95", lambda x: np.percentile(x, 95))]:
        print(f"  {label:<12} {fn(d50s):>6.1f} {fn(d90s):>6.1f} {fn(gaps):>10.2f}")

    print(f"\n  Range: d50 = {d50s.min()}–{d50s.max()}D  (target 5-22D)")
    print(f"  Clusters with d50 in 5–22D: {in_range}/{len(results)} ({pct:.0f}%)")

    verdict = "✓ REPLICATION PASSES" if pct >= 50 else "✗ REPLICATION FAILS"
    print(f"\n  {verdict}")
    print(f"  Criterion: ≥50% of clusters have d50 in 5–22D (got {pct:.0f}%)")

    passing = sorted([r for r in results if 5 <= r["d50"] <= 22], key=lambda r: r["d50"])
    print(f"\n── Passing clusters ({len(passing)}) — sorted by d50 ──")
    print(f"  {'c':>4} {'n':>6} {'d50':>5} {'d90':>5}  top pairs")
    for r in passing:
        pairs_str = "  ".join(f"{p['in']}→{p['out']}({p['count']})" for p in r["top_pairs"][:5])
        print(f"  {r['cluster']:>4} {r['n']:>6} {r['d50']:>5} {r['d90']:>5}  {pairs_str}")

    failing = sorted([r for r in results if r["d50"] > 22], key=lambda r: -r["n"])
    print(f"\n── Failing clusters ({len(failing)}) — top-10 by size ──")
    print(f"  {'c':>4} {'n':>6} {'d50':>5} {'d90':>5}  top pairs")
    for r in failing[:10]:
        pairs_str = "  ".join(f"{p['in']}→{p['out']}({p['count']})" for p in r["top_pairs"][:5])
        print(f"  {r['cluster']:>4} {r['n']:>6} {r['d50']:>5} {r['d90']:>5}  {pairs_str}")

    # ── interpretation hint ──
    print(f"\n── Interpretation hint ──")
    if regime == "gate":
        print("  Input-anchored view — clusters represent features that fire on similar inputs.")
    elif regime == "margin":
        print("  Output-anchored view — clusters represent features that push toward similar outputs.")
    else:
        print("  Strict intersection — clusters represent features clean on both input and output sides.")

    if gate_pass_rate is not None and pct > gate_pass_rate + 10:
        print("  Output side appears more organized than input side — feature asymmetry suggests")
        print("  the model encodes relations as clean output attractors with diffuse input gates.")

    return {"n_clusters_analyzed": len(results), "skipped": skipped,
            "d50_median": float(np.median(d50s)), "d50_range": [int(d50s.min()), int(d50s.max())],
            "pct_in_target": float(pct), "passes": bool(pct >= 50),
            "n_passing": int(in_range), "n_failing": int(len(results) - in_range)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Flipped-proxy SVD: compare gate vs margin vs both filter regimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python flipped_proxy_svd.py --vindex gemma3-4b.vindex --k 128 --filter-regime gate
  python flipped_proxy_svd.py --vindex mistral-7b.vindex --k 128 --filter-regime margin
  python flipped_proxy_svd.py --vindex mistral-7b.vindex --k 128 --filter-regime both
""")
    parser.add_argument("--vindex", default="gemma3-4b.vindex")
    parser.add_argument("--k", type=int, default=64,
                        help="Number of k-means clusters (default 64)")
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--filter-regime", choices=["gate", "margin", "both"], default="gate",
                        dest="filter_regime",
                        help="gate: input-anchored (default); margin: output-anchored; "
                             "both: intersection")
    parser.add_argument("--gate-cos-percentile", type=float, default=85,
                        dest="gate_cos_percentile",
                        help="Gate cosine percentile threshold (default 85). "
                             "In margin regime this is relaxed to p50 automatically.")
    parser.add_argument("--down-margin-percentile", type=float, default=85,
                        dest="down_margin_percentile",
                        help="Down-logit margin percentile threshold (default 85). "
                             "Used in margin and both regimes.")
    parser.add_argument("--same-concept-threshold", type=float, default=0.9,
                        dest="same_concept_threshold")
    parser.add_argument("--gate-pass-rate", type=float, default=None,
                        dest="gate_pass_rate",
                        help="Pass rate from a prior gate-regime run (0-100). If provided "
                             "and the current regime exceeds it by >10pp, prints an asymmetry note.")
    args = parser.parse_args()

    if not os.path.isdir(args.vindex):
        sys.exit(f"Not a directory: {args.vindex}")

    print(f"Computing offset directions from {args.vindex} "
          f"[regime={args.filter_regime}] ...")
    dirs, out_ids, in_ids, embed = compute_directions(
        args.vindex,
        regime=args.filter_regime,
        gate_cos_pct=args.gate_cos_percentile,
        down_margin_pct=args.down_margin_percentile,
        same_concept_threshold=args.same_concept_threshold,
    )

    if len(dirs) < args.k * args.min_cluster_size:
        print(f"\nWarning: {len(dirs):,} directions / k={args.k} = "
              f"{len(dirs) // args.k} per cluster avg — "
              f"below min_cluster_size={args.min_cluster_size}. "
              f"Consider lowering --k or relaxing filter percentiles.")

    from sklearn.cluster import MiniBatchKMeans
    print(f"\nFitting k={args.k} clusters ...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=42, batch_size=4096,
                         max_iter=300, n_init=5)
    assignments = km.fit_predict(dirs)
    print(f"  Inertia: {km.inertia_:.1f}  "
          f"smallest cluster: {np.bincount(assignments).min()}")

    vocab_inv = build_vocab_inv(args.vindex)

    print(f"\nRunning per-cluster SVD (min_size={args.min_cluster_size}) ...")
    results, skipped = per_cluster_svd(dirs, assignments, out_ids, in_ids, vocab_inv,
                                       args.min_cluster_size)
    summary = report(results, skipped, args.k,
                     regime=args.filter_regime,
                     gate_pass_rate=args.gate_pass_rate)

    vindex_name = os.path.basename(args.vindex.rstrip("/"))
    out_dir = os.path.join("outputs", vindex_name)
    os.makedirs(out_dir, exist_ok=True)

    out_fname = f"flipped_proxy_{args.filter_regime}_k{args.k}.json"
    out_path  = os.path.join(out_dir, out_fname)
    with open(out_path, "w") as f:
        json.dump({
            "filter_regime": args.filter_regime,
            "filter_params": {
                "gate_cos_percentile": args.gate_cos_percentile,
                "down_margin_percentile": args.down_margin_percentile,
                "same_concept_threshold": args.same_concept_threshold,
            },
            "summary": summary,
            "clusters": results,
            "params": vars(args),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
