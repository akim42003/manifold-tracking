#!/usr/bin/env python3
"""Generate relation_clusters.json for a vindex that was extracted without clustering.

Replicates LARQL's knowledge pipeline:
  - Restricts to "knowledge layers" (from layer_bands, or L14–L28 heuristic)
  - Gate token = argmax cosine of gate row vs. ASCII-Latin whole-word embeddings only
  - Confidence threshold: drop features where gate cosine < 0.3
  - Offset = normalize(E[output] - E[gate_token])
  - Semantic filter: drop pairs where cosine(E[out], E[in]) > 0.9 (same concept)
  - Special token exclusion: output tokens that are <pad>/<eos>/<unused>/etc. are dropped
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

ASCII_WORD = re.compile(r"^\u2581[A-Za-z]{3,}$")


# ── helpers ───────────────────────────────────────────────────────────────────

def read_index(vindex_path: str) -> dict:
    return json.load(open(os.path.join(vindex_path, "index.json")))


def read_embeddings(vindex_path: str, index: dict) -> np.ndarray:
    dtype_str = index.get("dtype", "f32")
    hidden = index["hidden_size"]
    vocab = index.get("vocab_size")
    path = os.path.join(vindex_path, "embeddings.bin")
    raw = np.fromfile(path, dtype=np.float16 if dtype_str == "f16" else np.float32)
    if vocab:
        return raw.reshape(vocab, hidden).astype(np.float32)
    return raw.reshape(-1, hidden).astype(np.float32)


def read_down_meta(vindex_path: str) -> list[tuple[int, int, int]]:
    """Read down_meta.bin → list of (layer_idx, feat_in_layer, top_token_id)."""
    MAGIC = 0x444D4554
    path = os.path.join(vindex_path, "down_meta.bin")
    triples = []
    with open(path, "rb") as f:
        magic, _version, num_layers, top_k = struct.unpack("<IIII", f.read(16))
        if magic != MAGIC:
            raise ValueError(f"Bad down_meta.bin magic: 0x{magic:08X}")
        entry_size = 4 + 4 + top_k * 8
        for layer in range(num_layers):
            (num_features,) = struct.unpack("<I", f.read(4))
            raw = f.read(num_features * entry_size)
            for i in range(num_features):
                off = i * entry_size
                (top_token_id,) = struct.unpack_from("<I", raw, off)
                triples.append((layer, i, top_token_id))
    return triples


def build_whole_word_vocab(tokenizer_path: str, embed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ASCII-Latin whole-word tokens only: ▁ + at least 3 Latin letters, no special tokens."""
    tok_json = json.load(open(tokenizer_path))
    vocab = tok_json["model"]["vocab"]
    # Collect all special/added token IDs to exclude from output side
    special_ids = {t["id"] for t in tok_json.get("added_tokens", [])}

    ww_ids = np.array(
        [tid for token, tid in vocab.items()
         if ASCII_WORD.match(token)
         and tid < embed.shape[0]
         and tid not in special_ids],
        dtype=np.int32,
    )
    ww_embed = embed[ww_ids]
    return ww_ids, ww_embed, special_ids


def read_gate_vectors(vindex_path: str, index: dict) -> np.ndarray:
    dtype_str = index.get("dtype", "f32")
    hidden = index["hidden_size"]
    path = os.path.join(vindex_path, "gate_vectors.bin")
    raw = np.fromfile(path, dtype=np.float16 if dtype_str == "f16" else np.float32)
    n_features = raw.size // hidden
    return raw.reshape(n_features, hidden).astype(np.float32)


def decode_tokens(tok_json: dict, ids: list[int]) -> list[str]:
    inv = {v: k.replace("\u2581", " ").strip() for k, v in tok_json["model"]["vocab"].items()}
    return [inv.get(i, f"#{i}") for i in ids]


# ── main ──────────────────────────────────────────────────────────────────────

def generate(vindex_path: str, k: int, seed: int,
             gate_cos_threshold: float = 0.3,
             same_concept_threshold: float = 0.9) -> None:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    print(f"Reading vindex: {vindex_path}")
    index = read_index(vindex_path)
    num_layers = index["num_layers"]

    layer_bands = index.get("layer_bands", {})
    if "knowledge" in layer_bands:
        kl_min = layer_bands["knowledge"][0]
        kl_max = layer_bands["knowledge"][1] + 1
        print(f"  Knowledge layers (from layer_bands): {kl_min}–{kl_max - 1}")
    else:
        kl_min, kl_max = min(14, num_layers), min(28, num_layers)
        print(f"  Knowledge layers (heuristic): {kl_min}–{kl_max - 1}")

    embed = read_embeddings(vindex_path, index)
    print(f"  Embeddings: {embed.shape}")

    tokenizer_path = os.path.join(vindex_path, "tokenizer.json")
    ww_ids, ww_embed, special_ids = build_whole_word_vocab(tokenizer_path, embed)
    print(f"  ASCII-Latin whole-word vocab: {len(ww_ids):,} tokens  "
          f"(special IDs excluded: {len(special_ids):,})")

    tok_json = json.load(open(tokenizer_path))

    print("  Reading down_meta ...")
    triples = read_down_meta(vindex_path)

    layers_info = index.get("layers", [])
    layer_of = np.empty(len(triples), dtype=np.int32)
    feat_global = 0
    for li in layers_info:
        n = li["num_features"]
        layer_of[feat_global:feat_global + n] = li["layer"]
        feat_global += n

    kl_mask = (layer_of >= kl_min) & (layer_of < kl_max)
    kl_indices = np.where(kl_mask)[0]
    kl_output_ids = np.array([triples[i][2] for i in kl_indices], dtype=np.int32)
    print(f"  Knowledge-layer features: {len(kl_indices):,}")

    print("  Loading gate vectors ...")
    all_gates = read_gate_vectors(vindex_path, index)
    kl_gates = all_gates[kl_indices]

    ww_norm = ww_embed / (np.linalg.norm(ww_embed, axis=1, keepdims=True) + 1e-8)

    print(f"  Gate token NN search ({len(kl_indices):,} × {len(ww_ids):,} ww-tokens) ...")
    batch = 2048
    n_kl = len(kl_indices)
    input_token_ids = np.zeros(n_kl, dtype=np.int32)
    gate_best_cos = np.zeros(n_kl, dtype=np.float32)
    n_batches = (n_kl + batch - 1) // batch

    import time
    t0 = time.perf_counter()
    for bi, start in enumerate(range(0, n_kl, batch)):
        end = min(start + batch, n_kl)
        g = kl_gates[start:end]
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        scores = g_norm @ ww_norm.T
        best = scores.argmax(axis=1)
        input_token_ids[start:end] = ww_ids[best]
        gate_best_cos[start:end] = scores[np.arange(len(best)), best]
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"    batch {bi+1}/{n_batches}  {elapsed:.0f}s elapsed  {eta:.0f}s eta",
                  end="\r", flush=True)
    print()

    # ── Filter 1: valid output token (real word, not special/unused)
    valid_out = (
        (kl_output_ids > 0) &
        (~np.isin(kl_output_ids, list(special_ids))) &
        (kl_output_ids < embed.shape[0])
    )
    print(f"  After output-token filter:     {valid_out.sum():>8,}  "
          f"(dropped {(~valid_out).sum():,} special/unused output tokens)")

    # ── Filter 2: gate confidence threshold
    conf_mask = gate_best_cos >= gate_cos_threshold
    valid = valid_out & conf_mask
    print(f"  After gate cosine ≥{gate_cos_threshold}:       {valid.sum():>8,}  "
          f"(dropped {(valid_out & ~conf_mask).sum():,} low-confidence features)")

    # ── Filter 3: not same token
    valid &= (input_token_ids != kl_output_ids)
    print(f"  After in≠out filter:           {valid.sum():>8,}")

    # ── Compute offsets
    out_vecs = embed[kl_output_ids[valid]]
    in_vecs  = embed[input_token_ids[valid]]

    # ── Filter 4: same-concept filter (cosine similarity too high → no real relation)
    dot = np.einsum("ij,ij->i",
                    out_vecs / (np.linalg.norm(out_vecs, axis=1, keepdims=True) + 1e-8),
                    in_vecs  / (np.linalg.norm(in_vecs,  axis=1, keepdims=True) + 1e-8))
    diff_mask = dot <= same_concept_threshold
    out_vecs = out_vecs[diff_mask]
    in_vecs  = in_vecs[diff_mask]
    valid_indices = np.where(valid)[0][diff_mask]
    print(f"  After cosine(out,in) ≤{same_concept_threshold}:    {diff_mask.sum():>8,}  "
          f"(dropped {(~diff_mask).sum():,} same-concept pairs)")

    directions = (out_vecs - in_vecs).astype(np.float32)
    directions = normalize(directions, norm="l2")
    norms = np.linalg.norm(directions, axis=1)
    keep = norms > 0.1
    directions = directions[keep]
    valid_indices = valid_indices[keep]
    print(f"  After degenerate filter:       {len(directions):>8,}")

    final_out_ids = kl_output_ids[valid_indices]
    final_in_ids  = input_token_ids[valid_indices]

    # ── K-means
    actual_k = min(k, len(directions))
    print(f"\n  Fitting k-means (k={actual_k}) ...")
    km = MiniBatchKMeans(n_clusters=actual_k, random_state=seed, batch_size=4096,
                         max_iter=300, n_init=5)
    labels = km.fit_predict(directions)
    centres = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    print(f"  Done.  Inertia: {km.inertia_:.1f}")

    counts = np.bincount(labels, minlength=actual_k).tolist()

    # ── Populate top_tokens per cluster (most common output tokens, decoded)
    inv_vocab = {v: k.replace("\u2581", " ").strip()
                 for k, v in tok_json["model"]["vocab"].items()}

    def decode(tid):
        return inv_vocab.get(int(tid), f"#{tid}")

    top_tokens: list[list[str]] = []
    labels_list: list[str] = []
    for c in range(actual_k):
        mask = labels == c
        if not mask.any():
            top_tokens.append([])
            labels_list.append(f"cluster_{c}")
            continue
        out_ids = final_out_ids[mask]
        in_ids  = final_in_ids[mask]
        # Most common "input→output" pair as label
        pairs = list(zip(in_ids.tolist(), out_ids.tolist()))
        from collections import Counter
        common = Counter(pairs).most_common(5)
        cluster_tops = [f"{decode(inp)}→{decode(out)}" for (inp, out), _ in common]
        top_tokens.append(cluster_tops)
        labels_list.append(cluster_tops[0] if cluster_tops else f"cluster_{c}")

    result = {
        "k": actual_k,
        "centres": centres.tolist(),
        "labels": labels_list,
        "counts": counts,
        "top_tokens": top_tokens,
        "provenance": {
            "gate_cos_threshold": gate_cos_threshold,
            "same_concept_threshold": same_concept_threshold,
            "ww_vocab_size": len(ww_ids),
            "n_features_before_filter": int(kl_mask.sum()),
            "n_directions_final": len(directions),
            "knowledge_layers": [kl_min, kl_max - 1],
            "seed": seed,
        },
    }

    out_path = os.path.join(vindex_path, "relation_clusters.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB, k={actual_k})")
    print(f"Sample labels: {labels_list[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate relation_clusters.json via k-means on offset directions."
    )
    parser.add_argument("vindex", help="Path to .vindex directory")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-cos-threshold", type=float, default=0.3,
                        help="Min cosine similarity for gate→whole-word match (default 0.3)")
    parser.add_argument("--same-concept-threshold", type=float, default=0.9,
                        help="Max cosine(E[out], E[in]) — drop if too similar (default 0.9)")
    args = parser.parse_args()

    if not os.path.isdir(args.vindex):
        sys.exit(f"Not a directory: {args.vindex}")

    generate(args.vindex, args.k, args.seed,
             args.gate_cos_threshold, args.same_concept_threshold)


if __name__ == "__main__":
    main()
