#!/usr/bin/env bash
# OLMo-2 trajectory sweep: download → extract → analyze → delete raw weights
# Run from ~/manifold-tracking/
# Each checkpoint is ~29GB download, ~3.5GB vindex. Raw weights deleted after extraction.

set -euo pipefail

# ── argument parsing ──────────────────────────────────────────────────────────
CLUSTERING="euclidean"
FORCE=0
FORCE_ANALYSIS=0
USE_NLTK=""
LOAD_CACHE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clustering) CLUSTERING="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --force-analysis) FORCE_ANALYSIS=1; shift ;;
        --use-nltk) USE_NLTK="--use-nltk"; shift ;;
        --load-cache) LOAD_CACHE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
if [[ "$CLUSTERING" != "euclidean" && "$CLUSTERING" != "spherical" ]]; then
    echo "Error: --clustering must be 'euclidean' or 'spherical'" >&2
    exit 1
fi
SUFFIX=""
[[ "$CLUSTERING" == "spherical" ]] && SUFFIX="_spherical"
[[ "$CLUSTERING" == "spherical" && -n "$USE_NLTK" ]] && SUFFIX="_spherical_nltk"
echo "Clustering method: ${CLUSTERING}${USE_NLTK:+ (nltk)}"

LARQL="./larql/target/release/larql"
CKPTS=(
    stage1-step150-tokens1B
    stage1-step600-tokens3B
    stage1-step3000-tokens13B
    stage1-step12000-tokens51B
    stage1-step51000-tokens214B
    stage1-step217000-tokens911B
    stage1-step928646-tokens3896B
)

for ckpt in "${CKPTS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  ${ckpt}"
    echo "════════════════════════════════════════════════"

    raw_dir="olmo2-checkpoints/olmo2-${ckpt}"
    vindex_dir="olmo2-checkpoints/olmo2-${ckpt}.vindex"
    analysis_out="outputs/olmo2-${ckpt}.vindex/per_cluster_svd${SUFFIX}.json"
    analysis_log="outputs/olmo2-${ckpt}.vindex/per_cluster_svd${SUFFIX}.log"

    # ── Step 1: download ──
    if [[ $FORCE -eq 0 && -f "${vindex_dir}/index.json" ]]; then
        echo "  [skip] vindex already exists"
    elif [[ $FORCE -eq 0 && -f "${raw_dir}/config.json" ]]; then
        echo "  [skip] raw weights already present"
    else
        echo "  Downloading ..."
        hf download allenai/OLMo-2-1124-7B \
            --revision "${ckpt}" \
            --local-dir "${raw_dir}"
        echo "  Download complete."
    fi

    # ── Step 2: extract ──
    if [[ $FORCE -eq 0 && -f "${vindex_dir}/index.json" ]]; then
        echo "  [skip] extraction already done"
    else
        echo "  Extracting → ${vindex_dir} ..."
        "${LARQL}" extract-index "${raw_dir}" \
            -o "${vindex_dir}" \
            --level browse \
            --f16
        echo "  Extraction complete."
    fi

    # ── Step 3: delete raw weights ──
    if [ -d "${raw_dir}" ]; then
        echo "  Deleting raw weights ..."
        rm -rf "${raw_dir}"
        echo "  Deleted."
    fi

    # ── Step 4: analyze ──
    if [[ $FORCE -eq 0 && $FORCE_ANALYSIS -eq 0 && -f "${analysis_out}" ]]; then
        echo "  [skip] analysis already done"
    else
        echo "  Running per-cluster SVD (${CLUSTERING}) ..."
        mkdir -p "$(dirname "${analysis_log}")"
        CACHE_ARG=""
        if [[ $LOAD_CACHE -eq 1 ]]; then
            base_cache="outputs/olmo2-${ckpt}.vindex/offsets_cached_spherical.npz"
            if [[ -f "${base_cache}" ]]; then
                CACHE_ARG="--load-cache ${base_cache}"
                echo "  Using cached offsets: ${base_cache}"
            else
                echo "  Warning: --load-cache requested but ${base_cache} not found; running gate search"
            fi
        fi
        python per_cluster_svd.py \
            --vindex "${vindex_dir}" \
            --gate-cos-percentile 85 \
            --k 512 \
            --n-bootstrap 100 \
            --clustering "${CLUSTERING}" \
            ${USE_NLTK} ${CACHE_ARG} 2>&1 | tee "${analysis_log}"
        echo "  Analysis complete. Log: ${analysis_log}"
    fi

done

echo ""
echo "════════════════════════════════════════════════"
echo "  Sweep complete. Aggregating ..."
echo "════════════════════════════════════════════════"
python aggregate_olmo2_trajectory.py
python visualize_olmo2_trajectory.py
echo ""
echo "Done. Results in outputs/olmo2_trajectory.{json,csv}"
