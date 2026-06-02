#!/bin/bash
# ============================================================
# Evaluate all B-series experiments on every benchmark.
#
# Usage: bash scripts/test_b_series.sh [gpu_ids]
#   e.g.  bash scripts/test_b_series.sh 0
#         bash scripts/test_b_series.sh 0,1,2,3
# ============================================================

set -euo pipefail

GPU_IDS="${1:-0}"

# ---------- paths ----------
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
RESULT_ROOT="${RESULT_ROOT:-./results/b_series}"
DINO_PATH="${DINO_PATH:-/oldhome/zengyuqi/model/dinov3}"

# ---------- experiment name -> (checkpoint, fusion_type, full_res_flag) ----------
# Format: "exp_name ckpt fusion_type extra_flags"
EXPERIMENTS=(
    "b1_none         checkpoints/b1_none/errnet_latest.pt         none        "
    "b2_film         checkpoints/b2_film/errnet_latest.pt         film        "
    "b3_cross_attn   checkpoints/b3_cross_attn/errnet_latest.pt   cross_attn  "
    "b4_film_cross   checkpoints/b4_film_cross/errnet_latest.pt   film_cross  "
    "b5_downsampled  checkpoints/b5_downsampled/errnet_latest.pt  film_cross  --no_full_res"
)

for entry in "${EXPERIMENTS[@]}"; do
    read -r exp_name ckpt fusion_type extra_flags <<< "${entry}"

    if [ ! -f "${ckpt}" ]; then
        echo "[skip] ${exp_name}: checkpoint not found at ${ckpt}"
        continue
    fi

    echo ""
    echo "============================================"
    echo " Testing: ${exp_name}"
    echo "   fusion_type: ${fusion_type}"
    echo "   extra:       ${extra_flags}"
    echo "============================================"

    python test_errnet.py \
        --name "${exp_name}" \
        --dataset all \
        --data_root "${DATA_ROOT}" \
        --result_dir "${RESULT_ROOT}/${exp_name}" \
        --icnn_path "${ckpt}" \
        --hyper \
        --fusion_type "${fusion_type}" \
        --feature_model_path "${DINO_PATH}" \
        --gpu_ids "${GPU_IDS}" \
        ${extra_flags}

    echo "[done] ${exp_name} -> ${RESULT_ROOT}/${exp_name}"
done

echo ""
echo "============================================"
echo " All B-series evaluation finished."
echo " Results under: ${RESULT_ROOT}/"
echo "============================================"
