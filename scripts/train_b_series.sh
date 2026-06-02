#!/bin/bash
# ============================================================
# B-Series Experiment: DINOv3 Fusion Ablation
#
# Fixed settings across all B experiments:
#   --hyper            VGG hypercolumn + DINOv3 feature extraction
#   --synthesis mixed  same data pipeline
#   --lr_policy cosine modern training recipe
#   --optimizer adamw  modern training recipe
#
# Varied:
#   --fusion_type  {none, film, cross_attn, film_cross}
#   --no_full_res  (B5 only, use original downsampled path)
# ============================================================

set -euo pipefail

# ---------- paths (adjust to your server) ----------
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
DINO_PATH="${DINO_PATH:-/oldhome/zengyuqi/model/dinov3}"

# ---------- common training flags ----------
COMMON_FLAGS=(
    --hyper
    --synthesis mixed
    --lr_policy cosine
    --optimizer adamw
    --ema_decay 0.999
    --clip_grad_norm 1.0
    --nEpochs 60
    --eval_freq 5
    --early_stop_metric PSNR
    --early_stop_patience 10
    --batchSize 1
    --data_root "${DATA_ROOT}"
    --feature_model_path "${DINO_PATH}"
    --no_html
)

# ---------- experiment name ----------
EXP_NAME="${1:?Usage: $0 <exp_name> [extra_flags...]}"
shift

echo "============================================"
echo " Launching B-series: ${EXP_NAME}"
echo " Common flags: ${COMMON_FLAGS[*]}"
echo " Extra flags:  $*"
echo "============================================"

python train_errnet.py \
    --name "${EXP_NAME}" \
    "${COMMON_FLAGS[@]}" \
    "$@"
