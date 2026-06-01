#!/bin/bash
# =============================================================================
# Ablation P0: Core improvement validation
# 5 experiments isolating the contribution of DINO fusion, synthesis, and training
#
# Usage:
#   bash scripts/ablation_p0.sh [--dry-run]
#
# Prerequisites:
#   - Prepared datasets at ./datasets/data/
#   - DINOv3 model at DINO_MODEL_PATH
#   - uv installed with dependencies from pyproject.toml
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# --------------- configurable paths ---------------
# Download DINOv3 from https://huggingface.co/facebook/dinov3-vitb16-pretrained-lvd-142M
# or use transformers caching: set DINO_MODEL_PATH="facebook/dinov3-vitb16-pretrained-lvd-142M"
DINO_MODEL_PATH="${DINO_MODEL_PATH:?must set DINO_MODEL_PATH env var or edit this script}"
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

# --------------- common base arguments ---------------
common_args() {
    local name="$1"
    cat <<ARGS
--name ${name} \
--data_root ${DATA_ROOT} \
--feature_model_path ${DINO_MODEL_PATH} \
--batchSize 4 \
--nEpochs 60 \
--eval_freq 5 \
--early_stop_metric PSNR \
--early_stop_patience 10 \
--seed 2018
ARGS
}

# --------------- experiment runner ---------------
run_exp() {
    local exp_name="$1"; shift
    echo ""
    echo "========== $(date): ${exp_name} =========="
    echo "  CUDA_VISIBLE_DEVICES=${GPUS}"
    echo "  args: $@"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  [DRY RUN] skipping"
        return
    fi

    CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun \
        --master_port="${MASTER_PORT_BASE}" \
        --nproc_per_node="${NPROC}" \
        train_errnet.py \
        $(common_args "${exp_name}") \
        "$@"

    echo "========== ${exp_name} DONE =========="
}

# =============================================================================
# Experiment definitions
#
# Matrix:
#   E01 baseline : pre-trained checkpoint (checkpoints/errnet/errnet_060_00463920.pt) — SKIPPED, already exists
#   E02 +DINO    : legacy synth + DINO(both) + manual training
#   E03 +Synth   : mixed synth + no DINO    + manual training
#   E04 +Train   : legacy synth + no DINO   + modern training
#   E05 full     : mixed synth + DINO(both) + modern training
#
# Total: 4 experiments to train
# =============================================================================

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- E01: baseline (SKIPPED) ----
# Uses pre-trained checkpoint from baseline branch:
#   checkpoints/errnet/errnet_060_00463920.pt
# Include it in evaluation via: python scripts/ablation_eval.py --include-baseline-ref

# ---- E02: +DINO ----
# legacy synthesis, DINO both, manual LR schedule
run_exp ablation_dino \
    --hyper --fusion_mode both \
    --synthesis legacy \
    --optimizer adam --lr_policy manual --wd 0 \
    --clip_grad_norm 0 --ema_decay 0 \
    --lambda_vgg 0.05

# ---- E03: +Synth ----
# mixed synthesis, no DINO, manual LR schedule
run_exp ablation_synth \
    --hyper --fusion_mode none \
    --synthesis mixed --synth_mix_ratio 0.8,0.2 \
    --optimizer adam --lr_policy manual --wd 0 \
    --clip_grad_norm 0 --ema_decay 0 \
    --lambda_vgg 0.1

# ---- E04: +Train ----
# legacy synthesis, no DINO, modern training
run_exp ablation_train \
    --hyper --fusion_mode none \
    --synthesis legacy \
    --optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
    --clip_grad_norm 1.0 --ema_decay 0.999 \
    --lambda_vgg 0.1

# ---- E05: full ----
# mixed synthesis, DINO both, modern training
run_exp ablation_full \
    --hyper --fusion_mode both \
    --synthesis mixed --synth_mix_ratio 0.8,0.2 \
    --optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
    --clip_grad_norm 1.0 --ema_decay 0.999 \
    --lambda_vgg 0.05

echo ""
echo "========== All P0 experiments submitted =========="
