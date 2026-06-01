#!/bin/bash
# =============================================================================
# Ablation P1: DINO fusion internal ablation
# 4 experiments: film_only, cross_only, dino_none, both+strided
#
# All use: mixed synth + modern training (same as E05 full)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DINO_MODEL_PATH="${DINO_MODEL_PATH:?must set DINO_MODEL_PATH env var or edit this script}"
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"

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
--seed 2018 \
--synthesis mixed --synth_mix_ratio 0.8,0.2 \
--optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
--clip_grad_norm 1.0 --ema_decay 0.999 \
--lambda_vgg 0.05
ARGS
}

run_exp() {
    local exp_name="$1"; shift
    echo ""
    echo "========== $(date): ${exp_name} =========="

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  [DRY RUN] ${exp_name}: $@"
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

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- E06: film_only ----
# DINO FiLM fusion only (no Cross-Attention), full-res
run_exp ablation_film_only \
    --hyper --fusion_mode film

# ---- E07: cross_only ----
# DINO Cross-Attention only (no FiLM), full-res
run_exp ablation_cross_only \
    --hyper --fusion_mode cross

# ---- E08: dino_none ----
# VGG hypercolumn input + DINO model loaded but no fusion into backbone
# This isolates whether the DINO features even as input give gains
run_exp ablation_dino_none \
    --hyper --fusion_mode none

# ---- E09: both+strided ----
# DINO both fusion but keep stride-2 downsampling
run_exp ablation_both_strided \
    --hyper --fusion_mode both --fusion_strided

echo ""
echo "========== All P1 experiments submitted =========="
