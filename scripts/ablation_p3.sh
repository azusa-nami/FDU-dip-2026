#!/bin/bash
# =============================================================================
# Ablation P3: Training strategy ablation
# 4 experiments (E18 == E05 full, already covered)
#
# All use: mixed synth + DINO(both)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DINO_MODEL_PATH="${DINO_MODEL_PATH:?must set DINO_MODEL_PATH env var or edit this script}"
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"

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
--hyper --fusion_mode both \
--synthesis mixed --synth_mix_ratio 0.8,0.2 \
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

# ---- E14: adam_manual ----
# Adam + manual LR schedule, no grad clip, no EMA, no early stop
run_exp ablation_adam_manual \
    --optimizer adam --lr_policy manual --wd 0 \
    --clip_grad_norm 0 --ema_decay 0 --early_stop_patience 0

# ---- E15: adamw_cosine ----
# AdamW + cosine LR, no grad clip, no EMA
run_exp ablation_adamw_cosine \
    --optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
    --clip_grad_norm 0 --ema_decay 0 --early_stop_patience 0

# ---- E16: +grad_clip ----
# E15 + gradient clipping
run_exp ablation_grad_clip \
    --optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
    --clip_grad_norm 1.0 --ema_decay 0 --early_stop_patience 0

# ---- E17: +ema ----
# E16 + EMA
run_exp ablation_ema \
    --optimizer adamw --lr_policy cosine --wd 1e-4 --min_lr 1e-6 \
    --clip_grad_norm 1.0 --ema_decay 0.999 --early_stop_patience 0

# ---- E18: +early_stop (== E05 full) ----
# Already covered by ablation_full in P0 script.
# You can symlink or copy results from there.

echo ""
echo "========== All P3 experiments submitted =========="
echo "Note: E18 (+early_stop) is equivalent to E05 (ablation_full) from P0 batch."
