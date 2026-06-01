#!/bin/bash
# =============================================================================
# Ablation P2: Synthesis strategy ablation
# 4 experiments: legacy, reflection2, advanced, mixed(50/50)
#
# All use: DINO(both) + modern training (same as E05 full)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DINO_MODEL_PATH="${DINO_MODEL_PATH:?must set DINO_MODEL_PATH env var or edit this script}"
DATA_ROOT="${DATA_ROOT:-./datasets/data}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"

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

# ---- E10: synth_legacy ----
# Legacy ReflectionSythesis_1 (simple gaussian blur + global alpha)
run_exp ablation_synth_legacy \
    --synthesis legacy

# ---- E11: synth_reflection2 ----
# ReflectionSythesis_2 only
run_exp ablation_synth_reflection2 \
    --synthesis reflection2

# ---- E12: synth_advanced ----
# AdvancedReflectionSythesis only (all the hard cases)
run_exp ablation_synth_advanced \
    --synthesis advanced

# ---- E13: synth_mixed5050 ----
# Mixed 50% reflection2 + 50% advanced
run_exp ablation_synth_mixed5050 \
    --synthesis mixed --synth_mix_ratio 0.5,0.5

echo ""
echo "========== All P2 experiments submitted =========="
