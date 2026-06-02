#!/bin/bash
# ============================================================
# Run all B-series experiments sequentially.
#
# B1: full-res + no DINO fusion (baseline for this ablation)
# B2: full-res + FiLM fusion only
# B3: full-res + Cross-Attention fusion only
# B4: full-res + FiLM + Cross-Attention (current default)
# B5: downsampled + FiLM + Cross-Attention (test full-res value)
#
# All use --hyper (VGG hypercolumn + frozen DINOv3 features)
# and the same modern training recipe.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_b_series.sh"

chmod +x "${TRAIN_SCRIPT}"

# ---------- configurable paths ----------
export DATA_ROOT="${DATA_ROOT:-./datasets/data}"
export DINO_PATH="${DINO_PATH:-/oldhome/zengyuqi/model/dinov3}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "DINO_PATH=${DINO_PATH}"

# ============================================================
# B1: no fusion (VGG hypercolumn only, full-res architecture)
# ============================================================
echo ""
echo "############################################################"
echo "# B1: fusion_type=none  (full-res, no DINO fusion)"
echo "############################################################"
bash "${TRAIN_SCRIPT}" b1_none --fusion_type none

# ============================================================
# B2: FiLM fusion only
# ============================================================
echo ""
echo "############################################################"
echo "# B2: fusion_type=film  (full-res, FiLM only)"
echo "############################################################"
bash "${TRAIN_SCRIPT}" b2_film --fusion_type film

# ============================================================
# B3: Cross-Attention fusion only
# ============================================================
echo ""
echo "############################################################"
echo "# B3: fusion_type=cross_attn  (full-res, CrossAttn only)"
echo "############################################################"
bash "${TRAIN_SCRIPT}" b3_cross_attn --fusion_type cross_attn

# ============================================================
# B4: FiLM + Cross-Attention (current default)
# ============================================================
echo ""
echo "############################################################"
echo "# B4: fusion_type=film_cross  (full-res, both)"
echo "############################################################"
bash "${TRAIN_SCRIPT}" b4_film_cross --fusion_type film_cross

# ============================================================
# B5: FiLM + Cross-Attention but downsampled (no full-res)
# ============================================================
echo ""
echo "############################################################"
echo "# B5: fusion_type=film_cross --no_full_res  (downsampled, both)"
echo "############################################################"
bash "${TRAIN_SCRIPT}" b5_downsampled --fusion_type film_cross --no_full_res

echo ""
echo "============================================"
echo " All B-series experiments finished."
echo " Checkpoints saved under checkpoints/b[1-5]_*/"
echo "============================================"
