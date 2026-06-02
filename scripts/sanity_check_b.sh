#!/bin/bash
# ============================================================
# Quick sanity check: build each B-series model and print
# parameter counts. Uses a minimal import path to avoid
# triggering heavy dependencies (transformers, sklearn, etc).
# ============================================================
set -euo pipefail

echo "============================================"
echo " B-Series Architecture Sanity Check"
echo "============================================"

python -c "
import sys, types, os

# --- Build a minimal 'models' package so that models/__init__.py
# --- is never executed (it triggers errnet_model -> losses -> dinov3 -> transformers).
_mpkg = types.ModuleType('models')
_mpkg.__path__ = [os.path.join(os.getcwd(), 'models')]
_mpkg.__file__ = os.path.join(os.getcwd(), 'models', '__init__.py')
sys.modules['models'] = _mpkg

# Also pre-seed arch sub-package so Python can find models/arch/default.py
_apkg = types.ModuleType('models.arch')
_apkg.__path__ = [os.path.join(os.getcwd(), 'models', 'arch')]
_apkg.__file__ = os.path.join(os.getcwd(), 'models', 'arch', '__init__.py')
sys.modules['models.arch'] = _apkg

# And pre-seed arch's parent init symbols (DRNet is re-exported)
from models.arch.default import DRNet

in_ch = 3 + 64 + 128 + 256 + 512 + 512  # VGG hypercolumn feature concat

configs = [
    ('B1 (full-res, none)',          {'fusion_type': 'none',       'full_res': True,  'dino_channels': 768}),
    ('B2 (full-res, film)',          {'fusion_type': 'film',       'full_res': True,  'dino_channels': 768}),
    ('B3 (full-res, cross_attn)',    {'fusion_type': 'cross_attn', 'full_res': True,  'dino_channels': 768}),
    ('B4 (full-res, film_cross)',    {'fusion_type': 'film_cross', 'full_res': True,  'dino_channels': 768}),
    ('B5 (downsampled, film_cross)', {'fusion_type': 'film_cross', 'full_res': False, 'dino_channels': 768}),
]

for label, cfg in configs:
    model = DRNet(in_ch, 3, 256, 13, norm=None, res_scale=0.1,
                  se_reduction=8, bottom_kernel_size=1, pyramid=True, **cfg)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_film = model.film_fusion is not None
    has_cross = model.cross_fusion is not None
    has_conv3 = hasattr(model, 'conv3')
    has_deconv1 = hasattr(model, 'deconv1')
    arch = 'full-res' if model.full_res else 'downsampled'
    print(f'{label}:')
    print(f'  params={params:,}  trainable={trainable:,}  arch={arch}')
    print(f'  FiLM={has_film}  CrossAttn={has_cross}  conv3={has_conv3}  deconv1={has_deconv1}')

print()
print('All architectures built successfully.')
"
