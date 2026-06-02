#!/bin/bash
# ============================================================
# Quick sanity check: build each B-series model and print
# parameter counts. No data or GPU needed.
# ============================================================
set -euo pipefail

echo "============================================"
echo " B-Series Architecture Sanity Check"
echo "============================================"

python -c "
import torch
from models.arch import errnet

configs = {
    'B1 (full-res, none)':       {'fusion_type': 'none',       'full_res': True,  'dino_channels': 768},
    'B2 (full-res, film)':       {'fusion_type': 'film',       'full_res': True,  'dino_channels': 768},
    'B3 (full-res, cross_attn)': {'fusion_type': 'cross_attn', 'full_res': True,  'dino_channels': 768},
    'B4 (full-res, film_cross)': {'fusion_type': 'film_cross', 'full_res': True,  'dino_channels': 768},
    'B5 (downsampled, film_cross)': {'fusion_type': 'film_cross', 'full_res': False, 'dino_channels': 768},
}

for label, cfg in configs.items():
    in_ch = 3 + 64 + 128 + 256 + 512 + 512  # VGG hypercolumn
    model = errnet(in_ch, 3, **cfg)
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
