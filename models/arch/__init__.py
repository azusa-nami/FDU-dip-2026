# Add your custom network here
from .default import DRNet
import torch.nn as nn


def basenet(in_channels, out_channels, fusion_type='film_cross', full_res=True, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, fusion_type=fusion_type, full_res=full_res, **kwargs)


def errnet(in_channels, out_channels, fusion_type='film_cross', full_res=True, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, fusion_type=fusion_type, full_res=full_res, **kwargs)
