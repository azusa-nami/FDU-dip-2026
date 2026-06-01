# Define network components here
import torch
from torch import nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1, bottleneck=True):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.out_channels = in_channels + len(scales) * ct_channels
        self.bottleneck = None
        if bottleneck:
            self.bottleneck = nn.Conv2d(self.out_channels, out_channels, kernel_size=1, stride=1)
            self.out_channels = out_channels
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        if self.bottleneck is not None:
            priors = self.bottleneck(priors)
        return self.relu(priors)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DINOFiLMFusion(nn.Module):
    def __init__(
        self,
        cnn_channels,
        dino_channels,
        target_size=56,
        hidden_channels=None,
        gamma_scale=0.1,
        beta_scale=0.1,
    ):
        super().__init__()
        hidden_channels = hidden_channels or cnn_channels
        self.target_size = target_size
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.dino_to_film = nn.Sequential(
            nn.Conv2d(dino_channels, hidden_channels, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, cnn_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, cnn_feature, dino_feature):
        if dino_feature is None:
            return cnn_feature

        height, width = cnn_feature.shape[-2:]
        low_size = (self.target_size, self.target_size)
        cnn_low = F.adaptive_avg_pool2d(cnn_feature, low_size)
        dino_low = F.interpolate(dino_feature, size=low_size, mode='bilinear', align_corners=False)
        gamma, beta = self.dino_to_film(dino_low).chunk(2, dim=1)
        gamma = 1.0 + self.gamma_scale * torch.tanh(gamma)
        beta = self.beta_scale * beta

        modulated = gamma * cnn_low + beta
        delta = F.interpolate(modulated - cnn_low, size=(height, width), mode='bilinear', align_corners=False)
        return cnn_feature + delta


class DINO16CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        cnn_channels,
        dino_channels,
        attn_channels=None,
        num_heads=8,
        target_size=16,
        dropout=0.0,
    ):
        super().__init__()
        attn_channels = attn_channels or cnn_channels
        self.target_size = target_size

        self.cnn_proj = nn.Conv2d(cnn_channels, attn_channels, kernel_size=1)
        self.dino_proj = nn.Conv2d(dino_channels, attn_channels, kernel_size=1)
        self.cnn_norm = nn.LayerNorm(attn_channels)
        self.dino_norm = nn.LayerNorm(attn_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(attn_channels, cnn_channels, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _to_tokens(self, feature, norm):
        tokens = feature.flatten(2).transpose(1, 2)
        return norm(tokens)

    def forward(self, cnn_feature, dino_feature):
        if dino_feature is None:
            return cnn_feature

        _, _, height, width = cnn_feature.shape
        low_size = (self.target_size, self.target_size)
        cnn_low = F.adaptive_avg_pool2d(cnn_feature, low_size)

        dino_low = F.interpolate(dino_feature, size=low_size, mode='bilinear', align_corners=False)
        query_feature = self.cnn_proj(cnn_low)
        context_feature = self.dino_proj(dino_low)

        query = self._to_tokens(query_feature, self.cnn_norm)
        context = self._to_tokens(context_feature, self.dino_norm)
        fused, _ = self.attn(query, context, context, need_weights=False)
        fused = fused.transpose(1, 2).reshape(query_feature.shape)
        fused = self.out_proj(fused)
        fused = F.interpolate(fused, size=(height, width), mode='bilinear', align_corners=False)
        return cnn_feature + self.gamma * fused


class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d,
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False,
    dino_channels=None, fusion_mode='both', fusion_strided=False,
    fusion_heads=8, film_size=56, cross_attn_size=16,
    film_after_blocks=4, cross_after_blocks=8):
        super(DRNet, self).__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.fusion_mode = fusion_mode
        self.full_res_fusion = dino_channels is not None and fusion_mode != 'none' and not fusion_strided

        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        if not self.full_res_fusion:
            self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)
        self.film_fusion = None
        self.cross_fusion = None
        if dino_channels is not None and fusion_mode != 'none':
            if fusion_mode in ('film', 'both'):
                self.film_fusion = DINOFiLMFusion(
                    n_feats,
                    dino_channels,
                    target_size=film_size,
                    hidden_channels=n_feats,
                )
            if fusion_mode in ('cross', 'both'):
                self.cross_fusion = DINO16CrossAttentionFusion(
                    n_feats,
                    dino_channels,
                    attn_channels=n_feats,
                    num_heads=fusion_heads,
                    target_size=cross_attn_size,
                )

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])
        self.film_after_blocks = min(film_after_blocks, n_resblocks)
        self.cross_after_blocks = min(max(cross_after_blocks, self.film_after_blocks), n_resblocks)

        # Upsampling Layers
        if not self.full_res_fusion:
            self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(
                n_feats,
                n_feats,
                scales=(4,8,16,32),
                ct_channels=n_feats//4,
                bottleneck=not self.full_res_fusion,
            )
            self.deconv3 = ConvLayer(conv, self.pyramid_module.out_channels, out_channels, kernel_size=1, stride=1, norm=None, act=act)

    def forward(self, x, dino_feature=None):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.full_res_fusion:
            for block in self.res_module[:self.film_after_blocks]:
                x = block(x)
            if self.film_fusion is not None:
                x = self.film_fusion(x, dino_feature)
            for block in self.res_module[self.film_after_blocks:self.cross_after_blocks]:
                x = block(x)
            if self.cross_fusion is not None:
                x = self.cross_fusion(x, dino_feature)
            for block in self.res_module[self.cross_after_blocks:]:
                x = block(x)
        else:
            x = self.conv3(x)
            x = self.res_module(x)

            if self.fusion_mode != 'none' and dino_feature is not None:
                if self.film_fusion is not None:
                    x = self.film_fusion(x, dino_feature)
                if self.cross_fusion is not None:
                    x = self.cross_fusion(x, dino_feature)

        if not self.full_res_fusion:
            x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
