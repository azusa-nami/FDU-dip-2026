import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv3Features(nn.Module):
    handles_normalization = True

    def __init__(
        self,
        model_path="/oldhome/zengyuqi/model/dinov3",
        layers=None,
        feature_scale=0.1,
        normalize_features=True,
        requires_grad=False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_path),
        )
        self.layers = layers or [6, 12, 18, 24]
        self.feature_scale = feature_scale
        self.normalize_features = normalize_features
        self.patch_size = int(self.model.config.patch_size)
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))
        self.out_channels = int(self.model.config.hidden_size)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.parameters():
            param.requires_grad = requires_grad
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def _tokens_to_feature_map(self, tokens, height, width):
        patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            grid_h = grid_w = int(patch_tokens.shape[1] ** 0.5)

        feature = patch_tokens[:, : grid_h * grid_w, :]
        feature = feature.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)
        if self.normalize_features:
            mean = feature.mean(dim=(1, 2, 3), keepdim=True)
            std = feature.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            feature = (feature - mean) / std
        feature = feature * self.feature_scale
        return feature

    def forward(self, x, indices=None):
        _, _, height, width = x.shape
        normalized = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

        outputs = self.model(pixel_values=normalized, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        selected_layers = indices or self.layers
        features = []
        for layer in selected_layers:
            feature = self._tokens_to_feature_map(hidden_states[layer], height, width)
            features.append(feature)
        return features
