import math
import warnings
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Callable, List, Optional
from functools import partial

from models.image_classification.vanilla_vit import ViT, Encoder


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class DeiT(ViT):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 num_classes, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), *args,
                 **kwargs):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                         num_classes, norm_layer)

        sequence_length = self.num_patches + 2
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, self.hidden_dim))
        self.encoder = Encoder(seq_length=sequence_length, num_layers=num_layers, num_heads=num_heads,
                               hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout,
                               attention_dropout=attention_dropout, norm_layer=norm_layer)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.head_dist = nn.Linear(self.hidden_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.distillation_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, images):
        B = images.shape[0]
        x = self.conv_proj(images).flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        dist_token = self.distillation_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.encoder(x)
        return x[:, 0], x[:, 1]

    def forward(self, images):
        x, x_dist = self.forward_features(images)
        x = self.heads(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2
