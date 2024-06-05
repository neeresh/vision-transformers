from abc import ABC

import torch
from timm.layers import to_2tuple, DropPath, trunc_normal_
from torch import nn

from models.image_classification.base import BaseTransformer
from utils.args import get_args
from utils.load_data import get_train_test_loaders


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()

        image_size = to_2tuple(image_size)  # (image_size, image_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)

        self.image_size = image_size
        self.patch_size = patch_size

        self.height, self.width = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        images = self.proj(images).flatten(2).transpose(1, 2)
        images = self.norm(images)
        height, width = height // self.patch_size[0], width // self.patch_size[1]

        return images, (height, width)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        batch_size, N, C = x.shape
        q = self.q(x).reshape(batch_size, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, C, height, width)
            x_ = self.sr(x_).reshape(batch_size, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PVT(BaseTransformer, ABC):
    def __init__(self, image_size=32, patch_size=16, in_channels=3, num_classes=100, embed_dims=None, num_heads=None,
                 mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer=nn.LayerNorm, depths=None, sr_ratios=None, num_stages=4, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depths is None:
            depths = [3, 4, 6, 3]
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        # Stochastic Depth Decay Rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i in range(num_stages):
            patch_embedding = PatchEmbedding(image_size=image_size if i == 0 else image_size // (2 ** (i + 1)),
                                             patch_size=patch_size if i == 0 else 2,
                                             in_channels=in_channels if i == 0 else embed_dims[i - 1],
                                             embed_dim=embed_dims[i])
            num_patches = patch_embedding.num_patches if i != num_stages - 1 else patch_embedding.num_patches + 1
            position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            position_drop = nn.Dropout(p=drop_rate)

            cur = 0
            block = nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[i]) for j in range(depths[i])
            ])
            cur += depths[i]

            setattr(self, f"patch_embedding{i + 1}", patch_embedding)
            setattr(self, f"position_embedding{i + 1}", position_embedding)
            setattr(self, f"position_drop{i + 1}", position_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        for i in range(num_stages):
            pos_embed = getattr(self, f"position_embedding{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embedding{i + 1}")
            pos_embed = getattr(self, f"position_embedding{i + 1}")
            pos_drop = getattr(self, f"position_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256,
                                                                   val_split=0.2, num_workers=4)
    args = get_args("vit_tiny_cifar100")
    pvt = PVT(image_size=32, patch_size=16, in_channels=3, num_classes=100, embed_dims=None, num_heads=None,
              mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
              norm_layer=nn.LayerNorm, depths=None, sr_ratios=None, num_stages=4)
    # vit.to("cuda")
    print(pvt)
    # metrics = vit.train_model(vit, train_loader, test_loader, 100, val_loader)
