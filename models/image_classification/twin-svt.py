from typing import Optional

import torch
from timm.layers import to_2tuple, trunc_normal_, DropPath
from torch import nn, optim

from timm.models.vision_transformer import Block as TimmBlock
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.image_classification.base import BaseTransformer
from utils.load_data import get_train_test_loaders

"""
Reference: https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py#L161 (Slightly Modified)
"""


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dims):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (32, 32)
        self.img_size = img_size  # (4, 4)

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"Image size {img_size} should be divided by patch size {patch_size}"

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # (H, W) grid patches
        self.num_patches = self.H * self.W  # Total number of patches (64)

        self.proj = nn.Conv2d(in_chans, embed_dims, kernel_size=patch_size,
                              stride=patch_size)  # Non-overlapping Patches
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        B, C, H, W = x.shape  # (128, 3, 32, 32)

        x = self.proj(x)  # (128, 64, 8, 8)   -> (batch_size, embed_dim, height, width)
        x = x.flatten(2)  # (128, 64, 64)
        x = x.transpose(1, 2)  # (128, 64, 64)
        x = self.norm(x)  # (128, 64, 64)

        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=s, padding=1, bias=True, groups=embed_dim)
        self.s = s


class Attention(nn.Module):
    """
    Global sub-sampled attention (GSA)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GroupAttention(nn.Module):
    """
   Locally-grouped self-attention (LSA)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                         qk_norm=qk_scale, proj_drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                         act_layer=act_layer, norm_layer=norm_layer)
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TwinSVT(BaseTransformer):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[4, 4, 4], sr_ratios=[4, 2, 1],
                 block_cls=GroupBlock, wss=[7, 7, 7]):

        super().__init__()

        self.num_classes = num_classes
        self.depths = depths

        # Patch Embedding
        self.patch_embeds = nn.ModuleList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                                    embed_dims=embed_dims[i]))
            else:
                img_size = img_size // patch_size // 2 ** (i - 1)
                patch_size = 2
                in_chans = embed_dims[i - 1]
                self.patch_embeds.append(PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                                    embed_dims=embed_dims[i]))

            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[
                -1].num_patches

        # Positional Encodings
        self.pos_block = nn.ModuleList([PosCNN(in_chans=embed_dim, embed_dim=embed_dim) for embed_dim in embed_dims])

        # Pos Drops
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # Blocks
        self.wss = wss  # wss = window size sequence
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)

        # Classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x.mean(dim=1)  # GAP here

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

    def train_model(self, model, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
                    val_loader: Optional[DataLoader] = None):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_losses, val_losses, test_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            correct_train, total_train = 0, 0

            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for images, labels in train_loader_tqdm:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                # Update tqdm description for training progress
                train_loader_tqdm.set_postfix({
                    "Train Loss": running_train_loss / total_train,
                    "Train Acc": correct_train / total_train
                })

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_train_accuracy = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)

            # Validation phase
            if val_loader:
                model.eval()
                running_val_loss = 0.0
                correct_val, total_val = 0, 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                epoch_val_accuracy = correct_val / total_val
                val_losses.append(epoch_val_loss)
                val_accuracies.append(epoch_val_accuracy)
            else:
                epoch_val_loss = "N/A"
                epoch_val_accuracy = "N/A"

            # Testing phase
            model.eval()
            running_test_loss = 0.0
            correct_test, total_test = 0, 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(test_loader.dataset)
            epoch_test_accuracy = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_accuracy)

            tqdm.write(f"Epoch {epoch + 1}/{epochs} - "
                       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
                       f"Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_accuracy}, "
                       f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.4f}")

        return {"train_loss": train_losses, "val_loss": val_losses if val_loader else None, "test_loss": test_losses,
                "train_accuracy": train_accuracies, "val_accuracy": val_accuracies if val_loader else None,
                "test_accuracy": test_accuracies}


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=128,
                                                                   val_split=0.2, num_workers=4)
    device = 'mps'

    svt = TwinSVT(img_size=32, patch_size=2, in_chans=3, num_classes=100, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 10, 4],
                  sr_ratios=[8, 4, 2, 1], wss=[7, 7, 7, 7])
    svt.to(device)
    metrics = svt.train_model(svt, train_loader, test_loader, 100, val_loader)
