import math
from abc import ABC
from functools import partial
from typing import Callable, Optional

import torch
from timm.layers import to_2tuple, DropPath, trunc_normal_
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.image_classification.base import BaseTransformer
from utils.load_data import get_train_test_loaders

"""
Reference: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/tnt_pytorch/tnt.py
"""


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, inner_dim, inner_stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        image_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim

        # Number of patches in sub-patches (Sentence = Patch & Word = patch in a sub-patch)
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(3, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        assert height == self.image_size[0] and width == self.image_size[
            1], "Input Image and Expected size doesn't match"
        images = self.unfold(images)
        images = images.transpose(1, 2).reshape(batch_size * self.num_patches, channels * self.patch_size)
        images = self.proj(images)
        images = images.reshape(batch_size * self.num_patches, self.inner_dim, -1).transpose(1, 2)

        return images


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


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner Transformer
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = Attention(inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(inner_dim)
            self.inner_mlp = MLP(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.droppath = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = MLP(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, inner_tokens, outer_tokens):
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))
            B, N, C = outer_tokens.size()

            outer_tokens[:, 1:] = outer_tokens[:, 1:] + self.proj_norm2(
                self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1))))

        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))

        return inner_tokens, outer_tokens


class TNT(BaseTransformer, ABC):
    def __init__(self, image_size=32, patch_size=8, num_classes=100, outer_dim=512, inner_dim=48, num_layers=7,
                 outer_num_heads=4, inner_num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None, dropout=0.0,
                 attention_dropout=0.0, drop_path_rate=0.0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 inner_stride=4, se=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim

        self.patch_embed = PatchEmbedding(image_size, patch_size, inner_dim, inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer(num_words)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, inner_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic Depth Decay Rule
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        idxs = []
        blocks = []
        for i in range(num_layers):
            if i in idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=dropout,
                    attn_drop=attention_dropout, drop_path=self.dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=dropout,
                    attn_drop=attention_dropout, drop_path=self.dpr[i], norm_layer=norm_layer, se=se))

            self.blocks = nn.ModuleList(blocks)
            self.norm = norm_layer(outer_dim)

            # Classifier head
            self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.outer_pos, std=.02)
            trunc_normal_(self.inner_pos, std=.02)
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
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]

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
    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256,
                                                                   val_split=0.2, num_workers=4)
    tnt = TNT()
    metrics = tnt.train_model(tnt, train_loader, test_loader, 100, val_loader)
