import math
from functools import partial
from typing import Optional, Callable, List

import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import StochasticDepth, MLP, Permute
from tqdm import tqdm

from models.image_classification.base import BaseTransformer
from utils.args import get_args
from utils.load_data import get_train_test_loaders

"""
Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
"""


def _patch_merging_pad(images: torch.Tensor) -> torch.Tensor:
    H, W, _ = images.shape[-3:]
    images = F.pad(images, (0, 0, 0, W % 2, 0, H % 2))
    x0 = images[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = images[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = images[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = images[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    images = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return images


def _get_relative_position_bias(relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor,
                                window_size: List[int]) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


def shifted_window_attention(input: torch.Tensor, qkv_weight: torch.Tensor, proj_weight: torch.Tensor,
                             relative_position_bias: torch.Tensor,
                             window_size: List[int], num_heads: int, shift_size: List[int],
                             attention_dropout: float = 0.0, dropout: float = 0.0,
                             qkv_bias: Optional[torch.Tensor] = None,
                             proj_bias: Optional[torch.Tensor] = None, logit_scale: Optional[torch.Tensor] = None,
                             training: bool = True) -> torch.Tensor:
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()

    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length: 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]: h[1], w[0]: w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim  # Number of input channels
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = norm_layer(4 * dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        x (Tensor): input tensor with expected layout of [..., H, W, C]
        """
        images = _patch_merging_pad(images)
        images = self.norm(images)
        images = self.reduction(images)  # ... H/2 W/2 2*C

        return images


class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = True,
                 proj_bias: bool = True, attention_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(x, self.qkv.weight, self.proj.weight, relative_position_bias, self.window_size,
                                        self.num_heads, shift_size=self.shift_size,
                                        attention_dropout=self.attention_dropout, dropout=self.dropout,
                                        qkv_bias=self.qkv.bias, proj_bias=self.proj.bias, training=self.training)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: List[int], shift_size: List[int], mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attention_dropout: float = 0.0, stochastic_depth_prob: float = 0.0,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, window_size, shift_size, num_heads, attention_dropout=attention_dropout,
                               dropout=dropout)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(BaseTransformer):
    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int],
                 window_size: List[int], mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0,
                 stochastic_depth_prob: float = 0.1, num_classes: int = 100,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 block: Optional[Callable[..., nn.Module]] = None,
                 downsample_layer: Callable[..., nn.Module] = PatchMerging, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []

        # split image into non-overlapping patches
        layers.append(nn.Sequential(nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]),
                                              stride=(patch_size[0], patch_size[1])), Permute([0, 2, 3, 1]),
                                    norm_layer(embed_dim)))

        total_stage_blocks = sum(depths)
        stage_block_id = 0

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(block(dim, num_heads[i_stage], window_size=window_size,
                                   shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                                   mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout,
                                   stochastic_depth_prob=sd_prob, norm_layer=norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))

        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.device = 'cuda'

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
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
    # Epoch 100/100 - Train Loss: 0.1717, Train Acc: 0.9473,
    # Val Loss: 2.543592349624634, Val Acc: 0.5449,
    # Test Loss: 2.5355, Test Acc: 0.5441
    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256,
                                                                   val_split=0.2, num_workers=4)
    args = get_args("swin_tiny_cifar100")
    swin_tiny = SwinTransformer(patch_size=args["patch_size"], embed_dim=args["embed_dim"], depths=args["depths"],
                                num_heads=args["num_heads"], window_size=args["window_size"],
                                mlp_ratio=args["mlp_ratio"], dropout=args["dropout"],
                                attention_dropout=args["attention_dropout"],
                                stochastic_depth_prob=args["stochastic_depth_prob"], num_classes=args["num_classes"])
    swin_tiny.to("cuda")
    metrics = swin_tiny.train_model(swin_tiny, train_loader, test_loader, 100, val_loader)
