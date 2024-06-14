import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.image_classification.base import BaseTransformer

from typing import Callable, List, Optional
from functools import partial

from models.image_classification.token_performer import TokenPerformer
from models.image_classification.token_transformer import TokenTransformer
from utils.args import get_args
from utils.load_data import get_train_test_loaders

"""
References: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
"""


class MLP(torch.nn.Sequential):

    def __init__(self, in_channels: int, hidden_channels: List[int],
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, bias: bool = True, dropout: float = 0.0):

        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):

    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):

    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float,
                 attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                                                        norm_layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding

        return self.ln(self.layers(self.dropout(input)))


class T2T(nn.Module):
    def __init__(self, image_size, tokens_type, in_channels, embed_dim, token_dim):
        super().__init__()

        if tokens_type == 'transformer':
            print("Adopting transformer encoder for token-to-token")
            self.softsplit0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.softsplit1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.softsplit2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = TokenTransformer(dim = in_channels * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = TokenTransformer(dim = token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print("Adopting performer encoder for token-to-token")
            self.softsplit0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.softsplit1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.softsplit2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = TokenPerformer(dim=in_channels * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = TokenPerformer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':
            print("Adopting transformer encoder for token-to-token")
            self.softsplit0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(1, 1)) # 1st conv
            self.softsplit1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 2nd conv
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # the 3rd convolution

        self.num_patches = (image_size // (4 * 2 * 2)) * (image_size // (4 * 2 * 2))  # there are 3 soft split, stride are 4,2,2 seperately

    def forward(self, x):

        # Step 0
        x = self.softsplit0(x).transpose(1, 2)
        # iter 1 (re-sconstructions / re-structurizations)
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # iter1: softsplit
        x = self.softsplit1(x).transpose(1, 2)

        # iter2: (re-sconstructions / re-structurizations)
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.tranpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # iter2: softsplit
        x = self.softsplit2(x).transpose(1, 2)

        # Final tokens
        x = self.project(x)

        return x


class T2T_ViT(BaseTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 num_classes, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 token_dim=64, token_type="performer", *args, **kwargs):

        super().__init__(*args, **kwargs)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.num_patches = (image_size // patch_size) ** 2
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.t2t = T2T(image_size=image_size, tokens_type='performer', in_channels=3, embed_dim=hidden_dim,
                       token_dim=token_dim)
        self.num_patches = self.t2t.num_patches

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        seq_length = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(seq_length=seq_length, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim,
                               mlp_dim=mlp_dim,
                               dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward_features(self, images: torch.Tensor):
        n, c, h, w = images.shape
        p = self.patch_size

        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")

        n_h = h // p
        n_w = w // p

        images = self.conv_proj(images)
        images = images.reshape(n, self.hidden_dim, n_h * n_w)
        images = images.permute(0, 2, 1)

        n = images.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        images = torch.cat([batch_class_token, images], dim=1)

        images = self.encoder(images)

        return images

    def forward(self, images: torch.Tensor):

        images = self.forward_features(images)
        images = images[:, 0]
        images = self.heads(images)

        return images

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
    # Epoch 100/100 - Train Loss: 1.1087, Train Acc: 0.6708,
    # Val Loss: 2.3863208625793457, Val Acc: 0.4553,
    # Test Loss: 2.3039, Test Acc: 0.4626
    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256,
                                                                   val_split=0.2, num_workers=4)
    args = get_args("vit_tiny_cifar100")
    t2t_vit = T2T_ViT(image_size=args["image_size"], patch_size=args["patch_size"], num_layers=args["num_layers"],
              num_heads=args["num_heads"], hidden_dim=args["hidden_dim"], mlp_dim=args["mlp_dim"],
              dropout=args["dropout"], attention_dropout=args["attention_dropout"],
              num_classes=args["num_classes"])
    t2t_vit.to("mps")
    # print(t2t_vit)
    metrics = t2t_vit.train_model(t2t_vit, train_loader, test_loader, 100, val_loader)
