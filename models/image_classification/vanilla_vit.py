import math
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.image_classification.base import BaseTransformer

from typing import Callable, List, Optional
from functools import partial

from torch.nn import Unfold


class MLP(torch.nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: List[int],
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, bias: bool = True, dropout: float = 0.0):

        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
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

        # Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

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
    def __init__(self, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, images: torch.Tensor):
        images = self.dropout(images)
        images = self.layers(images)
        images = self.ln(images)

        return images


class ViT(BaseTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 num_classes, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.patch_embedding = nn.Linear(self.patch_size * self.patch_size * 3, self.hidden_dim)
        self.positional_encoding = nn.Parameter(torch.empty(1, self.num_patches, self.hidden_dim).normal_(std=0.02))
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.device = 'cpu'

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
                               dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)
        self._initialize_weights()  # For classification head (output layer)

    def _initialize_weights(self): # Initialize the weights of the classification head
        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_patches(self, images):
        """
        Divides an image into patches
        """
        self.batch_size, channels, _, _ = images.shape
        images = Unfold(kernel_size=self.patch_size, stride=self.patch_size)(images)
        images = images.view(self.batch_size, channels, self.patch_size, self.patch_size, -1)
        images = images.permute(0, 4, 1, 2, 3)  # (batch_size, num_patches, channels, patch_size, patch_size)

        return images

    def patch_embedding_position_encoding(self, patches):
        # Flatten the patches
        patches = patches.view(self.batch_size, self.num_patches,
                               -1)  # (batch_size, num_patches, patch_size*patch_size*3)
        patch_embeddings = self.patch_embedding(patches)  # (batch_size, num_patches, hidden_dim)

        # Sequence Length = number of patches + class token
        embeddings = patch_embeddings + self.positional_encoding

        # Adding class token
        class_token = self.class_token.expand(self.batch_size, -1, -1)
        embeddings = torch.cat([class_token, embeddings], dim=1)

        return embeddings

    def forward(self, images):
        patches = self.get_patches(images)
        embeddings = self.patch_embedding_position_encoding(patches)
        encoder_output = self.encoder(embeddings)

        # Classifier "token" as used by standard language architectures
        class_token_output = encoder_output[:, 0]
        output = self.heads(class_token_output)

        return patches, embeddings, encoder_output, class_token_output, output

    def train_model(self, model, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
                    val_loader: Optional[DataLoader] = None):

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # To store metrics
        train_losses = []
        val_losses = []
        test_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training phase
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for images, labels in train_loader_tqdm:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                _, _, _, _, outputs = model(images)
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
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        _, _, _, _, outputs = model(images)
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
                epoch_val_loss = None
                epoch_val_accuracy = None

            # Testing phase
            model.eval()
            running_test_loss = 0.0
            correct_test = 0
            total_test = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, _, _, _, outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(test_loader.dataset)
            epoch_test_accuracy = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_accuracy)

            # Update tqdm description for the entire epoch
            train_loader_tqdm.set_postfix({
                "Train Loss": epoch_train_loss,
                "Train Acc": epoch_train_accuracy,
                "Val Loss": epoch_val_loss if val_loader else "N/A",
                "Val Acc": epoch_val_accuracy if val_loader else "N/A",
                "Test Loss": epoch_test_loss,
                "Test Acc": epoch_test_accuracy
            })

        return {
            "train_loss": train_losses,
            "val_loss": val_losses if val_loader else None,
            "test_loss": test_losses,
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies if val_loader else None,
            "test_accuracy": test_accuracies
        }

















