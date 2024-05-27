
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.nn import Unfold


class BaseTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 num_classes, representation_size,
                 norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6), *args, **kwargs):
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
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.patch_embedding = nn.Linear(self.patch_size * self.patch_size * 3, self.hidden_dim)
        self.positional_encoding = nn.Parameter(torch.empty(1, self.num_patches, self.hidden_dim).normal_(std=0.02))
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.device = 'mps'

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
