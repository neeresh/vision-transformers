from functools import partial
from typing import Callable

import torch
from torch import nn

from abc import abstractmethod


class BaseTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def train_model(self, model, epochs, train_loader, test_loader, val_loader=None):
        pass
