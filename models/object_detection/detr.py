from typing import List

import torch
import torchvision
from torch import nn
from torch.nested._internal.nested_tensor import NestedTensor
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

import torch.nn.functional as F

from utils.coco.util.misc import nested_tensor_from_tensor_list


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)

            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class AbsolutePositionalEncoding(nn.Module):

    """
    Adding positional information to each pixel.
    Pixel is located by row and col indicies.
    """

    def __init__(self, positional_features=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, positional_features)
        self.col_embed = nn.Embedding(50, positional_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: NestedTensor):
        x = x.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat(  # 1 means keeps the dimensions same
            [x_emb.unsqueeze(0).repeat(h, 1, 1),  # (h, 1, 1) repeats h times, (h, 1, pos_feat)
                    y_emb.unsqueeze(1).repeat(1, w, 1)],  # (1, w, 1) repeats "w" times, (1, w, pos_feat)
                    dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # (h, w, 2*pos_features)

        return pos


def build_backbone(trainable_backbone: bool):

    model = torchvision.models.resnet50(
        weights=ResNet50_Weights.DEFAULT,
        replace_stride_with_dilation=[False, False, True],
        norm_layer=FrozenBatchNorm2d)

    # Freeze the backbone or not
    train_backbone = trainable_backbone
    if not train_backbone:
        # Freeze layers if not training backbone
        for name, parameter in model.named_parameters():
            if 'layer2' in name or 'layer3' in name or 'layer4' in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

    # Return intermediate layers based on arguments
    return_interm_layers = True

    # Construct the backbone model with IntermediateLayerGetter if required
    model = IntermediateLayerGetter(model, return_layers={'layer4': '0'}) if not return_interm_layers else \
        IntermediateLayerGetter(model, return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"})

    num_channels = 2048  # ResNet-50 has 2048 channels in its last feature map (layer4)
    return model, num_channels


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class Detr(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        :param backbone: Backbone to be used before encoder
        :param transformer: Transformer
        :param num_classes: Number of object classes
        :param num_queries: Maximum number of objects DETR can detect
        :param aux_loss: Auxillary decoding losses are to be used
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.dmodel
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.in_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        print(features.shape, pos.shape)



def set_model_and_positional_embeddings():
    hidden_dim = 512
    N_steps = hidden_dim // 2

    positional_embedding = AbsolutePositionalEncoding(N_steps)
    backbone, num_channels = build_backbone(trainable_backbone=False)

    # Positional Embedding and Backbone
    model = Joiner(backbone, positional_embedding)

    model.num_channels = num_channels

    return model


if __name__ == '__main__':
    backbone = set_model_and_positional_embeddings()
    transformer =

