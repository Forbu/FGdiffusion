"""
Utility modules for graph generation.

Provides MLP, TransformerConvEdge (modified for edge feature retrieval),
and helper functions used across trainers.
"""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.typing import OptTensor
from torch import Tensor
from torch_geometric.utils import softmax


class MLP(nn.Module):
    """Multi-layer perceptron with optional normalization."""

    def __init__(
        self, in_dim, out_dim=128, hidden_dim=128, hidden_layers=2, norm_type=None
    ):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TransformerConvEdge(TransformerConv):
    """
    Modified TransformerConv that saves edge-level output features.
    Used in GraphGDP for edge-level score prediction.
    """

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr
        out = out * alpha.view(-1, self.heads, 1)

        self.out = out
        return out


def init_weights(m):
    """Xavier initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def stratified_uniform_sample(batch_size, device=None, dtype=torch.float32):
    """
    Generate stratified uniform samples in [0, 1).

    Divides [0, 1] into batch_size strata and draws one sample per stratum.
    Used for time-step sampling in discrete diffusion training.
    """
    if device is None:
        device = torch.device("cpu")

    t = torch.arange(batch_size, device=device, dtype=dtype)
    offsets = torch.rand(batch_size, device=device, dtype=dtype)
    return (t + offsets) / batch_size
