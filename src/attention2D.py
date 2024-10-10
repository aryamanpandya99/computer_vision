"""
This module contains the building blocks for a 2D attention mechanism.
Author: Aryaman Pandya
"""

import math

import torch
import torch.nn as nn


class Attention2D(nn.Module):
    """
    Multihead attention.
    """

    def __init__(
        self,
        d_k: int,
        dropout: float,
        num_heads: int,
        num_channels: int,
        num_groups: int = 8,
        mask: bool = False,
    ):
        """
        Args:
            d_k: dimension of the key
            dropout: dropout rate
            num_heads: number of heads
            num_channels: number of channels
            num_groups: number of groups for group normalization
            mask: whether to a mask
        """
        super(Attention2D, self).__init__()
        self.d_k, self.num_heads = d_k, num_heads
        self.query_projection, self.key_projection, self.value_projection = (
            nn.Linear(num_channels, num_heads * d_k),
            nn.Linear(num_channels, num_heads * d_k),
            nn.Linear(num_channels, num_heads * d_k),
        )
        self.layer_norm = nn.LayerNorm(num_channels)
        self.output_layer = nn.Linear(num_heads * d_k, num_channels)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask
        self.num_channels = num_channels

    def forward(self, x, y=None):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, height * width).permute(0, 2, 1)
        residual = x

        x = self.layer_norm(x)
        if y is not None:
            k, q, v = y, x, y
        else:
            k, q, v = x, x, x

        k_len, q_len, v_len, batch_size = k.size(1), q.size(1), v.size(1), q.size(0)

        k = self.key_projection(k).view(batch_size, k_len, self.num_heads, self.d_k)
        q = self.query_projection(q).view(batch_size, q_len, self.num_heads, self.d_k)
        v = self.value_projection(v).view(batch_size, v_len, self.num_heads, self.d_k)

        attention = scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), self.d_k, self.mask
        )
        output = torch.matmul(attention, v.transpose(1, 2))
        output = self.output_layer(
            output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        )

        return self.dropout(output) + residual


def scaled_dot_product_attention(q, k, d_k, mask):
    """
    Scaled dot product attention.
    Args:
        q: query
        k: key
        d_k: dimension of the key
        mask: whether to use mask
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is True:
        mask = torch.tril(torch.ones(scores.shape)).to(q.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return nn.Softmax(-1)(scores)
