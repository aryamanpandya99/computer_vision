import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, d_k, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is True:
        mask = torch.tril(torch.ones(scores.shape)).to(q.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return nn.Softmax(-1)(scores)


class Attention(nn.Module):
    """
    Multihead attention class implementation. Can act as self-attention (default, y is None)
    or cross attention if y not none
    """

    def __init__(self, d_k, d_model, d_v, dropout, num_heads, mask) -> None:
        super(Attention, self).__init__()
        self.d_k, self.d_v, self.d_model, self.num_heads = d_k, d_v, d_model, num_heads
        self.query_layer, self.key_layer, self.value_layer = (
            nn.Linear(d_model, num_heads * d_k),
            nn.Linear(d_model, num_heads * d_k),
            nn.Linear(d_model, num_heads * d_v),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.concat_projection = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def forward(self, x, y=None):
        residual = x
        x = self.layer_norm(x)
        if y is not None:
            k, q, v = y, x, y
        else:
            k, q, v = x, x, x

        k_len, q_len, v_len, batch_size = k.size(1), q.size(1), v.size(1), q.size(0)
        k = self.key_layer(k).view(batch_size, k_len, self.num_heads, self.d_k)
        q = self.query_layer(q).view(batch_size, q_len, self.num_heads, self.d_k)
        v = self.value_layer(v).view(batch_size, v_len, self.num_heads, self.d_v)
        attention = scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), self.d_k, self.mask
        )
        output = torch.matmul(attention, v.transpose(1, 2))
        output = self.concat_projection(
            output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        )

        return self.dropout(output) + residual
