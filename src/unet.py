"""
This file contains the implementation of a UNet model. It is designed so that it can
be used for DDPM, but it can also be used for other tasks like segmentation.

This file also contains the building blocks for the UNet model, including the left
and right blocks, the middle conv block, and the timestep embedding.
"""

import torch
import torch.nn as nn

from .attention2D import Attention2D
from .resnet import ResBlock


class UNet(nn.Module):
    """
    UNet model for DDPM.
    """

    def __init__(
        self,
        down_filters: list[int],
        in_channels: int,
        num_layers: int,
        has_attention: list[bool] = [False, True, False],
        num_heads: int = 8,
        diffusion_steps: int = None,
        num_groups: int = 8,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.1,
    ):
        """
        Args:
            down_filters: list of downsampling filters
            in_channels: number of input channels
            num_layers: number of layers in the conv block
            has_attention: list of booleans indicating whether to use attention in each layer
            num_heads: number of attention heads
            diffusion_steps: number of diffusion steps
            num_groups: number of groups for group normalization
            activation: activation function
            dropout: dropout rate
        """
        super(UNet, self).__init__()
        self.T = diffusion_steps
        self.down_filters = down_filters
        self.up_filters = [x * 2 for x in reversed(down_filters)]
        self.num_groups = num_groups
        self.activation = activation()
        self.dropout = dropout

        self.time_embed_dim = down_filters[0] * 4
        self.num_layers = num_layers

        if self.T is not None:
            self.timestep_embedding = TimestepEmbedding(
                in_channels=self.down_filters[0],
                embedding_dim=self.time_embed_dim,
                activation=activation,
            )

        self.left_block = LeftBlock(
            filters=down_filters,
            num_layers=num_layers,
            in_channels=in_channels,
            has_attention=has_attention,
            num_heads=num_heads,
            dropout=dropout,
            timestep_emb_dim=self.time_embed_dim,
        )

        # the bottom-most (middle) conv block

        self.middle_conv = ConvBlock(
            down_filters[-1],
            down_filters[-1] * 2,
            num_layers,
            timestep_emb_dim=self.time_embed_dim,
        )
        self.middle_attention = Attention2D(
            d_k=64, dropout=0.1, num_heads=num_heads, num_channels=down_filters[-1] * 2
        )
        self.middle_upsample = nn.ConvTranspose2d(
            down_filters[-1] * 2, down_filters[-1], 2, stride=2
        )

        self.right_block = RightBlock(
            filters=self.up_filters,
            num_layers=num_layers,
            has_attention=has_attention,
            num_heads=num_heads,
            dropout=dropout,
            timestep_emb_dim=self.time_embed_dim,
        )

        first_group = num_groups if in_channels > num_groups else in_channels
        self.group_norm = nn.GroupNorm(num_groups=first_group, num_channels=in_channels)

        self.group_norm2 = nn.GroupNorm(
            num_groups=num_groups, num_channels=down_filters[0]
        )

        self.conv_out = nn.Conv2d(
            in_channels=down_filters[0],
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, t):
        """
        Forward pass of the UNet model.

        Args:
            x: input tensor
            t: timestep tensor
        """
        if self.T is not None:
            t_encoded = timestep_encoding(
                t, self.T, self.down_filters[0], n=4000, device=x.device
            )
            t_emb = self.timestep_embedding(curr_t=t_encoded, T=self.T)

            t_emb = t_emb.view(-1, self.time_embed_dim)
        else:
            t_emb = None

        h = self.group_norm(x)

        res, h = self.left_block(h, t_emb)

        h = self.middle_conv(h, t_emb)
        h = self.middle_attention(h)

        h = self.middle_upsample(h)
        h = self.right_block(h, res, t_emb)

        h = self.activation(self.group_norm2(h))
        output = self.conv_out(h)
        return output


def timestep_encoding(
    curr_t: torch.Tensor,
    T: torch.Tensor,
    embedding_dim: int,
    n=10000,
    device: torch.device = "cpu",
):
    """
    Naive sin/cosin positional embedding adapted for timestep embedding in DDPM
    """
    curr_t = curr_t / T
    p = torch.zeros((curr_t.shape[-1], embedding_dim)).to(device)

    m = torch.arange(int(embedding_dim / 2)).to(device)
    denominators = torch.pow(n, (2 * m / embedding_dim))

    p[:, 0::2] = torch.sin(curr_t.unsqueeze(1) / denominators.unsqueeze(0))
    p[:, 1::2] = torch.cos(curr_t.unsqueeze(1) / denominators.unsqueeze(0))
    return p


class TimestepEmbedding(nn.Module):
    """
    Embeds the timestep into a higher dimensional space using a 2 layer MLP.
    """

    def __init__(
        self, in_channels: int, embedding_dim: int, activation: nn.Module = nn.ReLU
    ):
        """
        Args:
            in_channels: number of input channels
            embedding_dim: dimension of the embedding space
            activation: activation function
        """
        super(TimestepEmbedding, self).__init__()
        self.linear1 = nn.Linear(in_channels, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = activation()

    def forward(self, curr_t: torch.Tensor, T: torch.Tensor):
        x = self.linear1(curr_t)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)

        return x


class LeftBlock(nn.Module):
    """
    Downampling (left) side of the UNet.
    Excludes the bottom-most conv block.
    """

    def __init__(
        self,
        in_channels: int,
        filters: list[int],
        num_layers: int,
        has_attention: list[bool] = [False, True, False],
        num_heads: int = 8,
        dropout: float = 0.2,
        timestep_emb_dim: int = None,
    ):
        super(LeftBlock, self).__init__()

        self.has_attention = has_attention
        conv_blocks = [
            ConvBlock(
                in_channels,
                filters[0],
                num_layers,
                timestep_emb_dim=timestep_emb_dim,
                dropout=dropout,
            )
        ]
        attention_blocks = (
            [
                Attention2D(
                    d_k=64,
                    dropout=dropout,
                    num_heads=num_heads,
                    num_channels=filters[0],
                )
            ]
            if has_attention
            else []
        )

        for i in range(1, len(filters)):
            conv_blocks.append(
                ConvBlock(
                    filters[i - 1],
                    filters[i],
                    num_layers,
                    timestep_emb_dim=timestep_emb_dim,
                    dropout=dropout,
                )
            )
            if has_attention[i]:
                attention_blocks.append(
                    Attention2D(
                        d_k=64,
                        dropout=dropout,
                        num_heads=num_heads,
                        num_channels=filters[i],
                    )
                )
            else:
                attention_blocks.append(None)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, timestep_emb=None):
        residual_outputs = []
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x, timestep_emb)
            if self.has_attention[i]:
                x = self.attention_blocks[i](x)

            residual_outputs.append(x)
            x = self.maxpool(x)

        return residual_outputs, x


class RightBlock(nn.Module):
    """
    Upsampling (right) side of the UNet.
    """

    def __init__(
        self,
        filters: list[int],
        num_layers: int,
        has_attention: list[bool] = [False, True, False],
        num_heads: int = 8,
        dropout: float = 0.2,
        timestep_emb_dim: int = None,
    ):
        super(RightBlock, self).__init__()
        self.has_attention = has_attention

        conv_layers = []
        upsample_layers = []
        attention_layers = []

        for i in range(len(filters) - 1):
            conv_layers.append(
                ConvBlock(
                    filters[i],
                    filters[i + 1],
                    num_layers,
                    timestep_emb_dim=timestep_emb_dim,
                    dropout=dropout,
                )
            )
            upsample_layers.append(
                nn.ConvTranspose2d(filters[i + 1], filters[i + 1] // 2, 2, stride=2)
            )

            if has_attention[i]:
                attention_layers.append(
                    Attention2D(
                        d_k=64,
                        dropout=0.1,
                        num_heads=num_heads,
                        num_channels=filters[i + 1] // 2,
                    )
                )
            else:
                attention_layers.append(None)

        conv_layers.append(
            ConvBlock(
                filters[-1],
                filters[-1] // 2,
                num_layers,
                timestep_emb_dim=timestep_emb_dim,
                dropout=dropout,
            )
        )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.attention_layers = nn.ModuleList(attention_layers)
        self.upsample_layers = nn.ModuleList(upsample_layers)

    def forward(self, x, residual_outputs, timestep_emb=None):
        for i in range(len(self.conv_layers)):
            residual = residual_outputs[-(i + 1)]
            _, _, h, w = x.shape
            residual = residual[:, :, :h, :w]

            x = torch.cat([x, residual], dim=1)
            x = self.conv_layers[i](x, timestep_emb)

            if i < len(self.upsample_layers):
                x = self.upsample_layers[i](x)
                if self.has_attention[i]:
                    x = self.attention_layers[i](x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block for the UNet.
    This is the basic building block of the UNet.
    It is a sequence of residual blocks that may accept a timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        num_groups: int = 1,
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU,
        timestep_emb_dim: int = None,
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            num_layers: number of layers in the conv block
            num_groups: number of groups for group normalization
            dropout: dropout rate
            timestep_emb_dim: dimension of the timestep embedding
        """
        super(ConvBlock, self).__init__()
        convs = []
        convs.append(
            ResBlock(
                in_channels,
                out_channels,
                num_groups=num_groups,
                dropout=dropout,
                activation=activation,
                timestep_emb_dim=timestep_emb_dim,
            )
        )

        for _ in range(num_layers - 1):
            convs.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                    timestep_emb_dim=timestep_emb_dim,
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x, timestep_emb=None):
        for res_block in self.convs:
            x = res_block(x, timestep_emb)

        return x
