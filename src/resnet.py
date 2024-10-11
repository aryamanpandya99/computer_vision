"""
This module contains the building blocks for a ResNet model.
Author: Aryaman Pandya
"""

import torch.nn as nn


class ResBlock(nn.Module):
    """
    A residual block with optional timestep embedding handling.
    Can be used as a building block for a ResNet or a UNet used
    for diffusion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.ReLU,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.2,
        num_groups: int = 1,
        timestep_emb_dim: int = None,
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            activation: activation function
            kernel_size: convolutional kernel size
            stride: convolutional stride
            padding: padding
            dropout: dropout rate
            num_groups: number of groups for group normalization
            timestep_emb_dim: number of dimensions in the timestep embedding
        """
        super().__init__()
        self.norm1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels), activation()
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        )
        self.activation = activation()

        if in_channels == out_channels:
            self.idconv = nn.Identity()
        else:
            self.idconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.avgpool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=2, stride=stride)
        )
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.timestep_emb_dim = timestep_emb_dim

        if self.timestep_emb_dim is not None:
            self.timestep_emb_proj = nn.Linear(timestep_emb_dim, out_channels)

    def forward(self, x, timestep_emb=None):
        """
        Forward pass of the residual block.
        Args:
            x: input tensor
            timestep_emb: optional timestep embedding
        """
        h = self.norm1(x)
        h = self.conv1(h)

        if self.timestep_emb_dim is not None:
            timestep_emb = self.timestep_emb_proj(timestep_emb)[:, :, None, None] # Enables broadcasting
            h += self.activation(timestep_emb)

        h = self.conv2(h)
        h_id = self.avgpool(self.idconv(x))

        return self.dropout(self.activation(h + h_id))


class ResNet(nn.Module):
    """
    A simple ResNet model with a series of residual blocks.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        filters: list[int],
        activation: nn.Module = nn.ReLU,
        stride: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_channels: number of input channels
            num_classes: number of output classes
            filters: list of filter sizes
            activation: activation function
            stride: convolutional stride
            dropout: dropout rate
        """
        super().__init__()
        res_layers = [ResBlock(num_channels, filters[0], activation, stride=1)]
        for i in range(len(filters) - 1):
            res_layers += [
                ResBlock(
                    filters[i],
                    filters[i + 1],
                    activation,
                    stride=stride,
                    dropout=dropout,
                )
            ]
        self.res_layers = nn.Sequential(*res_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(filters[-1], num_classes, bias=False),
            nn.BatchNorm1d(num_classes),
            activation(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for _, layer in enumerate(self.res_layers):
            x = layer(x)

        h = self.avgpool(x)
        h = self.flatten(h)
        h = self.dropout(h)
        return self.linear(h)
