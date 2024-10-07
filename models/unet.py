import torch
import torch.nn as nn

from .resnet import ResBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        self.conv1 = ResBlock(in_channels, out_channels)

        for i in range(num_layers - 1):
            self.conv1.append(ResBlock(out_channels, out_channels))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    """
    Downampling (left) side of the UNet.
    Excludes the bottom-most conv block.
    """

    def __init__(self, filters, in_channels, num_layers):
        super(DownBlock, self).__init__()
        res_blocks = [ResBlock(in_channels, filters[0])]
        for i in range(1, len(filters)):
            res_blocks.append(ResBlock(filters[i - 1], filters[i]))

        self.res_blocks = nn.Sequential(*res_blocks)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual_outputs = []
        for res_block in self.res_blocks:
            x = res_block(x)
            residual_outputs.append(x)
            x = self.maxpool(x)

        return residual_outputs, x


class UpBlock(nn.Module):
    """
    Upsampling (right) side of the UNet.
    """

    def __init__(self, filters):
        super(UpBlock, self).__init__()
        layers = []
        for i in range(len(filters) - 2):
            layers.append(
                nn.Sequential(
                    ResBlock(filters[i], filters[i + 1]),
                    nn.ConvTranspose2d(
                        filters[i + 1], filters[i + 1] // 2, 2, stride=2
                    ),
                )
            )

        layers.append(ResBlock(filters[-2], filters[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual_outputs):
        for i in range(len(self.layers)):
            print(f"i: {i}")
            residual = residual_outputs[-(i + 1)]
            _, _, h, w = x.shape
            residual = residual[:, :, :h, :w]
            print(f"x: {x.shape}, residual: {residual.shape}")
            x = torch.cat([x, residual], dim=1)
            x = self.layers[i](x)
            print(f"x: {x.shape}")

        return x


class UNet(nn.Module):
    def __init__(self, down_filters, in_channels):
        super(UNet, self).__init__()
        self.down_filters = down_filters
        self.down_block = DownBlock(down_filters, in_channels)

        # the bottom-most conv block is different from the rest of the blocks
        # in that it doesn't contain a maxpool and upsamples without a residual connection
        self.bottom_conv = nn.Sequential(
            ResBlock(down_filters[-1], down_filters[-1] * 2),
            nn.ConvTranspose2d(down_filters[-1] * 2, down_filters[-1], 2, stride=2),
        )

        self.up_filters = [down_filters[-1] * 2]
        self.up_filters.extend(reversed(down_filters))
        self.up_block = UpBlock(self.up_filters)

    def forward(self, x):
        residual_outputs, down_output = self.down_block(x)
        bottom_output = self.bottom_conv(down_output)
        return self.up_block(bottom_output, residual_outputs)
