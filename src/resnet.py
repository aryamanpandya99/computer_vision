import torch.nn as nn


class ResBlock(nn.Module):
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
        super().__init__()
        self.norm1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            activation()
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
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
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
    
    
    def forward(self, x, timestep_emb=None):
        h = self.conv(x)
        ## TODO: add timestep embedding handling 
        h_id = self.avgpool(self.idconv(x))
        return self.dropout(self.activation(h + h_id))


class ResNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        filters: list[int],
        activation: nn.Module = nn.ReLU,
        stride: int = 2,
        dropout: float = 0.2,
    ):
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
