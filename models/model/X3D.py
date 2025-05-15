import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=False, use_relu=False):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class X3D_Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, temporal_kernel_size=3):
        super(X3D_Bottleneck, self).__init__()
        mid_channels = out_channels // expansion

        self.conv1 = Conv3DBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv3DBlock(
            mid_channels, mid_channels,
            kernel_size=(temporal_kernel_size, 3, 3),
            stride=(stride, stride, stride),
            padding=(temporal_kernel_size // 2, 1, 1)
        )
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(stride, stride, stride), bias=False),
            nn.BatchNorm3d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        return self.relu(x)


class X3D(nn.Module):
    def __init__(self, num_classes=400, layers=(3, 3, 3, 3,3), block=X3D_Bottleneck, channels=(64, 128, 256, 512, 4096), expansion=4):
        super(X3D, self).__init__()
        self.stem = Conv3DBlock(2, channels[0], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1))

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block, channels[i], channels[i+1] if i+1 < len(channels) else channels[i]*expansion, num_blocks, stride)
            self.layers.append(layer)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels[-1] * expansion, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
