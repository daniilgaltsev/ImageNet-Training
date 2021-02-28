"""Blocks used in ResNet."""


from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """A basic block for ResNet.

    Args:
        in_channels: number of input channels.
        channels: base number of channels.
        stride (optional): stride size to use in the first conv layer (downsampling in main path).
        downsample (optional): downsampling module for residual connection if
          shape of input is different from output.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.pre_residual = nn.Sequential(OrderedDict([
            ("bn1", nn.BatchNorm2d(in_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", conv3x3(in_channels, channels, stride=stride)),
            ("bn2", nn.BatchNorm2d(channels)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv2", conv3x3(channels, channels)),
        ]))
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward operation with a residual connection."""
        inp = x
        x = self.pre_residual(x)
        if self.downsample is not None:
            inp = self.downsample(inp)
        x += inp
        return x


class ResNetBottleneck(nn.Module):
    """A bottleneck block for ResNet.

    Args:
        in_channels: number of input channels.
        channels: base number of channels.
        stride (optional): stride size to use in the first conv layer (downsampling in main path).
        downsample (optional): downsampling module for residual connection if
          shape of input is different from output.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.pre_residual = nn.Sequential(OrderedDict([
            ("bn1", nn.BatchNorm2d(in_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", conv1x1(in_channels, channels, stride=stride)),
            ("bn2", nn.BatchNorm2d(channels)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv2", conv3x3(channels, channels, stride=1)),
            ("bn3", nn.BatchNorm2d(channels)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv3", conv1x1(channels, channels * 4, stride=1))
        ]))
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward operation with a residual connection."""
        inp = x
        x = self.pre_residual(x)
        if self.downsample is not None:
            inp = self.downsample(inp)
        x += inp
        return x


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution for resnet block."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 2) -> nn.Conv2d:
    """1x1 convolution for resnet block downsampling."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
