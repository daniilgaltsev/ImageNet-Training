"""Module with resnet like models."""


import argparse
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .blocks import ResNetBlock, ResNetBottleneck, conv1x1


TYPE_TO_ARGS = {
    "resnet18": ([2, 2, 2, 2], ResNetBlock),
    "resnet34": ([3, 4, 6, 3], ResNetBlock),
    "resnet50": ([3, 4, 6, 3], ResNetBottleneck),
    "resnet101": ([3, 4, 23, 3], ResNetBottleneck),
    "resnet152": ([3, 8, 36, 3], ResNetBottleneck),
}


class ResNet(nn.Module):
    """A convolutional resnet-like model.

    Args:
        data_config: a dictionary containing information about data.
        args (optional): args from argparser.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: Optional[argparse.Namespace] = None
    ):
        super().__init__()

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        input_dims = data_config["input_dims"]
        if input_dims[1] < 32 or input_dims[2] < 32:
            raise ValueError("Minimum input width and height for ResNet is 32 (3x32x32).")
        num_classes = len(data_config["mapping"])
        model_type = self.args.get("model_type", None)
        if model_type is None or model_type not in TYPE_TO_ARGS:
            raise ValueError("Type of resnet model is not specified or incorrect (resnet{18, 34, 50, 101, 152}).")

        num_blocks, block = TYPE_TO_ARGS[model_type]
        self.model = _ResNet(num_blocks, block, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tensor of logits for each class."""
        return self.model(x)

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds possible args to the given parser."""
        parser.add_argument(
            "--model_type", type=str, default="resnet18",
            help="Type of resnet to use (resnet{18, 34, 50, 101, 152})."
        )
        return parser


class _ResNet(nn.Module):
    """A convolutional resnet-like model.

    Args:
        num_blocks: a list of number of blocks in each resnet layer.
        block: a class constructor to use for creating resnet blocks.
        num_classes: a number of classes.
    """

    _base_channels: int = 64

    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int],
        block: Callable[..., Union[ResNetBlock, ResNetBottleneck]],
        num_classes: int
    ):
        super().__init__()

        self.block = block
        self._last_channels = self._base_channels
        self.model = nn.Sequential(OrderedDict([
            ("norm1", nn.BatchNorm2d(3)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", nn.Conv2d(kernel_size=7, in_channels=3, out_channels=self._base_channels, stride=2, padding=3)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer1", self._make_layer(self._base_channels, num_blocks[0], stride=1)),
            ("layer2", self._make_layer(self._base_channels * 2, num_blocks[1], stride=2)),
            ("layer3", self._make_layer(self._base_channels * 4, num_blocks[2], stride=2)),
            ("layer4", self._make_layer(self._base_channels * 8, num_blocks[3], stride=2)),
            ("relu3", nn.ReLU(inplace=True)),
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten()),
            ("fc", nn.Linear(self._base_channels * 8 * self.block.expansion, num_classes))
        ]))

    def _make_layer(
        self,
        channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Module:
        blocks = []

        out_channels = channels * self.block.expansion
        if self._last_channels == out_channels and stride == 1:
            downsample = None
        else:
            downsample = _downsample(self._last_channels, out_channels, stride=stride)

        blocks.append((
            "block1",
            self.block(self._last_channels, channels, stride=stride, downsample=downsample)
        ))
        self._last_channels = out_channels
        for i in range(1, num_blocks):
            blocks.append((
                f"block{i + 1}",
                self.block(self._last_channels, channels, stride=1, downsample=None)
            ))

        layer = nn.Sequential(OrderedDict(blocks))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tensor of logits for each class."""
        return self.model(x)


def _downsample(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    return nn.Sequential(OrderedDict([
        ("conv", conv1x1(in_channels, out_channels, stride)),
        ("norm", nn.BatchNorm2d(out_channels))
    ]))
