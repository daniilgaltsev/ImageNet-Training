"""Module with resnet like models."""


import argparse
from collections import OrderedDict
import importlib
from typing import Any, Dict, List, Optional, Type, Union

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
USE_TORCHVISION_MODEL = False


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
        if len(input_dims) != 4:
            raise ValueError(f"Expected input_dims to have 4 dimensions got {len(input_dims)} ({input_dims})")
        if input_dims[2] < 32 or input_dims[3] < 32:
            raise ValueError(f"Minimum input width and height for ResNet is 32 (3x32x32) got {input_dims}.")
        num_classes = len(data_config["mapping"])
        resnet_type = self.args.get("resnet_type", None)
        if resnet_type is None or resnet_type not in TYPE_TO_ARGS:
            raise ValueError("Type of resnet model is not specified or incorrect (resnet{18, 34, 50, 101, 152}).")
        use_torchvision = self.args.get("use_torchvision_model", USE_TORCHVISION_MODEL)

        if use_torchvision:
            tv_models_module = importlib.import_module("torchvision.models")
            self.model = getattr(tv_models_module, resnet_type)()
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            num_blocks, block = TYPE_TO_ARGS[resnet_type]
            self.model = _ResNet(num_blocks, block, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tensor of logits for each class."""
        return self.model(x)

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser  # pylint: disable=unused-argument
    ) -> argparse.ArgumentParser:
        """Adds possible args to the given parser."""
        parser.add_argument(
            "--resnet_type", type=str, default="resnet18",
            help="Type of resnet to use (resnet{18, 34, 50, 101, 152})."
        )
        parser.add_argument(
            "--use_torchvision_model", default=False, action="store_true",
            help="If true, will use resnet architecture from torchvision."
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
        num_blocks: List[int],
        block: Type[Union[ResNetBlock, ResNetBottleneck]],
        num_classes: int
    ):
        super().__init__()

        if len(num_blocks) != 4:
            raise ValueError(f"Incorrect number of blocks. Should be 4 got {len(num_blocks)} ({num_blocks}).")
        self.block = block
        self._last_channels = self._base_channels
        self.model = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(kernel_size=7, in_channels=3, out_channels=self._base_channels, stride=2, padding=3)),
            ("bn1", nn.BatchNorm2d(self._base_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("layer1", self._make_layer(self._base_channels, num_blocks[0], stride=1)),
            ("layer2", self._make_layer(self._base_channels * 2, num_blocks[1], stride=2)),
            ("layer3", self._make_layer(self._base_channels * 4, num_blocks[2], stride=2)),
            ("layer4", self._make_layer(self._base_channels * 8, num_blocks[3], stride=2)),
            ("bn2", nn.BatchNorm2d(self._base_channels * 8 * self.block.expansion)),
            ("relu2", nn.ReLU(inplace=True)),
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
