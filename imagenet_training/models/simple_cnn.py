"""A simple cnn model."""


import argparse
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple CNN model.

    Args:
        data_config: a dictionary containing information about data.
        args (optional): args from argparser.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: Optional[argparse.Namespace] = None,
    ):
        super().__init__()

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        num_classes = len(data_config["mapping"])

        self.cnn = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)),
            ("relu1", nn.ReLU(inplace=True)),
            ("bn1", nn.BatchNorm2d(32)),
            ("maxpool1", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(32, 64, kernel_size=3, bias=False)),
            ("relu2", nn.ReLU(inplace=True)),
            ("bn2", nn.BatchNorm2d(64)),
            ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv3", nn.Conv2d(64, 128, kernel_size=3, bias=False)),
            ("relu3", nn.ReLU(inplace=True)),
            ("bn3", nn.BatchNorm2d(128))
        ]))
        self.head = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(128, 128)),
            ("relu1", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(128, num_classes))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward operation on a given tensor."""
        x = self.cnn(x)
        x = self.head(x)
        return x

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds possible agrs to the given parser."""
        return parser
