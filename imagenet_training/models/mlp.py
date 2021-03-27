"""A simple MLP model."""


import argparse
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


FC1_DIM = 512
FC2_DIM = 512


class MLP(nn.Module):
    """A simple MLP model.

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

        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])
        fc1_dim = self.args.get("fc1", FC1_DIM)
        fc2_dim = self.args.get("fc2", FC2_DIM)

        self.mlp = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(input_dim, fc1_dim)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.5)),
            ("fc2", nn.Linear(fc1_dim, fc2_dim)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.5)),
            ("fc3", nn.Linear(fc2_dim, num_classes)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward operation on a given tensor."""
        x = self.mlp(x)
        return x

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser  # pylint: disable=unused-argument
    ) -> argparse.ArgumentParser:
        """Adds possible agrs to the given parser."""
        parser.add_argument("--fc1", type=int, default=FC1_DIM, help="Size of the first hidden MLP layer.")
        parser.add_argument("--fc2", type=int, default=FC2_DIM, help="Size of the second hidden MLP layer.")
        return parser
