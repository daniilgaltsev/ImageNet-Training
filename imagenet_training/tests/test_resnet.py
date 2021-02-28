"""Tests for resnet module."""

from typing import Tuple

import pytest
import torch

from imagenet_training.models.resnet import ResNet


@pytest.mark.parametrize("model_type, input_shape", [
    ("resnet18", (2, 3, 32, 32)),
    ("resnet18", (3, 3, 37, 43)),
    ("resnet18", (2, 3, 224, 224)),
    ("resnet34", (2, 3, 32, 32)),
    ("resnet50", (2, 3, 32, 32)),
    ("resnet50", (3, 3, 37, 43)),
    ("resnet101", (2, 3, 32, 32)),
    ("resnet152", (2, 3, 32, 32))
])
def test_resnet_shapes(model_type: str, input_shape: Tuple[int, int, int, int]):
    """ Test that scripted LightningModule forward works. """
    class Args():
        pass
    args = Args()
    args.model_type = model_type
    config = {
        "input_dims": input_shape[1:],
        "mapping": [1] * (10 + 3 * input_shape[-1]),
    }

    inp = torch.randn(input_shape)
    model = ResNet(config, args)
    output = model(inp)
    assert output.shape == (input_shape[0], len(config["mapping"]))
