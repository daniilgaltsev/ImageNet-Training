"""A module for running experiments in a unifed way."""

import argparse
import importlib
import random

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from imagenet_training import lit_models

DATA_CLASS_TEMPLATE = "imagenet_training.data.{}"
MODEL_CLASS_TEMPLATE = "imagenet_training.models.{}"
SEED = 0


def _import_class(module_and_class_name: str) -> type:
    """Imports class from the module.

    Args:
        module_and_class_name: A string containing module and class name ("<module>.<class>").
    """
    module_name, class_name = module_and_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser() -> argparse.ArgumentParser:
    """Setups ArgumentParser with all needed args (from data, model, trainer, etc.)."""
    parser = argparse.ArgumentParser(add_help=False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--data_class", type=str, default="CIFAR10", help="Name of data module to use.")
    parser.add_argument("--model_class", type=str, default="MLP", help="Name of model module to use.")

    parser.add_argument("--use_wandb", default=False, action="store_true", help="If True, will use wandb.")
    parser.add_argument("--group", type=str, default="", help="Experiment group to log in wandb.")

    parser.add_argument("--patience", type=int, default=10, help="Patience for Early Stopping.")
    parser.add_argument(
        "--use_lr_monitor", default=False, action="store_true", help="If True. will use LRMonitor callback."
    )

    parser.add_argument("--seed", type=int, default=SEED)

    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(DATA_CLASS_TEMPLATE.format(temp_args.data_class))
    model_class = _import_class(MODEL_CLASS_TEMPLATE.format(temp_args.model_class))

    data_group = parser.add_argument_group(f"Data Args for {temp_args.data_class}")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group(f"Model Args for {temp_args.model_class}")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser


def _set_seeds(seed: int) -> None:
    """Sets seed for RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def main() -> None:
    """Runs an experiment with specified args."""
    parser = _setup_parser()
    args = parser.parse_args()
    print("Running an experiment with specified args:")
    print(args)

    _set_seeds(args.seed)

    data_class = _import_class(DATA_CLASS_TEMPLATE.format(args.data_class))
    model_class = _import_class(MODEL_CLASS_TEMPLATE.format(args.model_class))
    data = data_class(args=args)
    model = model_class(data_config=data.config(), args=args)

    lit_model = lit_models.BaseLitModel(model, args=args)

    loggers = []
    if args.use_wandb:
        wandb_logger = WandbLogger(config=args, group=args.group)
        loggers.append(wandb_logger)

    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.patience))
    if args.use_lr_monitor:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    args.weights_summary = "full"

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=loggers, default_root_dir="training/logs")

    trainer.tune(lit_model, datamodule=data)  # pylint: disable=no-member

    trainer.fit(lit_model, datamodule=data)  # pylint: disable=no-member
    trainer.test(lit_model, datamodule=data)  # pylint: disable=no-member


if __name__ == "__main__":
    main()
