# ImageNet Training

Implementation of a training pipeline for ImageNet.

## Setup

To run the project you need to install all dependencies in `requirements` folder. It can be done with `pip install` or using `tasks/sync_requirements.sh` if `pip-tools` is installed.

If you need to update requirements, edit the `requirements(-dev).in` file, then run `tasks/compile_requirements.sh`.

### Set up the conda environment

With conda you need to create an environment, as defined in `environment.yml`:

```sh
conda env create -f environment.yml
```

After running `conda env create`, activate the new environment and install the requirements:

```sh
conda activate imagenet-training
tasks/sync_requirements.sh
```

## Running the Project

To start training launch `training/run_experiment.py` script. For example,

```sh
python training/run_experiment.py --model_class ResNet --resnet_type resnet18 --data_class ImageNet --use_kaggle --max_epochs 5
```

To see help for available args add `-h` flag.

### ImageNet

To train using ImageNet you first need to download the dataset from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge) and put `imagenet_object_localization_patched2019` and `LOC_val_solution.csv` into `data/downloaded/imagenet`. It can be done using these commands:

```sh
kaggle competitions download -c imagenet-object-localization-challenge -p ./data/downloaded/imagenet -f imagenet_object_localization_patched2019.tar.gz
tar -xzf imagenet_object_localization_patched2019.tar.gz
kaggle competitions download -c imagenet-object-localization-challenge -p ./data/downloaded/imagenet -f LOC_val_solution.csv
```

### Weights & Biases

To use Weights & Biases specify your info in `wandb/settings` (entity and project) and add flag `--use_wandb`.


## References

* [Full Stack Deep Learning](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs)
* [ResNet](https://arxiv.org/abs/1512.03385)
* [PyTorch Implementation of ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
* [Pre-Activation ResNet](https://arxiv.org/abs/1603.05027)
* [One Cycle](https://arxiv.org/abs/1708.07120)
