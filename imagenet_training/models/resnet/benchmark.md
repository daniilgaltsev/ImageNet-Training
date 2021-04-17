# Benchmarks for ResNet.

All runs were done on the same machine, which has 12 CPU cores and 1 GPU (6GB, NVIDIA GeForce GTX 1060).

## ImageNet

Train data was split into train and validation (90%/10%), and validation set is used as test split. The second run uses only a maximum of 1000 images per class.

| Run | Train Loss | Val. Loss | Test Loss | Train Acc. | Val. Acc. | Test Acc. | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.328 | 1.431 | 1.59 | 65.1% | 65.9% | 62.8% | ~19h 40m |
| 2 | 1.47 | 1.513 | 1.678 | 63.4% | 64.2% | 61.2% | ~15h 20m |

Run 1:
```sh
python training/run_experiment.py --model_class ResNet --resnet_type resnet18 --data_class ImageNet --use_kaggle --random_resized_crop --horizontal_flip --color_jitter 0.15 --gpus 1 --precision 16 --batch_size 256 --lr 0.006 --optimizer AdamW --weight_decay 0.0008 --lr_scheduler OneCycleLR --num_workers 4 --subsample_classes 1000 --images_per_class 1300 --max_epochs 15
```

Run 2:
```sh
python training/run_experiment.py --model_class ResNet --resnet_type resnet18 --data_class ImageNet --use_kaggle --random_resized_crop --horizontal_flip --color_jitter 0.15 --gpus 1 --precision 16 --batch_size 256 --lr 0.006 --optimizer AdamW --weight_decay 0.0008 --lr_scheduler OneCycleLR --num_workers 4 --subsample_classes 1000 --images_per_class 1000 --max_epochs 15
```

## CIFAR10

Train data was split into train and validation (80%/20%), and validation set is used as test split. *-a runs used implementation of ResNet in this repository, *-b runs use torchvision architecture (--use_torchvision_model flag).

| Run | Train Loss | Val. Loss | Test Loss | Train Acc. | Val. Acc. | Test Acc. | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-a | 0.278 | 0.455 | 0.487 | 90.5% | 84.7% | 83.7% | ~14m 50s |
| 1-b | 0.335 | 0.464 | 0.488 | 89.8% | 84.2% | 83.3% | ~14m 45s |
| 2-a | 0.252 | 0.442 | 0.467 | 93.0% | 85.9% | 85.3% | ~21m 20s |
| 2-b | 0.220 | 0.428 | 0.457 | 92.0% | 85.8% | 85.2% | ~21m 45s |


Run 1-*:
```sh
python training/run_experiment.py --horizontal_flip --batch_size=1024 --data_class=CIFAR10 --gpus=1 --image_shift=0.09 --lr=0.0163 --lr_scheduler=OneCycleLR --max_epochs=28 --model_class=ResNet --optimizer=AdamW --pct_start=0.45 --precision=16 --resnet_type=resnet18 --weight_decay=0.49
```

Run 2-*:
```sh
python training/run_experiment.py --horizontal_flip --batch_size=1024 --data_class=CIFAR10 --gpus=1 --image_shift=0.09 --lr=0.008 --lr_scheduler=OneCycleLR --max_epochs=31 --model_class=ResNet --optimizer=AdamW --pct_start=0.45 --precision=16 --resnet_type=resnet50 --weight_decay=0.49
```