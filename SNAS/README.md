## Requirements
```
Python >= 3.5.5, PyTorch == 0.4, torchvision == 0.2.0
```

## Datasets

CIFAR-10 can be automatically downloaded by torchvision, ImageNet needs to be downloaded manually.

## Architecture search
```
python train_search.py --snas --epochs 150 --seed 6 --layer 8 --init_channels 16 --temp 1 \
--temp_min 0.03 --nsample 1 --temp_annealing --resource_efficient \
--resource_lambda 1e-2 --log_penalty --drop_path_prob 3e-1 --method 'reparametrization' \
--loss --remark "snas_order_layer_8_batch_64_drop_0.3_error_lnR_1e-2_reparam_gpu_1" &
```

## Architecture evaluation (using full-sized models)
```
python train.py --auxiliary --cutout --arch {arch}   # CIFAR-10 (DARTS-like architecture)

python train_edge_all.py --auxiliary --cutout --arch {arch}  # CIFAR-10 (all-edge)

python train_imagenet.py --auxiliary --arch {arch}    # ImageNet
```

