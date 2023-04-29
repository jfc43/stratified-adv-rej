#!/bin/bash

# Train models on MNIST
python train_at.py --config-file configs/mnist_train.json --output-dir checkpoints/MNIST/at
python train_trades.py --config-file configs/mnist_train.json --output-dir checkpoints/MNIST/trades
python train_ccat.py --config-file configs/mnist_train_ccat.json --output-dir checkpoints/MNIST/ccat
python train_atrr.py --config-file configs/mnist_train_atrr.json --output-dir checkpoints/MNIST/atrr
python train_rcd.py --config-file configs/mnist_train_rcd.json --output-dir checkpoints/MNIST/rcd

# Train models on SVHN
python train_at.py --config-file configs/svhn_train.json --output-dir checkpoints/SVHN/at
python train_trades.py --config-file configs/svhn_train.json --output-dir checkpoints/SVHN/trades
python train_ccat.py --config-file configs/svhn_train_ccat.json --output-dir checkpoints/SVHN/ccat
python train_atrr.py --config-file configs/svhn_train_atrr.json --output-dir checkpoints/SVHN/atrr
python train_rcd.py --config-file configs/svhn_train_rcd.json --output-dir checkpoints/SVHN/rcd

# Train models on CIFAR-10
python train_at.py --config-file configs/cifar10_train.json --output-dir checkpoints/CIFAR10/at
python train_trades.py --config-file configs/cifar10_train.json --output-dir checkpoints/CIFAR10/trades
python train_ccat.py --config-file configs/cifar10_train_ccat.json --output-dir checkpoints/CIFAR10/ccat
python train_atrr.py --config-file configs/cifar10_train_atrr.json --output-dir checkpoints/CIFAR10/atrr
python train_rcd.py --config-file configs/cifar10_train_rcd.json --output-dir checkpoints/CIFAR10/rcd
