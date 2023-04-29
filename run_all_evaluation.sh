#!/bin/bash

# Evaluate robustness curve on MNIST
python eval_ccat.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/at
python eval_ccat.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/at
python eval_ccat.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/trades
python eval_ccat.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/trades
python eval_ccat.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/ccat
python eval_ccat.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/ccat
python eval_atrr.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/atrr
python eval_atrr.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/atrr
python eval_rcd.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/rcd
python eval_rcd.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/rcd
python eval_cpr.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/at
python eval_cpr.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/at
python eval_cpr.py --config-file configs/mnist_seen_eval.json --output-dir checkpoints/MNIST/trades
python eval_cpr.py --config-file configs/mnist_unseen_eval.json --output-dir checkpoints/MNIST/trades

# Evaluate robustness curve on SVHN
python eval_ccat.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/at
python eval_ccat.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/at
python eval_ccat.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/trades
python eval_ccat.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/trades
python eval_ccat.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/ccat
python eval_ccat.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/ccat
python eval_atrr.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/atrr
python eval_atrr.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/atrr
python eval_rcd.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/rcd
python eval_rcd.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/rcd
python eval_cpr.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/at
python eval_cpr.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/at
python eval_cpr.py --config-file configs/svhn_seen_eval.json --output-dir checkpoints/SVHN/trades
python eval_cpr.py --config-file configs/svhn_unseen_eval.json --output-dir checkpoints/SVHN/trades

# Evaluate robustness curve on CIFAR-10
python eval_ccat.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/at
python eval_ccat.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/at
python eval_ccat.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/trades
python eval_ccat.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/trades
python eval_ccat.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/ccat
python eval_ccat.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/ccat
python eval_atrr.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/atrr
python eval_atrr.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/atrr
python eval_rcd.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/rcd
python eval_rcd.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/rcd
python eval_cpr.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/at
python eval_cpr.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/at
python eval_cpr.py --config-file configs/cifar10_seen_eval.json --output-dir checkpoints/CIFAR10/trades
python eval_cpr.py --config-file configs/cifar10_unseen_eval.json --output-dir checkpoints/CIFAR10/trades
