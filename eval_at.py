"""Evaluation of standard adversarial training without rejection."""

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import utils.torch
import utils.numpy
import models
import json
from attacks.objectives import UntargetedObjective
from attacks.mb_pgd_attack import MBConfLinfPGDAttack
from utils.dataset import CustomDataset
from utils.constants import *
from imgaug import augmenters as iaa
import utils.imgaug_lib
from autoattack import AutoAttack


def eval_robustness_curve(model, test_dataloader, epsilon, eps0_range, logger=None, n_samples=N_SAMPLES):
    model.eval()

    # Use AutoAttack
    attacker = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
    
    adv_labels = None
    adv_probs = None
    cnt = 0
    for b, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        adv_labels = utils.numpy.concatenate(adv_labels, targets.detach().cpu().numpy())

        # large perturbations
        adv_inputs = attacker.run_standard_evaluation(inputs, targets)
        with torch.no_grad():
            adv_logits = model(adv_inputs)

        adv_probs = utils.numpy.concatenate(adv_probs, torch.softmax(adv_logits, dim=1).detach().cpu().numpy())
        cnt += inputs.shape[0]
        if cnt >= n_samples:
            break

    # Error on adversarial inputs: accept and misclassify
    adv_confs = np.max(adv_probs, axis=1)   # prediction confidence
    adv_preds = np.argmax(adv_probs, axis=1)    # predicted class
    adv_error = adv_preds != adv_labels
    logger.info(f"robustness with reject: {1-np.mean(adv_error):.2%}")

    curve_x = []
    curve_y = []
    for epsilon_0 in eps0_range:
        curve_x.append(epsilon_0/epsilon)
        
        final_adv_error = np.mean(adv_error)
        curve_y.append(1 - final_adv_error)

    return np.array(curve_x), np.array(curve_y)


def eval(model, test_dataloader, logger):
    # Accuracy on clean inputs
    model.eval()

    clean_probs = None
    clean_labels = None
    for b, (inputs, targets) in enumerate(test_dataloader):
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        
        clean_probs = utils.numpy.concatenate(clean_probs, torch.softmax(outputs, dim=1).detach().cpu().numpy())
        clean_labels = utils.numpy.concatenate(clean_labels, targets.detach().cpu().numpy())

    N = clean_probs.shape[0]
    test_N = int(N * (1 - VAL_RATIO))
    test_probs = clean_probs[:test_N]
    test_labels = clean_labels[:test_N]

    test_preds = np.argmax(test_probs, axis=1)
    test_errors = (test_preds != test_labels)
    test_acc = 1 - np.mean(test_errors)
    logger.info(f"clean accuracy: {test_acc:.2%}, F1 score: {test_acc*2/(test_acc+1)}")
        

def get_args():
    parser = argparse.ArgumentParser(description='train robust model with detection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='checkpoint dir')
    # args parse
    return parser.parse_args()


def main():

    args = get_args()
    # Set random seed
    utils.torch.set_seed(args.seed)

    with open(args.config_file) as config_file:
        config = json.load(config_file)

    dataset = config['dataset']
    model_arch = config['model_arch']
    epsilon = config['epsilon']
    batch_size = config['batch_size']
    checkpoint_dir = args.checkpoint_dir
    output_dir = checkpoint_dir.replace('checkpoints/', 'results/')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'eval_output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)
    logger.info(config)

    N_class = N_CLASSES[dataset]
    if dataset == 'cifar10':
        resolution = (3, 32, 32)
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        transform_test = transforms.ToTensor()

        train_dataset = datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform_train)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        epsilon /= 255.
        
    elif dataset == 'gtsrb':
        resolution = (3, 32, 32)
        train_loaded = np.load('datasets/gtsrb/train.npz')
        X_train = train_loaded['images']
        y_train = train_loaded['labels']
        test_loaded = np.load('datasets/gtsrb/test_selected.npz')
        X_test = test_loaded['images']
        y_test = test_loaded['labels']
        
        train_dataset = CustomDataset(X_train, y_train, transform=transforms.ToTensor())
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = CustomDataset(X_test, y_test, transform=transforms.ToTensor())
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        epsilon /= 255.

    elif dataset == 'svhn':
        N_class = 10
        resolution = (3, 32, 32)
        augmenters = [iaa.CropAndPad(
            percent=(0, 0.2),
            pad_mode='edge',
        ),
        iaa.ContrastNormalization((0.7, 1.3))]
        transform_train = transforms.Compose([
            np.asarray,
            iaa.Sequential([
                iaa.SomeOf(max(1, len(augmenters) // 2), augmenters),
                utils.imgaug_lib.Clip(),
            ]).augment_image,
            np.copy,
            transforms.ToTensor(),
        ])
        transform_test = transforms.ToTensor()

        train_dataset = utils.dataset.SVHNTrainSet(transform=transform_train)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = utils.dataset.SVHNTestSet(transform=transform_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        epsilon /= 255.
        
    elif dataset == 'mnist':
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    else:
        raise ValueError("Invalid or unsupported dataset '{}'".format(dataset))

    eps0_range = np.array(ALPHA_LIST, dtype=np.float32) * epsilon
    # Model Setup
    if model_arch == "lenet":
        model = models.FixedLeNet(N_class, resolution)
    elif model_arch == "resnet20":
        model = models.ResNet(N_class, resolution, blocks=[3, 3, 3])
    elif model_arch == "wideresnet":
        model = models.WideResNet(N_class, resolution, depth=28, width=10)
    else:
        raise ValueError

    checkpoint = torch.load(os.path.join(checkpoint_dir, "classifier.pth.tar"))
    model.load_state_dict(checkpoint['model'])
    model.cuda()

    eval(model, test_dataloader, logger)
    
    curve_x, curve_y = eval_robustness_curve(model, test_dataloader, epsilon,  eps0_range,
                                                            logger=logger, n_samples=N_SAMPLES)
    result_dict = {"curve_x": curve_x, "curve_y": curve_y}
    np.save(os.path.join(output_dir, f"rob_curve_eps_{int(epsilon*255):d}_at_result.npy"), result_dict)


if __name__ == "__main__":
    main()
