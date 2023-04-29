"""
Main script for training the baseline method RCD:
Provably robust classification of adversarial examples with detection, Fatemeh Sheikholeslami, Ali Lotfi Rezaabad,
Zico Kolter (https://openreview.net/pdf?id=sRA5rLNpmQc), ICLR 2021
"""

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
from imgaug import augmenters as iaa
import utils.imgaug_lib
from attacks.objectives import UntargetedObjective
from attacks.mb_pgd_attack import MBRCDStratifiedLinfPGDAttack
from utils.dataset import CustomDataset


def test(model, attacker, test_dataloader, N_class, max_batches, logger):
    model.eval()

    clean_accs = None
    for b, (inputs, targets) in enumerate(test_dataloader):

        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)

        clean_error = utils.torch.classification_error(outputs, targets, reduction='none')
        clean_accs = utils.numpy.concatenate(clean_accs, 1 - clean_error.detach().cpu().numpy())

    logger.info("clean acc: {:.2%}".format(np.mean(clean_accs)))

    adv_accs_1 = None
    adv_accs_2 = None
    for b, (inputs, targets) in enumerate(test_dataloader):
        if b >= max_batches:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()

        # Generate adversarial examples
        adv_inputs = attacker.perturb(inputs, targets)

        with torch.no_grad():
            adv_logits = model(adv_inputs)

        adv_split = int(attacker.fraction * adv_logits.size(0))
        adv_logits_1 = adv_logits[:adv_split]
        adv_targets_1 = targets[:adv_split]

        adv_logits_2 = adv_logits[adv_split:]
        adv_targets_2 = targets[adv_split:]

        if attacker.fraction > 0.0:
            adv_acc_1 = 1 - utils.torch.classification_error(adv_logits_1, adv_targets_1, reduction="none")
            adv_accs_1 = utils.numpy.concatenate(adv_accs_1, adv_acc_1.detach().cpu().numpy())

        if attacker.fraction < 1.0:
            adv_d_error_2 = utils.torch.classification_error(adv_logits_2, torch.ones_like(adv_targets_2) * N_class, reduction="none")
            adv_error_2 = utils.torch.classification_error(adv_logits_2, adv_targets_2, reduction="none")
            adv_acc_2 = 1 - torch.min(torch.stack([adv_error_2, adv_d_error_2], dim=1), dim=1)[0].float()
            adv_accs_2 = utils.numpy.concatenate(adv_accs_2, adv_acc_2.detach().cpu().numpy())

    if adv_accs_1 is None:
        adv_accs_1 = np.ones(1)
    if adv_accs_2 is None:
        adv_accs_2 = np.ones(1)

    logger.info("adv_acc_1: {:.2%}, adv_acc_2: {:.2%}".format(np.mean(adv_accs_1), np.mean(adv_accs_2)))


def lr_schedule(t, lr_max):
    if t < 100:
        return lr_max
    elif t < 105:
        return lr_max / 10.
    else:
        return lr_max / 100.


def train_robust_detection(model,
                           train_dataloader,
                           attacker,
                           optimizer,
                           scheduler,
                           fraction,
                           lamb_1,
                           lamb_2,
                           epoch,
                           lr_max,
                           N_class,
                           print_freq,
                           logger):

    num_training_iter = len(train_dataloader)
    for b, (inputs, targets) in enumerate(train_dataloader):

        if scheduler is None:
            epoch_now = epoch + (b + 1) / num_training_iter
            lr = lr_schedule(epoch_now, lr_max)
            optimizer.param_groups[0].update(lr=lr)

        inputs = inputs.cuda()
        targets = targets.cuda()
        b_size = inputs.size(0)
        split = int((1-fraction) * b_size)
        # update fraction for correct loss computation
        fraction = 1 - split / float(b_size)

        clean_inputs = inputs[:split]
        adv_inputs = inputs[split:]
        clean_targets = targets[:split]
        adv_targets = targets[split:]

        double_adv_inputs = torch.cat((adv_inputs, clean_inputs), 0)
        double_adv_targets = torch.cat((adv_targets, clean_targets), 0)
        # Generate adversarial examples
        adv_examples = attacker.perturb(double_adv_inputs, double_adv_targets)

        if adv_inputs.shape[0] < b_size: # fraction is not 1
            combined_inputs = torch.cat((clean_inputs, adv_examples), dim=0)
        else:
            combined_inputs = adv_examples

        model.train()
        optimizer.zero_grad()
        logits = model(combined_inputs)
        clean_logits = logits[:split]
        adv_logits = logits[split:]

        adv_split = int(attacker.fraction * adv_logits.size(0))
        if attacker.fraction > 0.0:
            # robust loss
            adv_logits_1 = adv_logits[:adv_split]
            adv_targets_1 = double_adv_targets[:adv_split]
            adv_loss_1 = utils.torch.classification_loss(adv_logits_1, adv_targets_1, reduction="mean")
            adv_acc_1 = 1 - utils.torch.classification_error(adv_logits_1, adv_targets_1, reduction="mean")
        else:
            adv_loss_1 = torch.zeros(1)
            adv_acc_1 = torch.ones(1)

        if attacker.fraction < 1.0:
            # robust abstain loss
            adv_logits_2 = adv_logits[adv_split:]
            adv_targets_2 = double_adv_targets[adv_split:]
            adv_loss_2 = utils.torch.robust_abstain_loss(adv_logits_2, adv_targets_2, reduction="mean")
            adv_error_2 = utils.torch.classification_error(adv_logits_2, adv_targets_2, reduction="none")
            adv_d_error_2 = utils.torch.classification_error(adv_logits_2, torch.ones_like(adv_targets_2) * N_class, reduction="none")
            adv_acc_2 = 1 - torch.mean(torch.min(torch.stack([adv_error_2, adv_d_error_2], dim=1), dim=1)[0].float())
        else:
            adv_loss_2 = torch.zeros(1)
            adv_acc_2 = torch.ones(1)

        if adv_inputs.shape[0] < b_size:
            # Loss on clean inputs
            clean_loss = utils.torch.classification_loss(clean_logits, clean_targets, reduction="mean")
            clean_acc = 1 - utils.torch.classification_error(clean_logits, clean_targets, reduction="mean")
            # Combined loss
            loss = fraction * (adv_loss_1 + lamb_1 * adv_loss_2) + (1 - fraction) * lamb_2 * clean_loss
        else:
            clean_acc = torch.ones(1)
            loss = adv_loss_1 + lamb_1 * adv_loss_2

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (b+1) % print_freq == 0:
            logger.info("Progress: {:d}/{:d}, loss: {:.4f}, clean_acc: {:.2%}, adv_acc_1: {:.2%}, adv_acc_2: {:.2%}".format(b+1,
                                                                                                        num_training_iter,
                                                                                                        loss.item(),
                                                                                                        clean_acc.item(),
                                                                                                        adv_acc_1.item(),
                                                                                                        adv_acc_2.item()))


def get_args():
    parser = argparse.ArgumentParser(description='train robust model with detection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    parser.add_argument('--output-dir', type=str, required=True, help='output dir')
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
    lr = config['lr']
    optimizer_name = config['optimizer_name']
    epsilon = config['epsilon']
    eps_iter = config['eps_iter']
    nb_iter = config['nb_iter']
    print_freq = config['print_freq']
    checkpoint_freq = config['checkpoint_freq']
    max_batches = config['max_batches']
    # fraction of clean inputs
    fraction = config['fraction']
    # adv_fraction: fraction of robust abstain examples
    adv_fraction = config['adv_fraction']
    lamb_1 = config['lamb_1']
    lamb_2 = config['lamb_2']
    nepoch = config['nepoch']
    batch_size = config['batch_size']
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train_output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)
    logger.info(config)

    if dataset == 'cifar10':
        N_class = 10
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
        eps_iter /= 255.

    elif dataset == 'gtsrb':
        N_class = 43
        resolution = (3, 32, 32)
        train_loaded = np.load('datasets/gtsrb/train.npz')
        X_train = train_loaded['images']
        y_train = train_loaded['labels']
        test_loaded = np.load('datasets/gtsrb/test.npz')
        X_test = test_loaded['images']
        y_test = test_loaded['labels']

        train_dataset = CustomDataset(X_train, y_train, transform=transforms.ToTensor())
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = CustomDataset(X_test, y_test, transform=transforms.ToTensor())
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        epsilon /= 255.
        eps_iter /= 255.

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
        eps_iter /= 255.

    elif dataset == 'mnist':
        N_class = 10
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    else:
        raise ValueError("Dataset '{}' is not supported.".format(dataset))

    # Model Setup
    if model_arch == "lenet":
        model = models.FixedLeNet(N_class+1, resolution)
    elif model_arch == "resnet20":
        model = models.ResNet(N_class+1, resolution, blocks=[3, 3, 3])
    elif model_arch == "wideresnet":
        model = models.WideResNet(N_class+1, resolution, depth=28, width=10)
    else:
        raise ValueError

    model.cuda()
    if optimizer_name == 'SGD-pp':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = None
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = utils.torch.get_exponential_scheduler(optimizer, batches_per_epoch=len(train_dataloader), gamma=0.95)
    else:
        raise ValueError

    if fraction >= 1.0:
        raise ValueError("Fraction of clean inputs should be less than 1")

    objective_1 = UntargetedObjective(loss=utils.torch.classification_loss)
    objective_2 = UntargetedObjective(loss=utils.torch.robust_abstain_loss)
    attacker = MBRCDStratifiedLinfPGDAttack(model,
                                objective_1,
                                objective_2,
                                fraction=adv_fraction,
                                epsilon=epsilon,
                                max_iterations=nb_iter,
                                base_lr=eps_iter,
                                momentum=0.0,
                                lr_factor=1.5,
                                backtrack=False,
                                rand_init_name="random+zero",
                                num_rand_init=1,
                                clip_min=0.0,
                                clip_max=1.0)

    for epoch in range(nepoch):
        logger.info("Epoch: {:d}".format(epoch))

        train_robust_detection(model,
                            train_dataloader,
                            attacker,
                            optimizer,
                            scheduler,
                            fraction,
                            lamb_1,
                            lamb_2,
                            epoch,
                            lr,
                            N_class,
                            print_freq,
                            logger)
        test(model, attacker, test_dataloader, N_class, max_batches, logger)

        if (epoch+1) % checkpoint_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_dir, 'checkpoint_{:d}.pth'.format(epoch+1)))

    torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(output_dir, 'classifier.pth.tar'))


if __name__ == "__main__":
    main()
