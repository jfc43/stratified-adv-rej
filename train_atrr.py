"""
Main script for training the baseline method ATRR:
Pang, Tianyu, et al. "Adversarial Training with Rectified Rejection." arXiv preprint arXiv:2105.14785 (2021).
https://arxiv.org/pdf/2105.14785.pdf

Based on the official code from: https://github.com/P2333/Rectified-Rejection
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
from attacks.mb_pgd_attack import *
from utils.dataset import CustomDataset


def test(model, attacker, test_dataloader, max_batches, logger):
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

    adv_accs = None
    for b, (inputs, targets) in enumerate(test_dataloader):
        if b >= max_batches:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()

        # Generate adversarial examples
        adv_inputs = attacker.perturb(inputs, targets)
        # Indicator for large perturbations
        with torch.no_grad():
            adv_logits = model(adv_inputs)

        adv_error = utils.torch.classification_error(adv_logits, targets, reduction="none")
        adv_accs = utils.numpy.concatenate(adv_accs, 1 - adv_error.detach().cpu().numpy())

    logger.info("adv_acc: {:.2%}".format(np.mean(adv_accs)))


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
                           lamb,
                           tempC, 
                           tempC_trueonly,
                           epoch,
                           lr_max,
                           print_freq,
                           logger):

    num_training_iter = len(train_dataloader)
    BCEcriterion = nn.BCELoss(reduction='mean')
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
        
        # Generate adversarial examples
        adv_examples = attacker.perturb(adv_inputs, adv_targets)

        if adv_inputs.shape[0] < b_size: # fraction is not 1
            combined_inputs = torch.cat((clean_inputs, adv_examples), dim=0)
        else:
            combined_inputs = adv_examples

        model.train()
        optimizer.zero_grad()
        # `d_outputs` should correspond to the probability of rejection
        logits, aux_outputs = model(combined_inputs, return_aux=True)
        clean_logits = logits[:split]
        clean_aux_outputs = aux_outputs[:split]
        adv_logits = logits[split:]
        adv_aux_outputs = aux_outputs[split:]
        
        # robust loss
        adv_loss = utils.torch.classification_loss(adv_logits, adv_targets, reduction="mean")
        adv_acc = 1 - utils.torch.classification_error(adv_logits, adv_targets, reduction="mean")

        # aux loss for detection
        robust_output_s = torch.softmax(adv_logits * tempC, dim=1)
        robust_con_pre, robust_con_label = robust_output_s.max(1) # predicted label and confidence
        robust_output_s_ = torch.softmax(adv_logits * tempC_trueonly, dim=1)
        robust_con_y = robust_output_s_[torch.tensor(range(adv_inputs.size(0))), adv_targets].detach() # predicted prob on the true label y
        
        correct_index = torch.where(adv_logits.max(1)[1] == adv_targets)[0]
        robust_con_pre[correct_index] = robust_con_pre[correct_index].detach()

        robust_output_aux = adv_aux_outputs.sigmoid().squeeze() # bs, Calibration function A \in [0,1]
        robust_detector = robust_con_pre * robust_output_aux
        aux_loss = BCEcriterion(robust_detector, robust_con_y)

        if adv_inputs.shape[0] < b_size:
            # Loss on clean inputs
            clean_loss = utils.torch.classification_loss(clean_logits, clean_targets, reduction="mean")
            clean_acc = 1 - utils.torch.classification_error(clean_logits, clean_targets, reduction="mean")
            # Combined loss
            loss = (1 - fraction) * clean_loss + fraction * (adv_loss + lamb * aux_loss)
        else:
            clean_acc = torch.ones(1)
            loss = adv_loss + lamb * aux_loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (b+1) % print_freq == 0:
            if adv_inputs.shape[0] < b_size:
                logger.info("Progress: {:d}/{:d}, loss: {:.4f}, clean_acc: {:.2%}, adv_acc: {:.2%}".format(b+1,
                                                                                                        num_training_iter,
                                                                                                        loss.item(),
                                                                                                        clean_acc.item(),
                                                                                                        adv_acc.item()))
            else:
                logger.info("Progress: {:d}/{:d}, loss: {:.4f}, adv_acc: {:.2%}".format(b+1,
                                                                                        num_training_iter,
                                                                                        loss.item(),
                                                                                        adv_acc.item()))


def get_args():
    parser = argparse.ArgumentParser(description='train robust model with detection')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--config-file', '-c', type=str, required=True, help='config file')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='output dir')
    parser.add_argument('--pretrain-model-dir', help='dir to load pretrained models', type=str, default='')
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
    lamb = config['lamb']
    tempC = config['tempC']
    tempC_trueonly = config['tempC_trueonly']
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
        model = models.FixedLeNetTwoBranchDenseV1(N_class, resolution, out_dim=1, use_BN=True, along=True)
    elif model_arch == "resnet20":
        model = models.ResNetTwoBranchDenseV1(N_class, resolution, blocks=[3, 3, 3], out_dim=1, use_BN=True, along=True)
    elif model_arch == "wideresnet":
        model = models.WideResNetTwoBranchDenseV1(N_class, resolution, depth=28, width=10, out_dim=1, use_BN=True, along=True)
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


    objective = UntargetedObjective(loss=utils.torch.classification_loss)
    attacker = MBLinfPGDAttack(model, 
                                objective, 
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
                            lamb,
                            tempC, 
                            tempC_trueonly,
                            epoch,
                            lr,
                            print_freq,
                            logger)
        test(model, attacker, test_dataloader, max_batches, logger)

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
