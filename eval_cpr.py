"""
Main script for evaluating the proposed method CPR.
"""

from functools import partial
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
import math
from attacks.objectives import *
from attacks.mb_pgd_attack import *
from utils.dataset import CustomDataset
from utils.constants import *
from imgaug import augmenters as iaa
import utils.imgaug_lib
from autoattack import AutoAttack


class ConsistentRejDefense(torch.nn.Module):

    def __init__(self, model, defense_attack_params):
        super(ConsistentRejDefense, self).__init__()
        self.model = model
        objective = UntargetedObjective(loss=utils.torch.classification_loss)
        self.defense_attacker = MBLinfPGDAttack(model,
                                    objective,
                                    **defense_attack_params)

    def transform(self, inputs):
        with torch.no_grad():
            logits = self.model(inputs)
        pred_labels = torch.max(logits, axis=1)[1]
        adv_inputs = self.defense_attacker.perturb(inputs, pred_labels)
        return adv_inputs

    def forward(self, inputs):
        with torch.no_grad():
            logits = self.model(inputs)
        pred_labels = torch.max(logits, axis=1)[1]
        adv_inputs = self.defense_attacker.perturb(inputs, pred_labels)
        with torch.no_grad():
            adv_logits = self.model(adv_inputs)
        adv_pred_labels = torch.max(adv_logits, axis=1)[1]
        adv_confs = (pred_labels==adv_pred_labels).float()
        return logits, adv_confs


def eval_outer_attack(attack_method,
                      outer_attack_config,
                      model,
                      test_dataloader,
                      num_classes,
                      threshold,
                      epsilon,
                      defense_attack_params,
                      validation, 
                      total_samples,
                      n_samples=N_SAMPLES):
    model.eval()
    defense = ConsistentRejDefense(model, defense_attack_params)
    if attack_method == 'auto_attack':
        attacker = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
    elif attack_method == 'HCMOA':
        objective = UntargetedObjective(loss=utils.torch.ccat_targeted_loss)
        attacker = MBConfLinfPGDAttackMultitargeted(model,
                                                    objective,
                                                    num_classes,
                                                    epsilon=epsilon,
                                                    **outer_attack_config)
    elif attack_method == 'CHCMOA':
        objective = UntargetedObjective(loss=utils.torch.ccat_targeted_loss)
        attacker = MBCONSRLinfPGDAttackMultitargeted(model,
                                                defense,
                                                objective,
                                                num_classes,
                                                epsilon=epsilon,
                                                **outer_attack_config)
    else:
        raise KeyError(f'Not supported attack method {attack_method}')

    adv_labels = None
    adv_probs = None
    adv_confs = None
    cnt = 0
    for b, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        if validation and cnt < total_samples - n_samples:
            # If validation: the last `n_samples` are used for evaluating the robustness with rejection
            cnt += inputs.shape[0]
            continue
        adv_labels = utils.numpy.concatenate(adv_labels, targets.detach().cpu().numpy())

        # large perturbations
        if attack_method == 'auto_attack':
            adv_inputs = attacker.run_standard_evaluation(inputs, targets)
        else:
            adv_inputs = attacker.perturb(inputs, targets)
        adv_logits, batch_adv_conf = defense(adv_inputs)

        adv_probs = utils.numpy.concatenate(adv_probs, torch.softmax(adv_logits, dim=1).detach().cpu().numpy())
        adv_confs = utils.numpy.concatenate(adv_confs, batch_adv_conf.detach().cpu().numpy())
        cnt += inputs.shape[0]
        if (not validation) and cnt >= n_samples:
            # If not validation: the first `n_samples` are used for evaluating the robustness with rejection
            break

    # Error on adversarial inputs: accept and misclassify
    adv_preds = np.argmax(adv_probs, axis=1)    # predicted class
    adv_error = np.logical_and(adv_preds != adv_labels, adv_confs >= threshold)

    return adv_error


def eval_inner_attack(attack_method,
                      inner_attack_config,
                      model,
                      test_dataloader,
                      num_classes,
                      threshold,
                      epsilon,
                      eps0_range,
                      defense_attack_params,
                      validation, 
                      total_samples,
                      n_samples=N_SAMPLES):
    model.eval()
    defense = ConsistentRejDefense(model, defense_attack_params)
    inner_attack_results = {}
    for epsilon_0 in eps0_range:
        if attack_method == 'LCIA':
            radius_objective = UntargetedObjective(loss=utils.torch.uniform_confidence_loss)
        elif attack_method == 'CLCIA':
            radius_objective = UntargetedObjective(loss=utils.torch.uniform_confidence_loss)
        elif attack_method == 'PDIA':
            radius_objective = UntargetedObjective(loss=utils.torch.ccat_targeted_loss)
        else:
            raise KeyError(f'Not supported attack method {attack_method}')

        if attack_method == 'CLCIA':
            radius_attacker = MBCONSRLinfPGDAttack(model,
                                                defense,
                                                radius_objective,
                                                epsilon=epsilon_0,
                                                **inner_attack_config)
        elif attack_method == 'PDIA':
            radius_attacker = MBCONSRLinfPGDInnerAttackMultitargeted(model,
                                                defense,
                                                radius_objective,
                                                num_classes=num_classes,
                                                epsilon=epsilon_0,
                                                **inner_attack_config)
        else:
            radius_attacker = MBConfLinfPGDAttack(model,
                                                radius_objective,
                                                epsilon=epsilon_0,
                                                **inner_attack_config)
        cnt = 0
        adv_probs = None
        adv_confs = None
        for b, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            if validation and cnt < total_samples - n_samples:
                cnt += inputs.shape[0]
                continue
            # small perturbations
            if epsilon_0 > 0.:
                adv_inputs = radius_attacker.perturb(inputs, targets)
            else:
                adv_inputs = inputs

            adv_logits, batch_adv_conf = defense(adv_inputs)

            adv_probs = utils.numpy.concatenate(adv_probs, torch.softmax(adv_logits, dim=1).detach().cpu().numpy())
            adv_confs = utils.numpy.concatenate(adv_confs, batch_adv_conf.detach().cpu().numpy())
            cnt += inputs.shape[0]
            if (not validation) and cnt >= n_samples:
                break

        inner_attack_results[epsilon_0] = adv_confs < threshold

    return inner_attack_results


def eval_robustness_curve(epsilon, outer_adv_error, inner_attack_results, logger):

    logger.info(f"robustness with reject: {1-np.mean(outer_adv_error):.2%}")

    curve_x = []
    curve_y = []
    for epsilon_0 in inner_attack_results:
        curve_x.append(epsilon_0 / epsilon)
        adv_error_2 = inner_attack_results[epsilon_0]
        final_adv_error = np.mean(np.logical_or(outer_adv_error, adv_error_2))
        curve_y.append(1 - final_adv_error)
        logger.info(f"eps0: {epsilon_0}, rejection rate: {np.mean(adv_error_2):.2%}, robustness:{1-final_adv_error:.2%}")

    return np.array(curve_x), np.array(curve_y)


def eval(model, test_dataloader, tpr, defense_attack_params, logger):
    model.eval()
    defense = ConsistentRejDefense(model, defense_attack_params)

    clean_probs = None
    clean_confs = None
    clean_labels = None
    for b, (inputs, targets) in enumerate(test_dataloader):

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs, batch_conf = defense(inputs)

        clean_probs = utils.numpy.concatenate(clean_probs, torch.softmax(outputs, dim=1).detach().cpu().numpy())
        clean_labels = utils.numpy.concatenate(clean_labels, targets.detach().cpu().numpy())
        clean_confs = utils.numpy.concatenate(clean_confs, batch_conf.detach().cpu().numpy())

    test_N = int(clean_probs.shape[0] * (1 - VAL_RATIO))
    val_probs = clean_probs[test_N:]
    val_labels = clean_labels[test_N:]
    val_confs = clean_confs[test_N:]
    test_probs = clean_probs[:test_N]
    test_labels = clean_labels[:test_N]
    test_confs = clean_confs[:test_N]

    # Find the confidence threshold using a validation set
    val_preds = np.argmax(val_probs, axis=1)
    val_errors = (val_preds != val_labels)
    val_acc = 1. - np.mean(val_errors)
    threshold = 0.5
    val_mask_accept = (val_confs >= threshold)
    val_acc_with_detection = 1. - np.sum(val_errors & val_mask_accept) / np.sum(val_mask_accept)
    val_rejection_rate = 1. - np.mean(val_mask_accept)
    logger.info(f"threshold: {threshold:.4f}")
    logger.info(f"clean val accuracy: {val_acc:.2%}")
    logger.info(f"clean val accuracy with detection: {val_acc_with_detection:.2%}, clean val rejection rate: {val_rejection_rate:.2%}")

    test_preds = np.argmax(test_probs, axis=1)
    test_errors = (test_preds != test_labels)
    test_acc = 1. - np.mean(test_errors)
    # Clean accuracy within the accepted inputs
    test_mask_accept = (test_confs >= threshold)
    test_acc_with_detection = 1. - np.sum(test_errors & test_mask_accept) / np.sum(test_mask_accept)
    test_rejection_rate = 1. - np.mean(test_mask_accept)
    logger.info(f"clean test accuracy: {test_acc:.2%}")
    logger.info(f"clean test accuracy with detection: {test_acc_with_detection:.2%}, clean test rejection rate: {test_rejection_rate:.2%}, F1 score: {2*test_acc_with_detection*(1-test_rejection_rate)/(test_acc_with_detection+1-test_rejection_rate):.2%}")

    return threshold


def combine_outer_attack_results(combine_adv_error, curr_adv_error):
    if combine_adv_error is None:
        return curr_adv_error
    else:
        return curr_adv_error | combine_adv_error


def combine_inner_attack_results(combine_inner_results, curr_inner_results):
    if combine_inner_results is None:
        return curr_inner_results

    for eps0 in curr_inner_results:
        combine_inner_results[eps0] |= curr_inner_results[eps0]

    return combine_inner_results


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the robustness with rejection of CCAT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='checkpoint dir')
    parser.add_argument('--validation', action='store_true', help='whether to use validation set to select hyper-parameters')
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
    validation = args.validation

    N_class = N_CLASSES[dataset]
    tpr = TPR_THRESHOLD[dataset]
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

        defense_attack_params = {'epsilon': 0.0055,
                             'max_iterations': 10,
                             'base_lr': 0.001,
                             'momentum': 0.0,
                             'lr_factor': 1.5,
                             'backtrack': False,
                             'rand_init_name': "zero",
                             'num_rand_init': 1,
                             'clip_min': 0.0,
                             'clip_max': 1.0}

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

        defense_attack_params = {'epsilon': 0.0055,
                             'max_iterations': 10,
                             'base_lr': 0.001,
                             'momentum': 0.0,
                             'lr_factor': 1.5,
                             'backtrack': False,
                             'rand_init_name': "zero",
                             'num_rand_init': 1,
                             'clip_min': 0.0,
                             'clip_max': 1.0}

    elif dataset == 'svhn':
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

        defense_attack_params = {'epsilon': 0.0055,
                             'max_iterations': 10,
                             'base_lr': 0.001,
                             'momentum': 0.0,
                             'lr_factor': 1.5,
                             'backtrack': False,
                             'rand_init_name': "zero",
                             'num_rand_init': 1,
                             'clip_min': 0.0,
                             'clip_max': 1.0}

    elif dataset == 'mnist':
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        defense_attack_params = {'epsilon': 0.1,
                             'max_iterations': 20,
                             'base_lr': 0.01,
                             'momentum': 0.0,
                             'lr_factor': 1.5,
                             'backtrack': False,
                             'rand_init_name': "zero",
                             'num_rand_init': 1,
                             'clip_min': 0.0,
                             'clip_max': 1.0}

    else:
        raise ValueError("Invalid or unsupported dataset '{}'".format(dataset))

    output_dir = checkpoint_dir.replace('checkpoints/', 'results/consistent_rejection_v3/')
    defense_epsilon = defense_attack_params['epsilon']
    defense_iter = defense_attack_params['max_iterations']
    if defense_iter==10:
        output_dir = os.path.join(output_dir, f'defense_eps_{defense_epsilon}')
    else:
        output_dir = os.path.join(output_dir, f'defense_eps_{defense_epsilon}_iter_{defense_iter}')
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
    # evaluate clean and get threshold
    threshold = eval(model, test_dataloader, tpr,
                    defense_attack_params=defense_attack_params,
                    logger=logger)
    n_test = len(test_dataset)

    final_outer_adv_error = None
    final_inner_attack_results = None

    # evaluate robustness with rejection under AutoAttack
    outer_attack_config = None
    outer_attack_method = 'auto_attack'
    outer_file_name = f"outer_{outer_attack_method}"
    outer_file_name += f"_eps_{int(epsilon*255):d}"
    outer_file_name += '_val_' if validation else ''
    outer_file_name += "_result.npy"
    if os.path.exists(os.path.join(output_dir, outer_file_name)):
        outer_adv_error = np.load(os.path.join(output_dir, outer_file_name), allow_pickle=True)
    else:
        outer_adv_error = eval_outer_attack(outer_attack_method,
                                            outer_attack_config,
                                            model,
                                            test_dataloader,
                                            N_class,
                                            threshold,
                                            epsilon,
                                            defense_attack_params=defense_attack_params,
                                            validation=validation, 
                                            total_samples=n_test,
                                            n_samples=N_SAMPLES)

    final_outer_adv_error = combine_outer_attack_results(final_outer_adv_error, outer_adv_error)
    np.save(os.path.join(output_dir, outer_file_name), outer_adv_error)

    # evaluate robustness with rejection under BPDA multitargeted attack
    outer_attack_config = CONFIG_MULTITARGET_ATTACK_OUTER
    outer_attack_method = 'CHCMOA'
    for base_lr in BASE_LR_RANGE[dataset]:
        outer_attack_config.update({'base_lr': base_lr})
        outer_file_name = f"outer_{outer_attack_method}"
        outer_file_name += '_bt' if outer_attack_config['backtrack'] else ''
        outer_file_name += f"_eps_{int(epsilon*255):d}"
        outer_file_name += f"_lr_{outer_attack_config['base_lr']}"
        outer_file_name += '_val_' if validation else ''
        outer_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, outer_file_name)):
            outer_adv_error = np.load(os.path.join(output_dir, outer_file_name), allow_pickle=True)
        else:
            outer_adv_error = eval_outer_attack(outer_attack_method,
                                                outer_attack_config,
                                                model,
                                                test_dataloader,
                                                N_class,
                                                threshold,
                                                epsilon,
                                                defense_attack_params=defense_attack_params,
                                                validation=validation, 
                                                total_samples=n_test,
                                                n_samples=N_SAMPLES)

        final_outer_adv_error = combine_outer_attack_results(final_outer_adv_error, outer_adv_error)
        np.save(os.path.join(output_dir, outer_file_name), outer_adv_error)
        logger.info(f'{outer_file_name} saved!')

    # evaluate robustness with rejection under BPDA inner attack
    inner_attack_config = CONFIG_PGD_ATTACK_INNER
    inner_attack_method = 'CLCIA'
    for base_lr in BASE_LR_RANGE[dataset]:

        inner_attack_config.update({'base_lr': base_lr})
        inner_file_name = f"inner_{inner_attack_method}"
        inner_file_name += '_bt' if inner_attack_config['backtrack'] else ''
        inner_file_name += f"_eps_{int(epsilon*255):d}"
        inner_file_name += f"_lr_{inner_attack_config['base_lr']}"
        inner_file_name += '_val_' if validation else ''
        inner_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, inner_file_name)):
            inner_attack_results = np.load(os.path.join(output_dir, inner_file_name), allow_pickle=True).item()
        else:
            inner_attack_results = eval_inner_attack(inner_attack_method,
                                                    inner_attack_config,
                                                    model,
                                                    test_dataloader,
                                                    N_class,
                                                    threshold,
                                                    epsilon,
                                                    eps0_range,
                                                    defense_attack_params=defense_attack_params,
                                                    validation=validation, 
                                                    total_samples=n_test,
                                                    n_samples=N_SAMPLES)

        final_inner_attack_results = combine_inner_attack_results(final_inner_attack_results, inner_attack_results)
        np.save(os.path.join(output_dir, inner_file_name), inner_attack_results)
        logger.info(f'{inner_file_name} saved!')

    # evaluate robustness with rejection under BPDA inner multitargeted attack
    inner_attack_config = CONFIG_MULTITARGET_ATTACK_INNER
    inner_attack_method = 'PDIA'
    for base_lr in BASE_LR_RANGE[dataset]:
        inner_attack_config.update({'base_lr': base_lr})
        inner_file_name = f"inner_{inner_attack_method}"
        inner_file_name += '_bt' if inner_attack_config['backtrack'] else ''
        inner_file_name += f"_eps_{int(epsilon*255):d}"
        inner_file_name += f"_lr_{inner_attack_config['base_lr']}"
        inner_file_name += '_val_' if validation else ''
        inner_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, inner_file_name)):
            inner_attack_results = np.load(os.path.join(output_dir, inner_file_name), allow_pickle=True).item()
        else:
            inner_attack_results = eval_inner_attack(inner_attack_method,
                                                    inner_attack_config,
                                                    model,
                                                    test_dataloader,
                                                    N_class,
                                                    threshold,
                                                    epsilon,
                                                    eps0_range,
                                                    defense_attack_params=defense_attack_params,
                                                    validation=validation, 
                                                    total_samples=n_test,
                                                    n_samples=N_SAMPLES)

        final_inner_attack_results = combine_inner_attack_results(final_inner_attack_results, inner_attack_results)
        np.save(os.path.join(output_dir, inner_file_name), inner_attack_results)
        logger.info(f'{inner_file_name} saved!')

    # evaluate robustness with rejection under multitargeted attack
    outer_attack_config = CONFIG_MULTITARGET_ATTACK_OUTER
    inner_attack_config = CONFIG_PGD_ATTACK_INNER
    outer_attack_method = 'HCMOA'
    inner_attack_method = 'LCIA'
    for base_lr in BASE_LR_RANGE[dataset]:
        outer_attack_config.update({'base_lr': base_lr})
        outer_file_name = f"outer_{outer_attack_method}"
        outer_file_name += '_bt' if outer_attack_config['backtrack'] else ''
        outer_file_name += f"_eps_{int(epsilon*255):d}"
        outer_file_name += f"_lr_{outer_attack_config['base_lr']}"
        outer_file_name += '_val_' if validation else ''
        outer_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, outer_file_name)):
            outer_adv_error = np.load(os.path.join(output_dir, outer_file_name), allow_pickle=True)
        else:
            outer_adv_error = eval_outer_attack(outer_attack_method,
                                                outer_attack_config,
                                                model,
                                                test_dataloader,
                                                N_class,
                                                threshold,
                                                epsilon,
                                                defense_attack_params=defense_attack_params,
                                                validation=validation, 
                                                total_samples=n_test,
                                                n_samples=N_SAMPLES)

        final_outer_adv_error = combine_outer_attack_results(final_outer_adv_error, outer_adv_error)
        np.save(os.path.join(output_dir, outer_file_name), outer_adv_error)

        inner_attack_config.update({'base_lr': base_lr})
        inner_file_name = f"inner_{inner_attack_method}"
        inner_file_name += '_bt' if inner_attack_config['backtrack'] else ''
        inner_file_name += f"_eps_{int(epsilon*255):d}"
        inner_file_name += f"_lr_{inner_attack_config['base_lr']}"
        inner_file_name += '_val_' if validation else ''
        inner_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, inner_file_name)):
            inner_attack_results = np.load(os.path.join(output_dir, inner_file_name), allow_pickle=True).item()
        else:
            inner_attack_results = eval_inner_attack(inner_attack_method,
                                                    inner_attack_config,
                                                    model,
                                                    test_dataloader,
                                                    N_class,
                                                    threshold,
                                                    epsilon,
                                                    eps0_range,
                                                    defense_attack_params=defense_attack_params,
                                                    validation=validation, 
                                                    total_samples=n_test,
                                                    n_samples=N_SAMPLES)

        final_inner_attack_results = combine_inner_attack_results(final_inner_attack_results, inner_attack_results)
        np.save(os.path.join(output_dir, inner_file_name), inner_attack_results)
        logger.info(f'{outer_file_name} saved!')
        logger.info(f'{inner_file_name} saved!')
        eval_robustness_curve(epsilon, outer_adv_error, inner_attack_results, logger=logger)

    # Evaluate robustness curve
    logger.info('Final results under an ensemble of attacks:')
    curve_x, curve_y = eval_robustness_curve(epsilon, final_outer_adv_error, final_inner_attack_results, logger=logger)
    result_dict = {"curve_x": curve_x, "curve_y": curve_y}
    np.save(os.path.join(output_dir, f"rob_curve_eps_{int(epsilon*255):d}_result.npy"), result_dict)


if __name__ == "__main__":
    main()
