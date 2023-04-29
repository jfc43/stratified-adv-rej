"""
Main script for evaluating the baseline method CCAT:
Stutz, David, Matthias Hein, and Bernt Schiele. "Confidence-calibrated adversarial training: Generalizing to unseen attacks."
International Conference on Machine Learning. PMLR, 2020.

Based on the official code from: https://github.com/davidstutz/confidence-calibrated-adversarial-training
Also used for evaluating the baseline method Adversarial Training + Rejection.
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


def eval_outer_attack(attack_method,
                      outer_attack_config, 
                      model, 
                      test_dataloader, 
                      num_classes,
                      threshold, 
                      epsilon, 
                      n_samples=N_SAMPLES):
    model.eval()
    if attack_method == 'auto_attack':
        attacker = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
    elif attack_method == 'HCMOA':
        objective = UntargetedObjective(loss=utils.torch.ccat_targeted_loss)
        attacker = MBConfLinfPGDAttackMultitargeted(model,
                                                    objective,
                                                    num_classes,
                                                    epsilon=epsilon,
                                                    **outer_attack_config)
    else:
        raise KeyError(f'Not supported attack method {attack_method}')

    adv_labels = None
    adv_probs = None
    cnt = 0
    for b, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        adv_labels = utils.numpy.concatenate(adv_labels, targets.detach().cpu().numpy())

        # large perturbations
        if attack_method == 'auto_attack':
            adv_inputs = attacker.run_standard_evaluation(inputs, targets)
        else:
            adv_inputs = attacker.perturb(inputs, targets)
        with torch.no_grad():
            adv_logits = model(adv_inputs)

        adv_probs = utils.numpy.concatenate(adv_probs, torch.softmax(adv_logits, dim=1).detach().cpu().numpy())
        cnt += inputs.shape[0]
        if cnt >= n_samples:
            break

    # Error on adversarial inputs: accept and misclassify
    adv_confs = np.max(adv_probs, axis=1)   # prediction confidence
    adv_preds = np.argmax(adv_probs, axis=1)    # predicted class
    adv_error = np.logical_and(adv_preds != adv_labels, adv_confs >= threshold)

    return adv_error


def eval_inner_attack(attack_method,
                      inner_attack_config, 
                      model, 
                      test_dataloader, 
                      threshold, 
                      epsilon, 
                      eps0_range, 
                      n_samples=N_SAMPLES):
    model.eval()
    inner_attack_results = {}
    for epsilon_0 in eps0_range:
        if attack_method == 'LCIA':
            radius_objective = UntargetedObjective(loss=utils.torch.uniform_confidence_loss)
        else:
            raise KeyError(f'Not supported attack method {attack_method}')

        radius_attacker = MBConfLinfPGDAttack(model,
                                              radius_objective,
                                              epsilon=epsilon_0,
                                              **inner_attack_config)
        cnt = 0
        adv_probs_2 = None
        for b, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            # small perturbations
            if epsilon_0 > 0.:
                adv_inputs = radius_attacker.perturb(inputs, targets)
            else:
                adv_inputs = inputs
                
            with torch.no_grad():
                adv_logits = model(adv_inputs)

            adv_probs_2 = utils.numpy.concatenate(adv_probs_2, torch.softmax(adv_logits, dim=1).detach().cpu().numpy())
            cnt += inputs.shape[0]
            if cnt >= n_samples:
                break
        
        adv_confs_2 = np.max(adv_probs_2, axis=1)   # prediction confidence
        adv_error_2 = adv_confs_2 < threshold   # regard rejection as error
        inner_attack_results[epsilon_0] = adv_error_2

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


def eval(model, test_dataloader, tpr, logger):
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

    test_N = int(clean_probs.shape[0] * (1 - VAL_RATIO))
    val_probs = clean_probs[test_N:]
    val_labels = clean_labels[test_N:]
    test_probs = clean_probs[:test_N]
    test_labels = clean_labels[:test_N]

    # Find the confidence threshold using a validation set
    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)
    val_errors = (val_preds != val_labels)
    # sorted_val_confs = np.sort(val_confs)
    sorted_val_confs = np.sort(val_confs[np.logical_not(val_errors)])   # sorted confidence of correctly classified
    cutoff = math.floor(sorted_val_confs.shape[0] * round((1. - tpr), 2))
    threshold = sorted_val_confs[cutoff]

    test_confs = np.max(test_probs, axis=1)
    test_preds = np.argmax(test_probs, axis=1)
    test_errors = (test_preds != test_labels)
    test_acc = 1. - np.mean(test_errors)
    # Clean accuracy within the accepted inputs
    mask_accept = (test_confs >= threshold)
    test_acc_with_detection = 1. - np.sum(test_errors & mask_accept) / np.sum(mask_accept)
    rejection_rate = 1. - np.mean(mask_accept)
    logger.info(f"threshold: {threshold:.4f}")
    logger.info(f"clean accuracy: {test_acc:.2%}")
    logger.info(f"clean accuracy with detection: {test_acc_with_detection:.2%}, clean rejection rate: {rejection_rate:.2%}, F1 score: {2*test_acc_with_detection*(1-rejection_rate)/(test_acc_with_detection+1-rejection_rate):.2%}")

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
    # evaluate clean and get threshold
    threshold = eval(model, test_dataloader, tpr, logger)

    final_outer_adv_error = None
    final_inner_attack_results = None
    
    # evaluate robustness with rejection under AutoAttack
    outer_attack_config = None
    outer_attack_method = 'auto_attack'
    outer_file_name = f"outer_{outer_attack_method}"
    outer_file_name += f"_eps_{int(epsilon*255):d}"
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
                                            n_samples=N_SAMPLES)

    final_outer_adv_error = combine_outer_attack_results(final_outer_adv_error, outer_adv_error)
    np.save(os.path.join(output_dir, outer_file_name), outer_adv_error)
    
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
                                                n_samples=N_SAMPLES)

        final_outer_adv_error = combine_outer_attack_results(final_outer_adv_error, outer_adv_error)
        np.save(os.path.join(output_dir, outer_file_name), outer_adv_error)

        inner_attack_config.update({'base_lr': base_lr})
        inner_file_name = f"inner_{inner_attack_method}"
        inner_file_name += '_bt' if inner_attack_config['backtrack'] else ''
        inner_file_name += f"_eps_{int(epsilon*255):d}"
        inner_file_name += f"_lr_{inner_attack_config['base_lr']}"
        inner_file_name += "_result.npy"
        if os.path.exists(os.path.join(output_dir, inner_file_name)):
            inner_attack_results = np.load(os.path.join(output_dir, inner_file_name), allow_pickle=True).item()
        else:
            inner_attack_results = eval_inner_attack(inner_attack_method,
                                                    inner_attack_config,
                                                    model,
                                                    test_dataloader,
                                                    threshold,
                                                    epsilon,
                                                    eps0_range,
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
