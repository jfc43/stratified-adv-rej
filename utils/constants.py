
BASE_LR_RANGE = {'mnist': [0.1, 0.01, 0.005],
                 'cifar10': [0.01, 0.005, 0.001],
                 'gtsrb': [0.01, 0.005, 0.001],
                 'svhn': [0.01, 0.005, 0.001]}

# Range of alpha values used in the definition of robust error with rejection
ALPHA_LIST = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]

GAP_FACTOR = {
    'mnist': 0.02,
    'cifar10': 0.5/255,
    'gtsrb': 0.5/255,
    'svhn': 0.5/255
}

# Number of classes
N_CLASSES = {
    'mnist': 10,
    'cifar10': 10,
    'gtsrb': 43,
    'svhn': 10
}

# Target true positive rate for selecting the rejection threshold
TPR_THRESHOLD = {
    'mnist': 0.99,
    'cifar10': 0.95,
    'gtsrb': 0.95,
    'svhn': 0.95
}

# Fraction of samples used for validation
VAL_RATIO = 0.1

# Sample size used for calculating the robustness with rejection
N_SAMPLES = 1000

# Setting for the adaptive attacks
CONFIG_PGDBT_ATTACK_OUTER = {
    'max_iterations': 1000,
    'base_lr': 0.001,
    'momentum': 0.9,
    'lr_factor': 1.1,
    'backtrack': True,
    'rand_init_name': 'random',
    'num_rand_init': 10,
    'clip_min': 0.0,
    'clip_max': 1.0
}

CONFIG_PGD_ATTACK_INNER = {
    'max_iterations': 200,
    'base_lr': 0.1,
    'momentum': 0.9,
    'lr_factor': 1.25,
    'backtrack': False,
    'rand_init_name': 'random',
    'num_rand_init': 5,
    'clip_min': 0.0,
    'clip_max': 1.0
}

CONFIG_MULTITARGET_ATTACK_OUTER = {
    'max_iterations': 200,
    'base_lr': 0.1,
    'momentum': 0.9,
    'lr_factor': 1.1,
    'backtrack': False,
    'rand_init_name': 'random',
    'num_rand_init': 5,
    'clip_min': 0.0,
    'clip_max': 1.0
}

CONFIG_MULTITARGET_ATTACK_INNER = {
     'max_iterations': 200,
     'base_lr': 0.1,
     'momentum': 0.9,
     'lr_factor': 1.25,
     'backtrack': False,
     'rand_init_name': 'random',
     'num_rand_init': 1,
     'clip_min': 0.0,
     'clip_max': 1.0
}


# Plot colors and markers
# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = ['r', 'b', 'c', 'orange', 'g', 'm', 'lawngreen', 'grey', 'hotpink', 'y', 'steelblue', 'tan',
          'lightsalmon', 'navy', 'gold']
MARKERS = ['o', '^', 'v', 's', '*', 'x', 'd', '>', '<', '1', 'h', 'P', '_', '2', '|', '3', '4']


def lr_schedule(t, lr_max):
    # Learning rate schedule
    if t < 100:
        return lr_max
    elif t < 105:
        return lr_max / 10.
    else:
        return lr_max / 100.
