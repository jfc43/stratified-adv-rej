import torch
import utils.torch
from .classifier import Classifier
import torch.nn as nn


class FixedLeNet(Classifier):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution=(1, 28, 28), **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        """

        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNet, self).__init__(N_class, resolution, **kwargs)

        self.append_layer('0', nn.Conv2d(resolution[0], 32, 5, padding=2))
        self.append_layer('1', nn.ReLU())
        self.append_layer('2', nn.MaxPool2d(2, 2))
        self.append_layer('3', nn.Conv2d(32, 64, 5, padding=2))
        self.append_layer('4', nn.ReLU())
        self.append_layer('5', nn.MaxPool2d(2, 2))
        self.append_layer('6', utils.torch.Flatten())
        self.append_layer('7', nn.Linear(7 * 7 * 64, 1024))
        self.append_layer('8', nn.ReLU())
        self.append_layer('9', nn.Linear(1024, self.N_class))


class FixedLeNetTwoBranch(torch.nn.Module):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution=(1, 28, 28), **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        """

        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNetTwoBranch, self).__init__(**kwargs)
        self.N_class = N_class

        self.feature_layers = nn.Sequential()
        self.feature_layers.add_module("f0", nn.Conv2d(resolution[0], 32, 5, padding=2))
        self.feature_layers.add_module("f1", nn.ReLU())
        self.feature_layers.add_module("f2", nn.MaxPool2d(2, 2))
        self.feature_layers.add_module("f3", nn.Conv2d(32, 64, 5, padding=2))
        self.feature_layers.add_module("f4", nn.ReLU())
        self.feature_layers.add_module("f5", nn.MaxPool2d(2, 2))
        self.feature_layers.add_module("f6", utils.torch.Flatten())
        self.feature_layers.add_module("f7", nn.Linear(7 * 7 * 64, 1024))
        self.feature_layers.add_module("f8", nn.ReLU())

        self.classifier_layers = nn.Sequential()
        self.classifier_layers.add_module("c0", nn.Linear(1024, self.N_class))
        
        # Shallow detector
        # self.dense_layers = nn.Sequential()
        # self.dense_layers.add_module("d0", nn.Linear(1024, 256))
        # self.dense_layers.add_module("d1", nn.ReLU())
        # self.dense_layers.add_module("d2", nn.Linear(256, 1))
        
        # Deeper detector
        self.dense_layers = nn.Sequential()
        self.dense_layers.add_module("d0", nn.Linear(1024, 256))
        self.dense_layers.add_module("d1", nn.ReLU())
        self.dense_layers.add_module("d2", nn.Linear(256, 256))
        self.dense_layers.add_module("d3", nn.ReLU())
        self.dense_layers.add_module("d4", nn.Linear(256, 1))
    
    def forward(self, x, return_d=False):
        feature = self.feature_layers(x)
        cls_output = self.classifier_layers(feature)
        d_output = self.dense_layers(feature)
        d_output = torch.sigmoid(d_output)
        
        if return_d:
            return cls_output, d_output
        else:
            return cls_output


class FixedLeNetTwoBranchDenseV1(torch.nn.Module):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution=(1, 28, 28), out_dim=10, use_BN=False, along=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        """

        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNetTwoBranchDenseV1, self).__init__(**kwargs)
        self.N_class = N_class
        self.along = along

        self.feature_layers = nn.Sequential()
        self.feature_layers.add_module("f0", nn.Conv2d(resolution[0], 32, 5, padding=2))
        self.feature_layers.add_module("f1", nn.ReLU())
        self.feature_layers.add_module("f2", nn.MaxPool2d(2, 2))
        self.feature_layers.add_module("f3", nn.Conv2d(32, 64, 5, padding=2))
        self.feature_layers.add_module("f4", nn.ReLU())
        self.feature_layers.add_module("f5", nn.MaxPool2d(2, 2))
        self.feature_layers.add_module("f6", utils.torch.Flatten())
        self.feature_layers.add_module("f7", nn.Linear(7 * 7 * 64, 1024))
        self.feature_layers.add_module("f8", nn.ReLU())

        self.classifier_layers = nn.Sequential()
        self.classifier_layers.add_module("c0", nn.Linear(1024, self.N_class))

        if use_BN:
            self.dense_layers = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
                )
        else:
            self.dense_layers = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
                )
    
    def forward(self, x, return_aux=False):
        feature = self.feature_layers(x)
        cls_output = self.classifier_layers(feature)

        if self.along:
            evidence_return = self.dense_layers(feature)
        else:
            evidence_return = self.classifier_layers(feature) + self.dense_layers(feature)
        
        if return_aux:
            return cls_output, evidence_return
        else:
            return cls_output


class FixedLeNetConf(torch.nn.Module):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution=(1, 28, 28), conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNetConf, self).__init__(**kwargs)
        self.N_class = N_class
        self.conf_approx = conf_approx
        assert self.conf_approx in ('logsumexp', 'network'), "Invalid input for 'conf_approx'"
        self.temperature = temperature
        self.model = FixedLeNet(N_class)
        if self.conf_approx == "network":
            self.conf_net = nn.Sequential()
            self.conf_net.add_module("d0", nn.Linear(self.N_class, 256))
            self.conf_net.add_module("d1", nn.ReLU())
            self.conf_net.add_module("d2", nn.Linear(256, 256))
            self.conf_net.add_module("d3", nn.ReLU())
            self.conf_net.add_module("d4", nn.Linear(256, 1))
        else:
            # Simple scale and shift of the confidence approximation score
            self.conf_net = nn.Sequential()
            self.conf_net.add_module("d0", nn.Linear(1, 1))
    
    def forward(self, x, return_d=False):
        # classifier prediction logits
        output = self.model(x)
        # detector prediction
        if self.conf_approx == 'logsumexp':
            # `output` has the prediction logits
            # `d_output` should correspond to the probability of rejection
            '''
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            K = softmax_output.size(1)
            d_output = 1. - torch.clamp((self.temperature * torch.logsumexp((1 / self.temperature) * softmax_output, dim=1,
                                                                            keepdims=True) - 1 / K) * (K / (K - 1)), min=0.0, max=1.0)
            '''
            max_logit = self.temperature * torch.logsumexp((1 / self.temperature) * output, 1, keepdims=True)
            d_output = 1. - torch.sigmoid(self.conf_net(max_logit))

        elif self.conf_approx == 'network':
            d_output = torch.sigmoid(self.conf_net(output))

        if return_d:
            return output, d_output
        else:
            return output


class FixedLeNetEnsemble(torch.nn.Module):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution, conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNetEnsemble, self).__init__(**kwargs)
        self.N_class = N_class
        # Classifier network C_1
        self.classifier = FixedLeNet(N_class, resolution)
        # Combined classifier and detector networks (C_0 and D_0). The trained classifier C_0 is used to initialize C_1,
        # but is not used for the final prediction
        self.classifier_with_reject = FixedLeNetConf(N_class,
                                                     resolution,
                                                     conf_approx=conf_approx,
                                                     temperature=temperature)
    
    def forward(self, x, return_d=False):
        cls_output = self.classifier(x)
        if return_d:
            _, d_output = self.classifier_with_reject(x, return_d=True)
            return cls_output, d_output
        else:
            return cls_output
