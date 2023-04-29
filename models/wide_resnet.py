"""
Wide ResNet.
Taken from https://github.com/meliketoy/wide-resnet.pytorch.
"""
import numpy
import torch
import utils.torch
from .classifier import Classifier
from .wide_resnet_block import WideResNetBlock
import torch.nn as nn


class WideResNet(Classifier):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        """

        super(WideResNet, self).__init__(N_class, resolution, **kwargs)

        self.depth = depth
        """ (int) Depth. """

        self.width = width
        """ (int) Width. """

        self.channels = channels
        """ (int) Channels. """

        self.dropout = dropout
        """ (int) Dropout. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.in_planes = channels
        """ (int) Helper for channels. """

        self.inplace = False
        """ (bool) Inplace. """

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = width

        planes = [self.channels, self.channels*k, 2*self.channels*k, 4*self.channels*k]

        downsampled = 1
        conv = torch.nn.Conv2d(resolution[0], planes[0], kernel_size=3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(conv.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv0', conv)

        block1 = self._wide_layer(WideResNetBlock, planes[1], n, stride=1)
        self.append_layer('block1', block1)
        block2 = self._wide_layer(WideResNetBlock, planes[2], n, stride=2)
        downsampled *= 2
        self.append_layer('block2', block2)
        block3 = self._wide_layer(WideResNetBlock, planes[3], n, stride=2)
        downsampled *= 2
        self.append_layer('block3', block3)

        if self.normalization:
            bn = torch.nn.BatchNorm2d(planes[3], momentum=0.9)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.append_layer('bn3', bn)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.append_layer('relu3', relu)

        representation = planes[3]
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.append_layer('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.append_layer('view', view)

        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(planes[3], self._N_output)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def _wide_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout, self.normalization))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)


class WideResNetTwoBranch(torch.nn.Module):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        """

        super(WideResNetTwoBranch, self).__init__(**kwargs)

        self.N_class = N_class
        self.resolution = resolution
        self.depth = depth
        """ (int) Depth. """

        self.width = width
        """ (int) Width. """

        self.channels = channels
        """ (int) Channels. """

        self.dropout = dropout
        """ (int) Dropout. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.in_planes = channels
        """ (int) Helper for channels. """

        self.inplace = False
        """ (bool) Inplace. """

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        self.feature_layers = nn.Sequential()

        n = int((depth-4)/6)
        k = width

        planes = [self.channels, self.channels*k, 2*self.channels*k, 4*self.channels*k]

        downsampled = 1
        conv = torch.nn.Conv2d(resolution[0], planes[0], kernel_size=3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(conv.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(conv.bias, 0)
        self.feature_layers.add_module('conv0', conv)

        block1 = self._wide_layer(WideResNetBlock, planes[1], n, stride=1)
        self.feature_layers.add_module('block1', block1)
        block2 = self._wide_layer(WideResNetBlock, planes[2], n, stride=2)
        downsampled *= 2
        self.feature_layers.add_module('block2', block2)
        block3 = self._wide_layer(WideResNetBlock, planes[3], n, stride=2)
        downsampled *= 2
        self.feature_layers.add_module('block3', block3)

        if self.normalization:
            bn = torch.nn.BatchNorm2d(planes[3], momentum=0.9)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.feature_layers.add_module('bn3', bn)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.feature_layers.add_module('relu3', relu)

        representation = planes[3]
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.feature_layers.add_module('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.feature_layers.add_module('view', view)

        self.classifier_layers = nn.Sequential()
        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(planes[3], self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.classifier_layers.add_module('logits', logits)

        self.dense_layers = nn.Sequential()
        # # Shallow detector
        # self.dense_layers.add_module("d0", nn.Linear(representation, 256))
        # self.dense_layers.add_module("d1", nn.BatchNorm1d(256))
        # self.dense_layers.add_module("d2", nn.ReLU())
        # self.dense_layers.add_module("d3", nn.Linear(256, 1))

        # Detector with more layers
        self.dense_layers.add_module("d0", nn.Linear(representation, 1024))
        self.dense_layers.add_module("bn0", nn.BatchNorm1d(1024))
        self.dense_layers.add_module("rl0", nn.ReLU())
        for j in range(5):
            self.dense_layers.add_module(f"d{j + 1:d}", nn.Linear(1024, 1024))
            self.dense_layers.add_module(f"rl{j + 1:d}", nn.ReLU())

        self.dense_layers.add_module("de", nn.Linear(1024, 1))

    def _wide_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout, self.normalization))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

    def forward(self, x, return_d=False):
        feature = self.feature_layers(x)
        cls_output = self.classifier_layers(feature)
        d_output = self.dense_layers(feature)
        d_output = torch.sigmoid(d_output)
        if return_d:
            return cls_output, d_output
        else:
            return cls_output


class WideResNetTwoBranchDenseV1(torch.nn.Module):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0, out_dim=10, use_BN=False, along=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        """

        super(WideResNetTwoBranchDenseV1, self).__init__(**kwargs)

        self.N_class = N_class
        self.along = along
        self.resolution = resolution
        self.depth = depth
        """ (int) Depth. """

        self.width = width
        """ (int) Width. """

        self.channels = channels
        """ (int) Channels. """

        self.dropout = dropout
        """ (int) Dropout. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.in_planes = channels
        """ (int) Helper for channels. """

        self.inplace = False
        """ (bool) Inplace. """

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        self.feature_layers = nn.Sequential()

        n = int((depth-4)/6)
        k = width

        planes = [self.channels, self.channels*k, 2*self.channels*k, 4*self.channels*k]

        downsampled = 1
        conv = torch.nn.Conv2d(resolution[0], planes[0], kernel_size=3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(conv.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(conv.bias, 0)
        self.feature_layers.add_module('conv0', conv)

        block1 = self._wide_layer(WideResNetBlock, planes[1], n, stride=1)
        self.feature_layers.add_module('block1', block1)
        block2 = self._wide_layer(WideResNetBlock, planes[2], n, stride=2)
        downsampled *= 2
        self.feature_layers.add_module('block2', block2)
        block3 = self._wide_layer(WideResNetBlock, planes[3], n, stride=2)
        downsampled *= 2
        self.feature_layers.add_module('block3', block3)

        if self.normalization:
            bn = torch.nn.BatchNorm2d(planes[3], momentum=0.9)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.feature_layers.add_module('bn3', bn)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.feature_layers.add_module('relu3', relu)

        representation = planes[3]
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.feature_layers.add_module('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.feature_layers.add_module('view', view)

        self.classifier_layers = nn.Sequential()
        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(planes[3], self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.classifier_layers.add_module('logits', logits)

        if use_BN:
            self.dense_layers = nn.Sequential(
                nn.Linear(representation, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
                )
        else:
            self.dense_layers = nn.Sequential(
                nn.Linear(representation, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
                )

    def _wide_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout, self.normalization))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

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


class WideResNetConf(torch.nn.Module):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0,
                 conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        super(WideResNetConf, self).__init__(**kwargs)

        self.N_class = N_class
        self.conf_approx = conf_approx
        assert self.conf_approx in ('logsumexp', 'network'), "Invalid input for 'conf_approx'"
        self.temperature = temperature
        self.model = WideResNet(N_class, resolution, depth, width, normalization, channels, dropout)
        if self.conf_approx == "network":
            self.conf_net = nn.Sequential()
            self.conf_net.add_module("d0", nn.Linear(self.N_class, 1024))
            self.conf_net.add_module("bn0", nn.BatchNorm1d(1024))
            self.conf_net.add_module("rl0", nn.ReLU())
            for k in range(5):
                self.conf_net.add_module(f"d{k+1:d}", nn.Linear(1024, 1024))
                self.conf_net.add_module(f"rl{k+1:d}", nn.ReLU())

            self.conf_net.add_module("de", nn.Linear(1024, 1))
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


class WideResNetEnsemble(torch.nn.Module):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0,
                 conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        super(WideResNetEnsemble, self).__init__(**kwargs)
        self.N_class = N_class
        # Classifier network C_1
        self.classifier = WideResNet(N_class, resolution, depth, width, normalization, channels, dropout)
        # Combined classifier and detector networks (C_0 and D_0). The trained classifier C_0 is used to initialize C_1,
        # but is not used for the final prediction
        self.classifier_with_reject = WideResNetConf(N_class,
                                                     resolution,
                                                     depth=depth,
                                                     width=width,
                                                     conf_approx=conf_approx,
                                                     temperature=temperature)
        
    def forward(self, x, return_d=False):
        cls_output = self.classifier(x)
        if return_d:
            _, d_output = self.classifier_with_reject(x, return_d=True)
            return cls_output, d_output
        else:
            return cls_output
