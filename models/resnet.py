"""
ResNet.
Take from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
"""
import torch
import utils.torch
from .classifier import Classifier
from .resnet_block import ResNetBlock
import torch.nn as nn


class ResNet(Classifier):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        """

        super(ResNet, self).__init__(N_class, resolution, **kwargs)

        self.blocks = blocks
        """ ([int]) Blocks. """

        self.channels = channels
        """ (int) Channels. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.inplace = False
        """ (bool) Inplace. """

        conv1 = torch.nn.Conv2d(self.resolution[0], self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        self.append_layer('conv1', conv1)

        if self.normalization:
            norm1 = torch.nn.BatchNorm2d(self.channels)
            torch.nn.init.constant_(norm1.weight, 1)
            torch.nn.init.constant_(norm1.bias, 0)
            self.append_layer('norm1', norm1)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.append_layer('relu1', relu)

        downsampled = 1
        for i in range(len(self.blocks)):
            in_planes = (2 ** max(0, i - 1)) * self.channels
            out_planes = (2 ** i) * self.channels
            layers = self.blocks[i]
            stride = 2 if i > 0 else 1

            downsample = None
            if stride != 1 or in_planes != out_planes:
                conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

                if self.normalization:
                    bn = torch.nn.BatchNorm2d(out_planes)
                    torch.nn.init.constant_(bn.weight, 1)
                    torch.nn.init.constant_(bn.bias, 0)
                    downsample = torch.nn.Sequential(*[conv, bn])
                else:
                    downsample = torch.nn.Sequential(*[conv])

            sequence = []
            sequence.append(ResNetBlock(in_planes, out_planes, stride=stride, downsample=downsample, normalization=self.normalization))
            for _ in range(1, layers):
                sequence.append(ResNetBlock(out_planes, out_planes, stride=1, downsample=None, normalization=self.normalization))

            self.append_layer('block%d' % i, torch.nn.Sequential(*sequence))
            downsampled *= stride

        representation = out_planes
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.append_layer('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.append_layer('view', view)

        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(representation, self._N_output)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)


class ResNetTwoBranch(torch.nn.Module):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        """

        super(ResNetTwoBranch, self).__init__(**kwargs)
        
        self.N_class = N_class
        self.resolution = resolution
        self.blocks = blocks
        """ ([int]) Blocks. """

        self.channels = channels
        """ (int) Channels. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.inplace = False
        """ (bool) Inplace. """

        self.feature_layers = nn.Sequential()
        conv1 = torch.nn.Conv2d(self.resolution[0], self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        self.feature_layers.add_module('conv1', conv1)

        if self.normalization:
            norm1 = torch.nn.BatchNorm2d(self.channels)
            torch.nn.init.constant_(norm1.weight, 1)
            torch.nn.init.constant_(norm1.bias, 0)
            self.feature_layers.add_module('norm1', norm1)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.feature_layers.add_module('relu1', relu)

        downsampled = 1
        for i in range(len(self.blocks)):
            in_planes = (2 ** max(0, i - 1)) * self.channels
            out_planes = (2 ** i) * self.channels
            layers = self.blocks[i]
            stride = 2 if i > 0 else 1

            downsample = None
            if stride != 1 or in_planes != out_planes:
                conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

                if self.normalization:
                    bn = torch.nn.BatchNorm2d(out_planes)
                    torch.nn.init.constant_(bn.weight, 1)
                    torch.nn.init.constant_(bn.bias, 0)
                    downsample = torch.nn.Sequential(*[conv, bn])
                else:
                    downsample = torch.nn.Sequential(*[conv])

            sequence = []
            sequence.append(ResNetBlock(in_planes, out_planes, stride=stride, downsample=downsample, normalization=self.normalization))
            for _ in range(1, layers):
                sequence.append(ResNetBlock(out_planes, out_planes, stride=1, downsample=None, normalization=self.normalization))

            self.feature_layers.add_module('block%d' % i, torch.nn.Sequential(*sequence))
            downsampled *= stride

        representation = out_planes
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.feature_layers.add_module('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.feature_layers.add_module('view', view)

        self.classifier_layers = nn.Sequential()
        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(representation, self.N_class)
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

    def forward(self, x, return_d=False):
        feature = self.feature_layers(x)
        cls_output = self.classifier_layers(feature)
        d_output = self.dense_layers(feature)
        d_output = torch.sigmoid(d_output)
        if return_d:
            return cls_output, d_output
        else:
            return cls_output


class ResNetTwoBranchDenseV1(torch.nn.Module):

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64, out_dim=10, use_BN=False, along=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        """

        super(ResNetTwoBranchDenseV1, self).__init__(**kwargs)
        
        self.N_class = N_class
        self.along = along
        self.resolution = resolution
        self.blocks = blocks
        """ ([int]) Blocks. """

        self.channels = channels
        """ (int) Channels. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.inplace = False
        """ (bool) Inplace. """

        self.feature_layers = nn.Sequential()
        conv1 = torch.nn.Conv2d(self.resolution[0], self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        self.feature_layers.add_module('conv1', conv1)

        if self.normalization:
            norm1 = torch.nn.BatchNorm2d(self.channels)
            torch.nn.init.constant_(norm1.weight, 1)
            torch.nn.init.constant_(norm1.bias, 0)
            self.feature_layers.add_module('norm1', norm1)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.feature_layers.add_module('relu1', relu)

        downsampled = 1
        for i in range(len(self.blocks)):
            in_planes = (2 ** max(0, i - 1)) * self.channels
            out_planes = (2 ** i) * self.channels
            layers = self.blocks[i]
            stride = 2 if i > 0 else 1

            downsample = None
            if stride != 1 or in_planes != out_planes:
                conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

                if self.normalization:
                    bn = torch.nn.BatchNorm2d(out_planes)
                    torch.nn.init.constant_(bn.weight, 1)
                    torch.nn.init.constant_(bn.bias, 0)
                    downsample = torch.nn.Sequential(*[conv, bn])
                else:
                    downsample = torch.nn.Sequential(*[conv])

            sequence = []
            sequence.append(ResNetBlock(in_planes, out_planes, stride=stride, downsample=downsample, normalization=self.normalization))
            for _ in range(1, layers):
                sequence.append(ResNetBlock(out_planes, out_planes, stride=1, downsample=None, normalization=self.normalization))

            self.feature_layers.add_module('block%d' % i, torch.nn.Sequential(*sequence))
            downsampled *= stride

        representation = out_planes
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.feature_layers.add_module('avgpool', pool)

        view = utils.torch.View(-1, representation)
        self.feature_layers.add_module('view', view)

        self.classifier_layers = nn.Sequential()
        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(representation, self.N_class)
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


class ResNetConf(torch.nn.Module):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64,
                 conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        super(ResNetConf, self).__init__(**kwargs)
        
        self.N_class = N_class
        self.conf_approx = conf_approx
        assert self.conf_approx in ('logsumexp', 'network'), "Invalid input for 'conf_approx'"
        self.temperature = temperature
        self.model = ResNet(N_class, resolution, blocks, normalization, channels)
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


class ResNetEnsemble(torch.nn.Module):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64,
                 conf_approx='network', temperature=0.01, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param conf_approx: type of approximation used for the prediction confidence. Valid choices are
                           ['logsumexp', 'network'].
        :type conf_approx: string
        :param temperature: temperature constant in (0, 1]. Small values can lead to better approximation of the confidence.
        :type temperature: float
        """
        super(ResNetEnsemble, self).__init__(**kwargs)
        self.N_class = N_class
        # Classifier network C_1
        self.classifier = ResNet(N_class, resolution, blocks, normalization, channels)
        # Combined classifier and detector networks (C_0 and D_0). The trained classifier C_0 is used to initialize C_1,
        # but is not used for the final prediction
        self.classifier_with_reject = ResNetConf(N_class,
                                                 resolution,
                                                 blocks=blocks,
                                                 conf_approx=conf_approx,
                                                 temperature=temperature)

    def forward(self, x, return_d=False):
        cls_output = self.classifier(x)
        if return_d:
            _, d_output = self.classifier_with_reject(x, return_d=True)
            return cls_output, d_output
        else:
            return cls_output
