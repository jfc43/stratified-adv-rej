from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.torch


class TargetedObjective(nn.Module):
    """
    Targeted objective based on a loss, e.g., the cross-entropy loss.
    """

    def __init__(self, loss=utils.torch.classification_loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(TargetedObjective, self).__init__()

        self.loss = loss
        """ (callable) Loss. """

    def forward(self, logits, target_classes, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """
        if perturbations is not None:
            return -self.loss(logits, target_classes, perturbations=perturbations, reduction='none')
        else:
            return -self.loss(logits, target_classes, reduction='none')


class UntargetedObjective(nn.Module):
    """
    Untargeted loss based objective, e.g., cross-entropy loss.
    """

    def __init__(self, loss=utils.torch.classification_loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(UntargetedObjective, self).__init__()

        self.loss = loss
        """ (callable) Loss. """

    def forward(self, logits, true_classes, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """
        # assert self.loss is not None
        if perturbations is not None:
            return self.loss(logits, true_classes, perturbations=perturbations, reduction='none')
        else:
            return self.loss(logits, true_classes, reduction='none')


class SelectiveUntargetedObjective(UntargetedObjective):
    """
    Untargeted loss based objective for selective classifier, e.g., cross-entropy loss.
    """

    def __init__(self, loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(SelectiveUntargetedObjective, self).__init__(loss)

    def forward(self, logits, d_logits, true_classes, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """
        # assert self.loss is not None
        if perturbations is not None:
            return self.loss(logits, d_logits, true_classes, perturbations=perturbations, reduction='none')
        else:
            return self.loss(logits, d_logits, true_classes, reduction='none')
