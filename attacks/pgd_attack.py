from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils.torch
import numpy as np


class LinfPGDAttack:

    def __init__(self, 
                model, 
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        self.model = model
        self.objective = objective
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.momentum = momentum
        self.rand_init_name = rand_init_name
        self.num_rand_init = num_rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max

    def random_init(self, delta, x, random_name="random"):
        
        if random_name == 'random':
            delta.data.normal_()
            u = torch.zeros(delta.size(0)).uniform_(0, 1).cuda()
            linf_norm = u / torch.max(delta.abs().view(delta.size(0), -1), dim=1)[0]
            delta.data = self.epsilon * delta.data * linf_norm.view(delta.size(0), 1, 1, 1).data
        elif random_name == 'zero':
            delta.data.zero_()
        else:
            raise ValueError

        delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs, d_outputs = self.model(adv_x, return_d=True)
        loss = self.objective(outputs, d_outputs, y, perturbations=delta)
        return loss

    def perturb_once(self, x, y):

        delta = torch.zeros_like(x)
        batch_size = x.shape[0]
        global_gradients = torch.zeros_like(delta)
        loss = self.get_loss(x, delta, y)
        success_errors = loss.data.clone()
        success_perturbs = delta.data.clone()

        lrs = (torch.ones_like(success_errors).float() * self.base_lr).cuda()
        """ (numpy.ndarray) Holds per element learning rates. """

        if self.rand_init_name == 'random+zero':
            random_name = random.choice(["random", "zero"])
            self.random_init(delta, x, random_name)
        else:
            self.random_init(delta, x, self.rand_init_name)

        delta = nn.Parameter(delta)
        delta.requires_grad_()

        for ii in range(self.max_iterations):
            loss = self.get_loss(x, delta, y)

            cond = loss.data > success_errors
            success_errors[cond] = loss.data[cond]
            success_perturbs[cond] = delta.data[cond]

            loss.mean().backward()
            grad = delta.grad.data

            # normalize and add momentum.
            grad.data = torch.sign(grad.data)
            global_gradients.data = self.momentum*global_gradients.data + (1 - self.momentum)*grad.data

            delta.data += torch.mul(utils.torch.expand_as(lrs, global_gradients), global_gradients)
            delta.data = torch.clamp(delta.data, min=-self.epsilon, max=self.epsilon)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        loss = self.get_loss(x, delta, y)
        cond = loss.data > success_errors
        success_errors[cond] = loss.data[cond]
        success_perturbs[cond] = delta.data[cond]

        return success_errors, success_perturbs

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = None
        worst_perb = None
        for k in range(self.num_rand_init):
            curr_worst_loss, curr_worst_perb = self.perturb_once(x, y)
            if worst_loss is None:
                worst_loss = curr_worst_loss
                worst_perb = curr_worst_perb
            else:
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb
    
class SATRStratifiedLinfPGDAttack:
    """
    Stratified PGD Attack for the proposed method SATR

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
                self, 
                model, 
                objective, 
                rr_objective,
                fraction,
                epsilon=0.3, 
                epsilon_0 = 0.1,
                max_iterations=100, 
                outer_base_lr=0.1, 
                inner_base_lr=0.1, 
                momentum=0.9,
                rand_init_name="random",
                num_rand_init=1,
                num_classes=10,
                clip_min=0.0,
                clip_max=1.0):
        
        self.model = model
        self.objective = objective
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.momentum = momentum
        self.rand_init_name = rand_init_name
        self.num_rand_init = num_rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon_0 = epsilon_0
        self.fraction = fraction
        self.rr_objective = rr_objective
        self.num_classes = num_classes
        self.outer_base_lr = outer_base_lr
        self.inner_base_lr = inner_base_lr

    def get_split(self, size_inp):
        return int(self.fraction * size_inp)

    def random_init(self, delta, x, random_name="random"):
        
        if random_name == 'random':
            size_inp = delta.size(0)
            delta.data.normal_()
            u = torch.zeros(size_inp).uniform_(0, 1).cuda()
            linf_norm = u / torch.max(delta.abs().view(size_inp, -1), dim=1)[0]
            linf_norm = linf_norm.view(size_inp, 1, 1, 1)
            split = self.get_split(size_inp)
            if self.fraction > 0.0:
                delta.data[:split] = self.epsilon * delta.data[:split] * linf_norm.data[:split]
            if self.fraction < 1.0:
                delta.data[split:] = self.epsilon_0 * delta.data[split:] * linf_norm.data[split:]
        elif random_name == 'zero':
            delta.data.zero_()
        else:
            raise ValueError

        delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs, d_outputs = self.model(adv_x, return_d=True)
        # Adversarial inputs are created from a fraction of inputs from a batch. Similar to 50% adversarial training
        split = self.get_split(x.size(0))

        if self.fraction == 1.0:
            targets = y[:split]
            adv_outputs = outputs[:split]
            adv_d_outputs = d_outputs[:split]
            loss_1 = self.rr_objective(adv_outputs, adv_d_outputs, targets, perturbations=delta)
            return loss_1
        elif self.fraction == 0.0:
            targets_2 = y[split:]
            adv_outputs_2 = outputs[split:]
            adv_d_outputs_2 = d_outputs[split:]
            loss_2 = self.objective(adv_outputs_2, adv_d_outputs_2, targets_2, perturbations=delta)
            return loss_2
        else:
            targets = y[:split]
            adv_outputs = outputs[:split]
            adv_d_outputs = d_outputs[:split]
            loss_1 = self.rr_objective(adv_outputs, adv_d_outputs, targets, perturbations=delta)
            targets_2 = y[split:]
            adv_outputs_2 = outputs[split:]
            adv_d_outputs_2 = d_outputs[split:]
            loss_2 = self.objective(adv_outputs_2, adv_d_outputs_2, targets_2, perturbations=delta)
            return torch.cat((loss_1, loss_2), dim=0)

    def perturb_once(self, x, y):

        split = self.get_split(x.size(0))
        delta = torch.zeros_like(x)
        batch_size = x.shape[0]
        global_gradients = torch.zeros_like(delta)
        loss = self.get_loss(x, delta, y)
        success_errors = loss.data.clone()
        success_perturbs = delta.data.clone()
        outer_lrs = (torch.ones_like(success_errors[:split]).float() * self.outer_base_lr).cuda()
        inner_lrs = (torch.ones_like(success_errors[split:]).float() * self.inner_base_lr).cuda()
        """ (numpy.ndarray) Holds per element learning rates. """

        if self.rand_init_name == 'random+zero':
            random_name = random.choice(["random", "zero"])
            self.random_init(delta, x, random_name)
        else:
            self.random_init(delta, x, self.rand_init_name)

        delta = nn.Parameter(delta)
        delta.requires_grad_()

        for ii in range(self.max_iterations):
            loss = self.get_loss(x, delta, y)

            cond = loss.data > success_errors
            success_errors[cond] = loss.data[cond]
            success_perturbs[cond] = delta.data[cond]

            loss.mean().backward()
            grad = delta.grad.data

            # normalize and add momentum.
            grad.data = torch.sign(grad.data)
            global_gradients.data = self.momentum*global_gradients.data + (1 - self.momentum)*grad.data
            
            delta.data[:split] += torch.mul(utils.torch.expand_as(outer_lrs, global_gradients[:split]), global_gradients[:split])
            delta.data[split:] += torch.mul(utils.torch.expand_as(inner_lrs, global_gradients[split:]), global_gradients[split:])

            if self.fraction > 0.0:
                delta.data[:split] = torch.clamp(delta.data[:split], min=-self.epsilon, max=self.epsilon)
            if self.fraction < 1.0:
                delta.data[split:] = torch.clamp(delta.data[split:], min=-self.epsilon_0, max=self.epsilon_0)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        loss = self.get_loss(x, delta, y)
        cond = loss.data > success_errors
        success_errors[cond] = loss.data[cond]
        success_perturbs[cond] = delta.data[cond]

        return success_errors, success_perturbs
    
    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()
            
        worst_loss = None
        worst_perb = None
        for k in range(self.num_rand_init):
            curr_worst_loss, curr_worst_perb = self.perturb_once(x, y)
            if worst_loss is None:
                worst_loss = curr_worst_loss
                worst_perb = curr_worst_perb
            else:
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb
