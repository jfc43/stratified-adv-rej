from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils.torch
import numpy as np


class MBBPDAAttack:

    def __init__(self, 
                model, 
                defense,
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        self.model = model
        self.defense = defense
        self.objective = objective
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.momentum = momentum
        self.lr_factor = lr_factor
        self.backtrack = backtrack
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

    def get_loss(self, adv_x, y):
        outputs = self.model(adv_x)
        loss = self.objective(outputs, y)
        return loss

    def perturb_once(self, x, y):

        delta = torch.zeros_like(x)
        batch_size = x.shape[0]
        global_gradients = torch.zeros_like(delta)
        adv_x = x + delta
        t_adv_x = self.defense.transform(adv_x)
        loss = self.get_loss(t_adv_x, y)
        success_errors = loss.data.clone()
        success_perturbs = delta.data.clone()

        lrs = (torch.ones_like(success_errors).float() * self.base_lr).cuda()
        """ (numpy.ndarray) Holds per element learning rates. """

        if self.rand_init_name == 'random+zero':
            random_name = random.choice(["random", "zero"])
            self.random_init(delta, x, random_name)
        else:
            self.random_init(delta, x, self.rand_init_name)

        for ii in range(self.max_iterations):
            adv_x = x + delta
            t_adv_x = self.defense.transform(adv_x)
            t_adv_x = nn.Parameter(t_adv_x)
            t_adv_x.requires_grad_()
            loss = self.get_loss(t_adv_x, y)

            cond = loss.data > success_errors
            success_errors[cond] = loss.data[cond]
            success_perturbs[cond] = delta.data[cond]

            loss.mean().backward()
            grad = t_adv_x.grad.data

            # normalize and add momentum.
            grad.data = torch.sign(grad.data)
            global_gradients.data = self.momentum*global_gradients.data + (1 - self.momentum)*grad.data

            if self.backtrack:
                next_perturbs = delta + torch.mul(utils.torch.expand_as(lrs, global_gradients), global_gradients)
                next_perturbs.data = torch.clamp(next_perturbs.data, min=-self.epsilon, max=self.epsilon)
                next_perturbs.data = torch.clamp(x.data + next_perturbs.data, min=self.clip_min, max=self.clip_max) - x.data
                with torch.no_grad():
                    next_adv_x = x + next_perturbs
                    t_next_adv_x = self.defense.transform(next_adv_x)
                    next_error = self.get_loss(t_next_adv_x, y)

                # Update learning rate if requested.
                for b in range(batch_size):
                    if next_error[b].item() >= loss.data[b]:
                        delta[b].data += lrs[b]*global_gradients[b].data
                    else:
                        lrs[b] = max(lrs[b] / self.lr_factor, 1e-20)
            else:
                delta.data += torch.mul(utils.torch.expand_as(lrs, global_gradients), global_gradients)

            delta.data = torch.clamp(delta.data, min=-self.epsilon, max=self.epsilon)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

        adv_x = x + delta
        t_adv_x = self.defense.transform(adv_x)
        loss = self.get_loss(t_adv_x, y)

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


class MBConfBPDAAttack(MBBPDAAttack):

    def __init__(self, 
                model, 
                defense,
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model, 
                defense,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

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

        self.rand_init_name = "zero"
        worst_loss, worst_perb = self.perturb_once(x, y)

        self.rand_init_name = "random"
        for k in range(self.num_rand_init):
            curr_worst_loss, curr_worst_perb = self.perturb_once(x, y)
            cond = curr_worst_loss > worst_loss
            worst_loss[cond] = curr_worst_loss[cond]
            worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBConfBPDAAttackMultitargeted(MBConfBPDAAttack):
    def __init__(self, 
                model, 
                defense,
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                defense,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class BPDAAttack:
    def __init__(self, model=None, defense=None, device=None, epsilon=None, learning_rate=0.5,
                 max_iterations=100, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.device = device

    def generate(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        adv = x.detach().clone()

        lower = np.clip(x.detach().cpu().numpy() - self.epsilon, self.clip_min, self.clip_max)
        upper = np.clip(x.detach().cpu().numpy() + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            adv_purified = self.defense(adv)
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward()

            grad_sign = adv_purified.grad.data.sign()

            # early stop, only for batch_size = 1
            # p = torch.argmax(F.softmax(scores), 1)
            # if y != p:
            #     break

            adv += self.LEARNING_RATE * grad_sign

            adv_img = np.clip(adv.detach().cpu().numpy(), lower, upper)
            adv = torch.Tensor(adv_img).to(self.device)
        return adv


class RandomAttack:

    def __init__(self, 
                model, 
                objective, 
                epsilon=0.3, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        self.model = model
        self.objective = objective
        self.epsilon = epsilon
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
        outputs = self.model(adv_x)
        loss = self.objective(outputs, y)
        return loss

    def perturb_once(self, x, y):

        delta = torch.zeros_like(x)
        batch_size = x.shape[0]
        global_gradients = torch.zeros_like(delta)
        loss = self.get_loss(x, delta, y)
        success_errors = loss.data.clone()
        success_perturbs = delta.data.clone()
        self.random_init(delta, x, self.rand_init_name)
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


class MBLinfPGDAttack:

    def __init__(self, 
                model, 
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
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
        self.lr_factor = lr_factor
        self.backtrack = backtrack
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
        outputs = self.model(adv_x)
        loss = self.objective(outputs, y)
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

            if self.backtrack:
                next_perturbs = delta + torch.mul(utils.torch.expand_as(lrs, global_gradients), global_gradients)
                next_perturbs.data = torch.clamp(next_perturbs.data, min=-self.epsilon, max=self.epsilon)
                next_perturbs.data = torch.clamp(x.data + next_perturbs.data, min=self.clip_min, max=self.clip_max) - x.data
                with torch.no_grad():
                    next_error = self.get_loss(x, next_perturbs, y)

                # Update learning rate if requested.
                for b in range(batch_size):
                    if next_error[b].item() >= loss.data[b]:
                        delta[b].data += lrs[b]*global_gradients[b].data
                    else:
                        lrs[b] = max(lrs[b] / self.lr_factor, 1e-20)
            else:
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


class MBConfLinfPGDAttack(MBLinfPGDAttack):

    def __init__(self, 
                model, 
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model, 
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

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

        self.rand_init_name = "zero"
        worst_loss, worst_perb = self.perturb_once(x, y)

        self.rand_init_name = "random"
        for k in range(self.num_rand_init):
            curr_worst_loss, curr_worst_perb = self.perturb_once(x, y)
            cond = curr_worst_loss > worst_loss
            worst_loss[cond] = curr_worst_loss[cond]
            worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBSATRLinfPGDAttack(MBConfLinfPGDAttack):

    def __init__(self, 
                model, 
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs, d_outputs = self.model(adv_x, return_d=True)
        loss = self.objective(outputs, d_outputs, y)
        return loss


class MBATRRLinfPGDAttack(MBConfLinfPGDAttack):

    def __init__(self, 
                model, 
                objective, 
                tempC=1.0,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

        self.tempC = tempC

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs, aux_outputs = self.model(adv_x, return_aux=True)
        # con_pre, _ = torch.softmax(outputs * self.tempC, dim=1).max(dim=1, keepdim=True) # predicted label and confidence
        aux_outputs = aux_outputs.sigmoid()
        # evi_outputs = con_pre * aux_outputs
        loss = self.objective(outputs, aux_outputs, y)
        return loss


class MBSATRStratifiedLinfPGDAttack(MBLinfPGDAttack):
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
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                multitargeted=False,
                num_classes=10,
                clip_min=0.0,
                clip_max=1.0):
        
        super().__init__(model, 
                objective, 
                epsilon, 
                max_iterations, 
                outer_base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

        self.epsilon_0 = epsilon_0
        self.fraction = fraction
        self.rr_objective = rr_objective
        self.multitargeted = multitargeted
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
            loss_1 = self.rr_objective(adv_outputs, adv_d_outputs, targets)
            return loss_1
        elif self.fraction == 0.0:
            targets_2 = y[split:]
            adv_outputs_2 = outputs[split:]
            adv_d_outputs_2 = d_outputs[split:]
            loss_2 = self.objective(adv_outputs_2, adv_d_outputs_2, targets_2)
            return loss_2
        else:
            targets = y[:split]
            adv_outputs = outputs[:split]
            adv_d_outputs = d_outputs[:split]
            loss_1 = self.rr_objective(adv_outputs, adv_d_outputs, targets)
            targets_2 = y[split:]
            adv_outputs_2 = outputs[split:]
            adv_d_outputs_2 = d_outputs[split:]
            loss_2 = self.objective(adv_outputs_2, adv_d_outputs_2, targets_2)
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
        
        if self.multitargeted:
            split = self.get_split(x.size(0))
            y_delta = np.random.randint(1, self.num_classes, size=split)
            y[:split] = (y[:split] + y_delta) % self.num_classes
            
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


class MBRCDStratifiedLinfPGDAttack(MBLinfPGDAttack):
    """
    Stratified PGD Attack for the RCD baseline

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
                objective_1, 
                objective_2,
                fraction,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):
        
        super().__init__(model, 
                None, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

        self.fraction = fraction
        self.objective_1 = objective_1
        self.objective_2 = objective_2

    def get_split(self, size_inp):
        return int(self.fraction * size_inp)

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs = self.model(adv_x)
        # Adversarial inputs are created from a fraction of inputs from a batch. Similar to 50% adversarial training
        split = self.get_split(x.size(0))

        if self.fraction == 1.0 or self.fraction == 0.0:
            raise ValueError
        else:
            targets_1 = y[:split]
            adv_outputs_1 = outputs[:split]
            loss_1 = self.objective_1(adv_outputs_1, targets_1)

            targets_2 = y[split:]
            adv_outputs_2 = outputs[split:]
            loss_2 = self.objective_2(adv_outputs_2, targets_2)
            return torch.cat((loss_1, loss_2), dim=0)


class MBSATRLinfPGDAttackMultitargeted(MBSATRLinfPGDAttack):
    def __init__(self, 
                model, 
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb

    
class MBRCDLinfPGDAttackMultitargeted(MBConfLinfPGDAttack):
    
    def __init__(self, 
                model, 
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBATRRLinfPGDAttackMultitargeted(MBATRRLinfPGDAttack):
    
    def __init__(self, 
                model, 
                objective, 
                num_classes,
                tempC=1.0,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                tempC, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)

        self.tempC = tempC
        self.num_classes = num_classes
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBConfLinfPGDAttackMultitargeted(MBConfLinfPGDAttack):
    def __init__(self, 
                model, 
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBCONSRLinfPGDAttackMultitargeted(MBConfLinfPGDAttack):
    def __init__(self, 
                model, 
                defense,
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        self.defense = defense

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        t_adv_x = self.defense.transform(adv_x)
        perb = t_adv_x.data-adv_x.detach().data
        t_adv_x = x + delta + perb
        outputs = self.model(adv_x)
        t_outputs = self.model(t_adv_x)
        loss = self.objective(outputs, y) + self.objective(t_outputs, y)
        return loss
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for y_delta in range(1, self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, (y + y_delta) % self.num_classes)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb


class MBCONSRLinfPGDAttack(MBConfLinfPGDAttack):

    def __init__(self, 
                model, 
                defense,
                objective, 
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model, 
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.defense = defense

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        t_adv_x = self.defense.transform(adv_x)
        perb = t_adv_x.data-adv_x.detach().data
        t_adv_x = x + delta + perb
        outputs = self.model(adv_x)
        t_outputs = self.model(t_adv_x)
        loss = self.objective(outputs, y) + self.objective(t_outputs, y)
        return loss


class MBCONSRLinfPGDInnerAttackMultitargeted(MBConfLinfPGDAttack):
    def __init__(self, 
                model, 
                defense,
                objective, 
                num_classes,
                epsilon=0.3, 
                max_iterations=100, 
                base_lr=0.1, 
                momentum=0.9, 
                lr_factor=1.5, 
                backtrack=True, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        super().__init__(model,
                objective, 
                epsilon, 
                max_iterations, 
                base_lr, 
                momentum, 
                lr_factor, 
                backtrack, 
                rand_init_name,
                num_rand_init,
                clip_min,
                clip_max)
        self.num_classes = num_classes
        self.defense = defense

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        t_adv_x = self.defense.transform(adv_x)
        perb = t_adv_x.data-adv_x.detach().data
        t_adv_x = x + delta + perb
        outputs = self.model(adv_x)
        t_outputs = self.model(t_adv_x)
        loss = self.objective(outputs, y) - self.objective(t_outputs, y)
        return loss
        
    def perturb(self, x, y):
        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = torch.empty(x.size()[0], device=x.device)
        worst_loss[:] = -np.inf
        worst_perb = torch.zeros_like(x)
        for target_y in range(self.num_classes):
            for k in range(self.num_rand_init):
                curr_worst_loss, curr_worst_perb = self.perturb_once(x, target_y)
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb
