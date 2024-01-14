from lightGE.utils.optimizer import Optimizer
import numpy as np


class Scheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        pass

    def step(self, loss):
        pass

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class MultiStepLR(Scheduler):

    def __init__(self, optimizer: Optimizer, milestones: [int], lr_decay: float = 0.1):
        super(MultiStepLR, self).__init__(optimizer)
        self.milestones = milestones
        self.lr_decay = lr_decay
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        if self.epoch in self.milestones:
            self.optimizer.lr *= self.lr_decay


class StepLR(Scheduler):

    def __init__(self, optimizer: Optimizer, step_size: int, lr_decay: float = 0.1):
        super(StepLR, self).__init__(optimizer)
        self.step_size = step_size
        self.lr_decay = lr_decay
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.lr_decay


class Exponential(Scheduler):

    def __init__(self, optimizer: Optimizer, lr_decay: float):
        super(Exponential, self).__init__(optimizer)
        self.lr_decay = lr_decay
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        self.optimizer.lr *= self.lr_decay


class Cosine(Scheduler):

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        super(Cosine, self).__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        self.optimizer.lr = self.eta_min + (self.optimizer.lr - self.eta_min) * (
                1 + np.cos(np.pi * self.epoch / self.T_max)) / 2


class LambdaLR(Scheduler):

    def __init__(self, optimizer: Optimizer, lr_lambda: callable):
        super(LambdaLR, self).__init__(optimizer)
        self.lr_lambda = lr_lambda
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        self.optimizer.lr = self.lr_lambda(self.epoch)


class ReduceLROnPlateau(Scheduler):

    def __init__(self, optimizer: Optimizer, lr_decay: float, patience: int):
        super(ReduceLROnPlateau, self).__init__(optimizer)
        self.lr_decay = lr_decay
        self.patience = patience
        self.epoch = 0
        self.best = float('inf')
        self.wait = 0

    def step(self, loss):
        self.epoch += 1
        if loss < self.best:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.optimizer.lr *= self.lr_decay
                self.wait = 0


class SchedulerFactory:
    @classmethod
    def generate(cls, name, optimizer, param_dict: dict):
        if name == 'MultiStepLR':
            return MultiStepLR(optimizer=optimizer, milestones=param_dict['milestones'],
                               lr_decay=param_dict['lr_decay'])
        elif name == 'StepLR':
            return StepLR(optimizer=optimizer, step_size=param_dict['step_size'], lr_decay=param_dict['lr_decay'])
        elif name == 'Exponential':
            return Exponential(optimizer=optimizer, lr_decay=param_dict['lr_decay'])
        elif name == 'Cosine':
            return Cosine(optimizer=optimizer, T_max=param_dict['T_max'], eta_min=param_dict['eta_min'])
        elif name == 'LambdaLR':
            return LambdaLR(optimizer=optimizer, lr_lambda=param_dict['lr_lambda'])
        elif name == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer=optimizer, lr_decay=param_dict['lr_decay'],
                                     patience=param_dict['patience'])
        else:  # default
            return Scheduler(optimizer=None)
