from lightGE.core.tensor import Tensor
import numpy as np


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
        self.t = 0

    def step(self, zero=True):
        raise NotImplemented

    def zero(self):
        for p in self.parameters:
            p.grad *= 0

    def clip_grad(self, clip_value):
        for p in self.parameters:
            p.grad.data = np.clip(p.grad.data, -clip_value, clip_value)

    def clip_grad_norm(self, max_norm, norm_type):
        total_norm = 0
        for p in self.parameters:
            param_norm = p.grad.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters:
                p.grad *= clip_coef

    def set_parameters(self, parameters):
        self.parameters = parameters


class SGD(Optimizer):

    def __init__(self, parameters, lr=0.1):
        super(SGD, self).__init__(parameters, lr)

    def step(self, zero=True):

        self.t += 1

        for p in self.parameters:

            p.data -= p.grad * self.lr

            if zero:
                p.grad *= 0


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta=(0.9, 0.999), eps=1e-8, d_model=512, warmup_step=4000):
        super(Adam, self).__init__(parameters, lr)
        self.beta = beta
        self.eps = eps

        self.m = list()
        self.v = list()
        self.d_model = d_model
        self.warmup_step = warmup_step

        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))
            self.v.append(Tensor(np.zeros_like(p.data)))

    def step(self, zero=True):

        self.t += 1
        self.lr = self.d_model ** (-0.5) * min(self.t ** (-0.5), self.t * self.warmup_step ** (-1.5))

        for p, m, v in zip(self.parameters, self.m, self.v):

            m.data = self.beta[0] * m.data + (1 - self.beta[0]) * p.grad
            v.data = self.beta[1] * v.data + (1 - self.beta[1]) * (p.grad ** 2)

            m_hat = m.data / (1 - self.beta[0] ** self.t)
            v_hat = v.data / (1 - self.beta[1] ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if zero:
                p.grad *= 0

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.m.clear()
        self.v.clear()
        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))
            self.v.append(Tensor(np.zeros_like(p.data)))


class AdaGrad(Optimizer):

    def __init__(self, parameters, lr=0.001, eps=1e-8):
        super(AdaGrad, self).__init__(parameters, lr)
        self.eps = eps

        self.m = list()

        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))

    def step(self, zero=True):

        self.t += 1

        for p, m in zip(self.parameters, self.m):

            m.data += p.grad * p.grad
            p.data -= self.lr * p.grad / (np.sqrt(m.data) + self.eps)

            if zero:
                p.grad *= 0

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.m.clear()
        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))


class RMSprop(Optimizer):

    def __init__(self, parameters, lr=0.001, beta=0.9, eps=1e-8):
        super(RMSprop, self).__init__(parameters, lr)
        self.beta = beta
        self.eps = eps

        self.m = list()

        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))

    def step(self, zero=True):

        self.t += 1

        for p, m in zip(self.parameters, self.m):

            m.data = self.beta * m.data + (1 - self.beta) * (p.grad * p.grad)
            p.data -= self.lr * p.grad / (np.sqrt(m.data) + self.eps)

            if zero:
                p.grad *= 0

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.m.clear()
        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))


class SGDMomentum(Optimizer):

    def __init__(self, parameters, lr=0.001, momentum=0.9):
        super(SGDMomentum, self).__init__(parameters, lr)
        self.momentum = momentum

        self.m = list()

        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))

    def step(self, zero=True):

        self.t += 1

        for p, m in zip(self.parameters, self.m):

            m.data = self.momentum * m.data + p.grad
            p.data -= self.lr * m.data

            if zero:
                p.grad *= 0

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.m.clear()
        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))


class OptimizerFactory:
    @classmethod
    def generate(cls, name, parameters, lr, parameter_dict=None):
        if name == 'SGD':
            if parameter_dict is None:
                return SGD(parameters=parameters, lr=lr)
            return SGD(parameters=parameters, lr=lr)
        elif name == 'Adam': # beta=(0.9, 0.999), eps=1e-8, d_model=512, warmup_step=4000
            if parameter_dict is None:
                return Adam(parameters=parameters, lr=lr)
            return Adam(parameters=parameters,
                        lr=lr,
                        beta=parameter_dict['beta'],
                        eps=parameter_dict['eps'])
        elif name == 'AdaGrad':
            if parameter_dict is None:
                return AdaGrad(parameters=parameters, lr=lr)
            return AdaGrad(parameters=parameters, lr=lr, eps=parameter_dict['eps'])
        elif name == 'RMSprop':
            if parameter_dict is None:
                return RMSprop(parameters=parameters, lr=lr)
            return RMSprop(parameters=parameters, lr=lr, beta=parameter_dict['beta'], eps=parameter_dict['eps'])
        elif name == 'SGDMomentum':
            if parameter_dict is None:
                return SGDMomentum(parameters=parameters, lr=lr)
            return SGDMomentum(parameters=parameters, lr=lr, momentum=parameter_dict['momentum'])
        else:  # default
            return SGD(parameters=parameters, lr=lr)
