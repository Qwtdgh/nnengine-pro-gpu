

import numpy as np
import torch
from torch import Tensor


def mseLoss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) * (pred - target)).sum(0)


def maeLoss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().sum(0)


def crossEntropyLoss(pred: Tensor, target: Tensor) -> Tensor:
    # return -target * pred.log() - (Tensor(np.ones_like(pred.data)) - target) * (
    #            Tensor(np.ones_like(pred.data)) - pred.log())
    # return (-target * pred.log()).sum(axes=None) / Tensor(np.array(pred.shape[0]), requires_grad=False).to("cuda:3")
    # pred加一个极小数防止除0
    return -target * (pred + Tensor(np.array(1e-300), requires_grad=False).to("cuda:3")).log()


def huberLoss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    if diff.abs().data < 1:
        return (diff * diff).sum(0)
    else:
        return diff.abs().sum(0)


def nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (-pred * target).sum(1).mean(0)


def multi_classification_cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    :param pred:       (..., n, d) at least 2 dimensions
    :param target:     (..., n, d) at least 2 dimensions
    :return:           (1) 1 dimensions
    """
    tgt_len = []
    for i in range(target.data.shape[0]):
        zero_rows = np.where(np.all(target.cpu().numpy()[i] == 0, axis = -1))[0]
        if zero_rows.size == 0:
            tgt_len.append(target.cpu().numpy()[i].shape[0])
        else:
            tgt_len.append(zero_rows[0])
    return (-target * pred.log()).sum((0, 1, 2)) / torch.tensor(np.array(sum(tgt_len)), requires_grad=False).to("cuda:3")


def rmseLoss(pred: Tensor, target: Tensor) -> Tensor:
    return mseLoss(pred, target).sqrt()


def mbeLoss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).sum(0)


def hingeLoss(pred: Tensor, target: Tensor) -> Tensor:
    '''
    :param pred: [-1, 1]
    :param target: {1, -1} 分别表示正负样本
    :return:
    '''
    I = Tensor(np.ones(pred.shape), autograd = False)
    return (I - pred * target).relu().sum(0)


def multi_classification_kld(pred: Tensor, target: Tensor) -> Tensor:
    """
    :param pred:       (..., n, d) at least 2 dimensions
    :param target:     (..., n, d) at least 2 dimensions
    :return:           (1) 1 dimensions
    """
    tgt_len = []
    for i in range(target.data.shape[0]):
        zero_rows = np.where(np.all(target.data[i] == 0, axis=-1))[0]
        if zero_rows.size == 0:
            tgt_len.append(target.data[i].shape[0])
        else:
            tgt_len.append(zero_rows[0])
    return (-target * (target / pred).log()).sum((0, 1, 2)) / Tensor(np.array(sum(tgt_len)), requires_grad=False).to("cuda:3")


class LossFuncFactory:
    @classmethod
    def generate(cls, name):
        # 'mseLoss', 'crossEntropyLoss', 'huberLoss', 'nll_loss', 'rmseLoss', 'mbeLoss')
        if name == 'mseLoss':
            return mseLoss
        elif name == 'maeLoss':
            return maeLoss
        elif name == 'crossEntropyLoss':
            return crossEntropyLoss
        elif name == 'huberLoss':
            return huberLoss
        elif name == 'nll_loss':
            return nll_loss
        elif name == 'multi_classification_cross_entropy_loss':
            return multi_classification_cross_entropy_loss
        elif name == 'rmseLoss':
            return rmseLoss
        elif name == 'mbeLoss':
            return mbeLoss
        elif name == 'multi_classification_kld':
            return multi_classification_kld
        elif name == 'hingeLoss':
            return hingeLoss
