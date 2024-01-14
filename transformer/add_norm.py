from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class AddNorm(Module):
    def __init__(self, sentence_len, n_inputs):
        """
        :param sentence_len:            句子的长度（一个句子的单词数）
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        """
        super(AddNorm, self).__init__()
        self.n_inputs = n_inputs
        self.sentence_len = sentence_len
        self.n_inputs = n_inputs
        self.gamma = torch.tensor(np.array(np.ones((1, self.sentence_len, self.n_inputs))), requires_grad=True).to("cuda:0")
        self.beta = torch.tensor(np.array(np.zeros((1, self.sentence_len, self.n_inputs))), requires_grad=True).to("cuda:0")
        self.eps = Tensor(np.array(np.zeros((1, self.sentence_len, self.n_inputs))) + 1e-12, requires_grad=False).to("cuda:0")

    def forward(self, inputs: Tuple[Tensor]):
        inputs = inputs[0] + inputs[1]
        return self.gamma * (inputs - inputs.mean(-1, keepdims=True)) / (
            (inputs.var(-1, keepdims=True)).sqrt() + self.eps) + self.beta
