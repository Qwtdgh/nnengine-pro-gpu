from typing import Tuple
from lightGE.core.tensor import Tensor
from lightGE.core.nn import Model
import numpy as np


class AddNorm(Model):
    def __init__(self, sentence_len, n_inputs):
        """
        :param sentence_len:            句子的长度（一个句子的单词数）
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        """
        super(AddNorm, self).__init__()
        self.n_inputs = n_inputs
        self.sentence_len = sentence_len
        self.n_inputs = n_inputs
        self.gamma = Tensor(np.array(np.ones((1, self.sentence_len, self.n_inputs))), autograd=True)
        self.beta = Tensor(np.array(np.zeros((1, self.sentence_len, self.n_inputs))), autograd=True)
        self.eps = Tensor(np.array(np.zeros((1, self.sentence_len, self.n_inputs))) + 1e-12, autograd=False)

    def forward(self, inputs: Tuple[Tensor]):
        inputs = inputs[0] + inputs[1]
        return self.gamma * (inputs - inputs.mean(-1, keepdims=True)) / (
            (inputs.var(-1, keepdims=True)).sqrt() + self.eps) + self.beta
