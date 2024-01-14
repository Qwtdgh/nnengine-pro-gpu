import math
from typing import Tuple
from torch import nn
from torch import Tensor

import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, n_inputs, n_outputs, mask=False):
        """
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        :param n_outputs:               self-attention输出的向量的长度
        :param mask:                    是否要进行mask操作（Encoder不需要 Decoder需要）
        """
        super(SelfAttention, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.mask = mask
        self.Wq = nn.nn.Liner(n_inputs, n_outputs)
        self.Wk = nn.Liner(n_inputs, n_outputs)
        self.Wv = nn.Liner(n_inputs, n_outputs)

    def forward(self, inputs: Tuple[Tensor], masks: Tuple[list]):
        q: Tensor = self.Wq(inputs[0])
        if len(inputs) == 1:
            k: Tensor = self.Wk(inputs[0])
            v: Tensor = self.Wv(inputs[0])
        else:
            k: Tensor = self.Wk(inputs[1])
            v: Tensor = self.Wv(inputs[1])

        attention: Tensor = q.bmm(k.transpose((0, 2, 1))) / Tensor(np.array([math.sqrt(self.n_outputs)]))
        if self.mask:
            attention = attention + maskMat(attention.shape)
        attention = lenMask(attention.shape, masks) + attention
        soft_attention: Tensor = attention.softmax()
        return soft_attention.bmm(v)


class MultiAttention(nn.Module):
    def __init__(self, n_heads, n_inputs, mask=False):
        """
        :param n_heads:                 Multi Attention中header的个数
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        :param mask:                    是否要进行mask操作（Encoder不需要 Decoder需要）
        """
        super(MultiAttention, self).__init__()
        self.n_heads = n_heads
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.self_attention = []
        self.mask = mask
        for i in range(n_heads):
            self.self_attention.append(SelfAttention(self.n_inputs, int(self.n_inputs / self.n_heads), self.mask))
        self.nn.Liner = nn.Liner(self.n_inputs, self.n_outputs)

    def forward(self, inputs: Tuple[Tensor], masks: Tuple[list]):
        con = concat([self.self_attention[i](inputs, masks) for i in range(self.n_heads)], axis=-1)
        return self.nn.Liner(con)


def maskMat(shape) -> Tensor:
    return Tensor(np.array([[-1e9 if j > i else 0 for j in range(shape[-1])] for i in range(shape[-2])]),
                  requires_grad=False).to("cuda:0")


def lenMask(shape, masks: Tuple[list]):
    mask_i = masks[0]
    if len(masks) == 1:
        mask_j = mask_i
    else:
        mask_j = masks[1]

    return Tensor(np.array([[[-1e9 if j >= mask_j[b_index] or i >= mask_i[b_index] else 0 for j in range(shape[2])] for i in range(shape[1])] for b_index in range(shape[0])]))

# s = Self_Attention(4, 5)
#
# ## s = Multi_Attention(3, 5, 4, mask=True)
# X = Tensor(np.array(np.random.randn(5, 4)))
# y = s((X,))
# print(y.shape)


