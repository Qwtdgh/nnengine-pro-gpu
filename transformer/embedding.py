import numpy as np
from torch import nn
from torch import Tensor
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_len, sentence_len):
        """
        :param embedding_len:           单词嵌入式向量 Embedding的长度
        :param sentence_len:            句子的长度（一个句子的单词数）
        """
        super(PositionalEmbedding, self).__init__()
        self.encoding = np.zeros((sentence_len, embedding_len))
        pos = np.array([[i for j in range(int (embedding_len/2))] for i in range(sentence_len)])
        _2i = np.array([[2*j for j in range(int (embedding_len/2))] for i in range(sentence_len)])
        self.encoding[:, 0::2] = np.sin(pos / (10000 ** (_2i / embedding_len)))
        self.encoding[:, 1::2] = np.cos(pos / (10000 ** (_2i / embedding_len)))

    def forward(self, input: Tensor):
        return torch.tensor(self.encoding, requires_grad=False).to("cuda:0")

