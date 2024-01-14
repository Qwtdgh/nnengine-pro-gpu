from collections import OrderedDict

from torch import nn
from torch import Tensor
from transformer.attention import MultiAttention
from transformer.add_norm import AddNorm
from transformer.feed_forward import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, drop_prob=0.1):
        """
        :param batch:                   batch个数
        :param sentence_len:            句子的长度（一个句子的单词数）
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        :param n_heads:                 Multi Attention中header的个数
        :param hidden_feedforward:      FeedForward 中间隐藏层的向量长度
        """
        super(EncoderBlock, self).__init__()
        self.multi_attention = MultiAttention(n_heads, n_inputs, False)
        self.add_norm1 = AddNorm(sentence_len, n_inputs)
        self.dropout1 = nn.Dropout(drop_prob)

        self.feedforward = FeedForward(n_inputs, hidden_feedforward)
        self.add_norm2 = AddNorm(sentence_len, n_inputs)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, inputs: Tensor, input_lens) -> Tensor:
        multi_out = self.multi_attention((inputs,), (input_lens,))

        dropout1 = self.dropout1(multi_out)
        add1_out = self.add_norm1((inputs, dropout1))

        feed_out = self.feedforward(add1_out)

        dropout2 = self.dropout2(feed_out)
        return self.add_norm2((add1_out, dropout2))


class Encoder(nn.Module):
    def __init__(self, block_num, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, drop_prob=0.1):
        super(Encoder, self).__init__()
        self.blocks = nn.Sequential(OrderedDict([("encoder_block" + str(i), EncoderBlock(batch, sentence_len, n_inputs, n_heads,
                                                                          hidden_feedforward, drop_prob=drop_prob))
                                                for i in range(block_num)]))

    def forward(self, inputs: Tensor, input_lens) -> Tensor:
        x = inputs
        for name, block in self.blocks.named_children():
            x = block(x, input_lens)
        return x
