from collections import OrderedDict
from typing import Tuple
from torch import Tensor
from torch import nn
from transformer.attention import MultiAttention
from transformer.add_norm import AddNorm
from transformer.feed_forward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, drop_prob=0.1):
        """
        :param batch:                   batch个数
        :param sentence_len:            句子的长度（一个句子的单词数）
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        :param n_heads:                 Multi Attention中header的个数
        :param hidden_feedforward:      FeedForward 中间隐藏层的向量长度
        """
        super(DecoderBlock, self).__init__()
        self.mask_multi_attention = MultiAttention(n_heads, n_inputs, True)
        self.add_norm1 = AddNorm(sentence_len, n_inputs)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.multi_attention = MultiAttention(n_heads, n_inputs, False)
        self.add_norm2 = AddNorm(sentence_len, n_inputs)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.feed_forward = FeedForward(n_inputs, hidden_feedforward)
        self.add_norm3 = AddNorm(sentence_len, n_inputs)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, inputs: Tuple[Tensor], input_lens, output_lens) -> Tensor:
        mask_multi_out1 = self.mask_multi_attention((inputs[0],), (output_lens,))
        add_out1 = self.add_norm1((inputs[0], mask_multi_out1))
        dropout1 = self.dropout1(add_out1)

        multi_out2 = self.multi_attention((dropout1, inputs[1]), (output_lens, input_lens))
        dropout2 = self.dropout2(multi_out2)

        add_out2 = self.add_norm2((dropout1, dropout2))
        feed_out = self.feed_forward(add_out2)

        dropout3 = self.dropout3(feed_out)
        return self.add_norm3((add_out2, dropout3))


class Decoder(nn.Module):
    def __init__(self, block_num, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.blocks = nn.Sequential(OrderedDict([("decoder_block" + str(i), DecoderBlock(batch, sentence_len, n_inputs, n_heads, hidden_feedforward, drop_prob=drop_prob)) for i
                       in range(block_num)]))

    def forward(self, encoder: Tensor, output: Tensor, input_lens, output_lens) -> Tensor:
        """
        :param encoder: encoder is Encoder out mat C
        :param output:  output is the label sentence
        :return: [batch, sentence_len, n_inputs]
        """
        x = output
        for name, block in self.blocks.named_children():
            x = block((x, encoder), input_lens, output_lens)
        return x

