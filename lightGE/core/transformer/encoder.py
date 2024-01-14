from lightGE.core.tensor import Tensor
from lightGE.core.nn import Model, Dropout, Sequential
from lightGE.core.transformer.attention import MultiAttention
from lightGE.core.transformer.add_norm import AddNorm
from lightGE.core.transformer.feed_forward import FeedForward


class EncoderBlock(Model):
    def __init__(self, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, ndrop_prob=0.9):
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
        self.dropout1 = Dropout(ndrop_prob)

        self.feedforward = FeedForward(n_inputs, hidden_feedforward)
        self.add_norm2 = AddNorm(sentence_len, n_inputs)
        self.dropout2 = Dropout(ndrop_prob)

    def forward(self, inputs: Tensor, input_lens) -> Tensor:
        multi_out = self.multi_attention((inputs,), (input_lens,))

        dropout1 = self.dropout1(multi_out)
        add1_out = self.add_norm1((inputs, dropout1))

        feed_out = self.feedforward(add1_out)

        dropout2 = self.dropout2(feed_out)
        return self.add_norm2((add1_out, dropout2))


class Encoder(Model):
    def __init__(self, block_num, batch, sentence_len, n_inputs, n_heads, hidden_feedforward, ndrop_prob=0.9):
        super(Encoder, self).__init__()
        self.blocks = Sequential([EncoderBlock(batch, sentence_len, n_inputs, n_heads, hidden_feedforward, ndrop_prob=ndrop_prob) for _ in range(block_num)])

    def forward(self, inputs: Tensor, input_lens) -> Tensor:
        x = inputs
        for block in self.blocks.sub_models.values():
            x = block(x, input_lens)
        return x
