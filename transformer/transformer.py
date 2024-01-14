import sys

import torch

from transformer.embedding import PositionalEmbedding
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from torch import nn
from torch import Tensor
import numpy as np

# Transformer模型
from dataloader.segmentation.segmentation import add_start_end_token


class Transformer(nn.Module):
    def __init__(self, encoder_block=6, decoder_block=6, batch=8, sentence_len=5, n_inputs=512, n_heads=2,
                 hidden_feedforward=2048, vocab_len=1000, drop_prob=0.1):
        """
        :param encoder_block:           Encoder模块中Encoder Block的个数
        :param decoder_block:           Decoder模块中Decoder Block的个数
        :param batch:                   batch个数
        :param sentence_len:            句子的长度（一个句子的单词数）
        :param n_inputs:                单词嵌入式向量 Embedding的长度
        :param n_heads:                 Multi Attention中header的个数
        :param hidden_feedforward:      FeedForward 中间隐藏层的向量长度
        :param vocab_len:                word_vec的单词个数
        """
        super(Transformer, self).__init__()
        self.word_vector_size = n_inputs
        self.positionalEmbedding = PositionalEmbedding(n_inputs, sentence_len)
        self.encoder = Encoder(block_num=encoder_block, batch=batch, sentence_len=sentence_len, n_inputs=n_inputs,
                               n_heads=n_heads, hidden_feedforward=hidden_feedforward, drop_prob=drop_prob)
        self.decoder = Decoder(block_num=decoder_block, batch=batch, sentence_len=sentence_len, n_inputs=n_inputs,
                               n_heads=n_heads, hidden_feedforward=hidden_feedforward, drop_prob=drop_prob)
        self.linear = nn.Linear(n_inputs, vocab_len, dtype=torch.float64)

    def forward(self, inputs, **kwargs) -> Tensor:
        src_batch = inputs[0]
        tgt_batch = inputs[1]
        src_batch_len = kwargs["lens"][0]
        tgt_batch_len = kwargs["lens"][1]
        src_batch = src_batch + self.positionalEmbedding.forward(None)
        tgt_batch = tgt_batch + self.positionalEmbedding.forward(None)
        encoders_output = self.encoder(src_batch, src_batch_len)
        x = self.decoder(encoders_output, tgt_batch, src_batch_len, tgt_batch_len)
        x: Tensor = self.linear(x)
        return x.softmax(dim=-1)

    def predict(self, inputs, **kwargs) -> str:
        """
        :param inputs: [sentence_len, n_inputs]
        :param kwargs: {"len": src_len}
        :return:
        """
        sentence_loader = kwargs["sentence_loader"]
        start_embedding = sentence_loader.tgt_word2vec.wv["[START]"]

        next_token = ""
        src = inputs
        src_len = [kwargs["len"]]

        tgt_array = np.array([[start_embedding]])
        tgt_array = np.pad(tgt_array, ((0, 0), (0, sentence_loader.max_sentence_len - len(tgt_array[0])), (0, 0)), mode='constant')
        tgt = Tensor(tgt_array, requires_grad=False).to("cuda:3")
        tgt_len = [1]
        src = src + self.positionalEmbedding.forward(None)
        encoders_output = self.encoder(src, src_len)
        out = ""
        while next_token != "[END]":
            x = self.decoder(encoders_output, tgt, src_len, tgt_len)
            x: Tensor = self.linear(x)
            next_token = sentence_loader.tgt_word2vec.wv.index_to_key[np.argmax(x.softmax().data[0][tgt_len[0] - 1])]
            out += next_token
            print(next_token + " ", end="")
            if tgt_len[0] < sentence_loader.max_sentence_len:
                tgt.data[0][tgt_len[0]] = sentence_loader.tgt_word2vec.wv[next_token]
                tgt_len[0] += 1
            else:
                break
        return out


def predict(english, sentence_loader, model: nn.Module):
    model.eval()
    tokens = add_start_end_token(english).strip().split()
    src_len = len(tokens)
    src = Tensor(np.array([[sentence_loader.src_word2vec.wv[tokens[i]].tolist()
                            if i < src_len
                            else [0 for _ in range(sentence_loader.word_vector_size)]] for i in
                           range(sentence_loader.max_sentence_len)]
                          ).reshape(1, sentence_loader.max_sentence_len, -1), requires_grad=False).to("cuda:3")
    out = model.predict(src, len=src_len, sentence_loader=sentence_loader)
    model.train()
    return out


if __name__ == '__main__':
    english = input()
    out = predict(english, "../../../res/simple_translation/simple_translation_5.sl",
                  "../../../tmp/translation_simple_5.pkl")
    sys.exit(0)
    batch = 10
    sentence_len = 10
    n_inputs = 20
    batchInput = Tensor(np.array(np.random.randn(batch, sentence_len, n_inputs)))
    batchOutput = Tensor(np.array(np.random.randn(batch, sentence_len, n_inputs)))
    inputLen = [8, 0, 6, 1, 9]
    outputLen = [1, 2, 3, 4, 5]
    T = Transformer(encoder_block=5, decoder_block=5, batch=batch, sentence_len=sentence_len, n_inputs=n_inputs)
    y: Tensor = T((batchInput, batchOutput), lens=(inputLen, outputLen))
    print(y.shape)
