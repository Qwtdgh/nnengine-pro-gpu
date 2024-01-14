
import streamlit as st
from lightGE.core.nn import LinearEndSequential, Conv2d, MaxPool2d, AvgPool2d, LSTM, Tanh, Sigmoid, ReLu, \
    Softmax, BatchNorm1d, BatchNorm2d, Dropout2d, Dropout, Linear, Model
from lightGE.core.mnist import MNIST
from lightGE.core.transformer.embedding import PositionalEmbedding
from lightGE.core.transformer.feed_forward import FeedForward
from lightGE.core.transformer.transformer import Transformer
from lightGE.core.transformer.decoder import DecoderBlock, Decoder
from lightGE.core.transformer.encoder import EncoderBlock, Encoder
from lightGE.core.transformer.add_norm import AddNorm
from lightGE.core.transformer.attention import SelfAttention, MultiAttention


class ModelCreator:
    @classmethod
    def create(cls, data: list):
        if len(data) == 0:
            st.warning('没有创建任何模型！请至少创建一个模型', icon="⚠️")
            return None
        else:
            models: list[Model] = []
            for model_data in data:
                model_name = model_data['type']
                if model_name == 'Linear':
                    models.append(Linear(n_inputs=model_data['n_inputs'],
                                         n_outputs=model_data['n_outputs']))
                elif model_name == 'Conv2d':
                    models.append(Conv2d(n_inputs=model_data['n_inputs'],
                                         n_outputs=model_data['n_outputs'],
                                         filter_size=model_data['filter_size'],
                                         stride=model_data['stride'],
                                         padding=model_data['padding'],
                                         bias=model_data['bias']))
                elif model_name == 'MaxPool2d':
                    models.append(MaxPool2d(filter_size=model_data['filter_size'],
                                            stride=model_data['stride'],
                                            padding=model_data['padding']))
                elif model_name == 'AvgPool2d':
                    models.append(AvgPool2d(filter_size=model_data['filter_size'],
                                            stride=model_data['stride'],
                                            padding=model_data['padding']))
                elif model_name == 'LSTM':
                    models.append(LSTM(n_inputs=model_data['n_inputs'],
                                       n_hidden=model_data['n_hidden'],
                                       n_outputs=model_data['n_outputs']))
                elif model_name == 'Tanh':
                    models.append(Tanh())
                elif model_name == 'Sigmoid':
                    models.append(Sigmoid())
                elif model_name == 'ReLu':
                    models.append(ReLu())
                elif model_name == 'Softmax':
                    models.append(Softmax())
                elif model_name == 'BatchNorm1d':
                    models.append(BatchNorm1d(n_inputs=model_data['n_inputs']))
                elif model_name == 'BatchNorm2d':
                    models.append(BatchNorm2d(n_inputs=model_data['n_inputs']))
                elif model_name == 'Dropout':
                    models.append(Dropout(p=model_data['p']))
                elif model_name == 'Dropout2d':
                    models.append(Dropout2d(p=model_data['p']))
                elif model_name == 'MNIST':
                    models.append(MNIST())
                elif model_name == 'SelfAttention':
                    models.append(SelfAttention(n_inputs=model_data['n_inputs'],
                                                n_outputs=model_data['n_outputs'],
                                                mask=model_data['mask']))
                elif model_name == 'MultiAttention':
                    models.append(MultiAttention(n_heads=model_data['n_heads'],
                                                 n_inputs=model_data['n_inputs'],
                                                 mask=model_data['mask']))
                elif model_name == 'AddNorm':
                    models.append(AddNorm(sentence_len=model_data['sentence_len'],
                                          n_inputs=model_data['n_inputs']))
                elif model_name == 'FeedForward':
                    models.append(FeedForward(n_inputs=model_data['n_inputs'],
                                              hidden=model_data['hidden'],
                                              ndrop_prob=model_data['ndrop_prob']))
                elif model_name == 'PositionalEmbedding':
                    models.append(PositionalEmbedding(embedding_len=model_data['embedding_len'],
                                                      sentence_len=model_data['sentence_len']))
                elif model_name == 'EncoderBlock':
                    models.append(EncoderBlock(batch=model_data['batch'],
                                               sentence_len=model_data['sentence_len'],
                                               n_inputs=model_data['n_inputs'],
                                               n_heads=model_data['n_heads'],
                                               hidden_feedforward=model_data['hidden_feedforward'],
                                               ndrop_prob=model_data['ndrop_prob']))
                elif model_name == 'Encoder':
                    models.append(Encoder(block_num=model_data['block_num'],
                                          batch=model_data['batch'],
                                          sentence_len=model_data['sentence_len'],
                                          n_inputs=model_data['n_inputs'],
                                          n_heads=model_data['n_heads'],
                                          hidden_feedforward=model_data['hidden_feedforward'],
                                          ndrop_prob=model_data['ndrop_prob']))
                elif model_name == 'DecoderBlock':
                    models.append(DecoderBlock(batch=model_data['batch'],
                                               sentence_len=model_data['sentence_len'],
                                               n_inputs=model_data['n_inputs'],
                                               n_heads=model_data['n_heads'],
                                               hidden_feedforward=model_data['hidden_feedforward'],
                                               ndrop_prob=model_data['ndrop_prob']))
                elif model_name == 'Decoder':
                    models.append(Decoder(block_num=model_data['block_num'],
                                          batch=model_data['batch'],
                                          sentence_len=model_data['sentence_len'],
                                          n_inputs=model_data['n_inputs'],
                                          n_heads=model_data['n_heads'],
                                          hidden_feedforward=model_data['hidden_feedforward'],
                                          ndrop_prob=model_data['ndrop_prob']))
                elif model_name == 'Transformer':
                    models.append(Transformer(encoder_block=model_data['encoder_block'],
                                              decoder_block=model_data['decoder_block'],
                                              batch=model_data['batch'],
                                              sentence_len=model_data['sentence_len'],
                                              n_inputs=model_data['n_inputs'],
                                              n_heads=model_data['n_heads'],
                                              hidden_feedforward=model_data['hidden_feedforward'],
                                              vocab_len=model_data['vocab_len'],
                                              ndrop_prob=model_data['ndrop_prob']))
            return LinearEndSequential(models)
