## Model参数接口

参数格式说明

> - 参数1：<参数名>（<类型>[， default=\<default value\>]）    <参数说明>
> - 参数2：<参数名>（<类型>[， default=\<default value\>]）    <参数说明>
> - ...
> - 参数n：<参数名>（<类型>[， default=\<default value\>]）    <参数说明>

### Linear

- 参数1：n_inputs（int）    输入的维度
- 参数2：n_outputs（int）    输出的维度

### Conv2d

- 参数1：n_inputs（int）    输入的维度
- 参数2：n_outputs（int）    输出的维度
- 参数3：filter_size（int）    卷积核边长
- 参数4：stride（int，default=1）    步长
- 参数5：padding（int，default=0）    填充
- 参数6：bias（Boolean，default=True）    结果是否加上偏置矩阵

### MaxPool2d

- 参数1：filter_size（int）    池化核大小
- 参数2：stride（int，default=1）    步长
- 参数3：padding（int，default=0）    填充

### AvgPool2d

- 参数1：filter_size（int）    池化核大小
- 参数2：stride（int，default=1）    步长
- 参数3：padding（int，default=0）    填充

### LSTM

- 参数1：n_inputs（int）    输入的维度
- 参数2：n_hidden（int）    隐藏层维度
- 参数3：n_outputs（int）    输出的维度

### Tanh

- 无参数

### Sigmoid

- 无参数

### Relu

- 无参数

### BatchNorm1d

- 参数1：n_inputs（int）    输入的维度

### BatchNorm2d

- 参数1：n_inputs（int）    输入的维度

### Dropout

- 参数1：p（float，default=0.5）    保留率

### Dropout2d

- 参数1：p（float，default=0.5）    保留率

### MNIST

- 无参数

### SelfAttention

- 参数1：n_inputs（int）    输入的维度
- 参数2：n_outputs（int）    输出的维度
- 参数3：mask（Boolean，default=Fasle）    是否需要mask

### MultiAttention

- 参数1：n_heads（int）    输入的维度
- 参数2：n_inputs（int）    输出的维度
- 参数3：mask（Boolean，default=Fasle）    是否需要mask

### AddNorm

- 参数1：sentence_len（int）    句子长度
- 参数2：n_inputs（int）    输入的维度

### FeedForward

- 参数1：n_inputs（int）    输入的维度
- 参数2：hidden（int）    隐藏层的维度
- ndrop_prob（float， default=1）    保留率

### PositionalEmbedding

- 参数1：embedding_len（int）    词嵌入向量的维度
- 参数2：sentence_len（int）    句子长度

### EncoderBlock

- 参数1：batch（int）    批处理大小
- 参数2：sentence_len（int）    句子长度
- 参数3：n_inputs（int）    输入的维度
- 参数4：n_heads（int）    多头注意力机制的头的个数
- 参数5：hidden_feedforward（int）    feedforward中间隐藏层的个数
- 参数6：ndrop_prob（float）    保留率

### Encoder

- 参数1：block_num（int）    encoder块的个数
- 参数1：batch（int）    批处理大小
- 参数2：sentence_len（int）    句子长度
- 参数3：n_inputs（int）    输入的维度
- 参数4：n_heads（int）    多头注意力机制的头的个数
- 参数5：hidden_feedforward（int）    feedforward中间隐藏层的个数
- 参数6：ndrop_prob（float）    保留率

### DecoderBlock

- 参数1：batch（int）    批处理大小
- 参数2：sentence_len（int）    句子长度
- 参数3：n_inputs（int）    输入的维度
- 参数4：n_heads（int）    多头注意力机制的头的个数
- 参数5：hidden_feedforward（int）    feedforward中间隐藏层的个数
- 参数6：ndrop_prob（float）    保留率

### Decoder

- 参数1：block_num（int）    decoder块的个数
- 参数1：batch（int）    批处理大小
- 参数2：sentence_len（int）    句子长度
- 参数3：n_inputs（int）    输入的维度
- 参数4：n_heads（int）    多头注意力机制的头的个数
- 参数5：hidden_feedforward（int）    feedforward中间隐藏层的个数
- 参数6：ndrop_prob（float）    保留率

### Transformer

- 参数1：encoder_block（int， default=3）    encoder块的个数
- 参数2：decoder_block（int，default=3）    decoder块的个数
- 参数3：batch（int，default=8）    批处理大小
- 参数4：sentence_len（int，default=5）    句子长度
- 参数5：n_inputs（int，default=512）    输入的维度
- 参数6：n_heads（int，default=2）    多头自注意力机制的头的个数
- 参数7：hidden_feedforward（int，default=2048）    feedforward中间隐藏层的维度
- 参数8：vocab_len（int，default=1000）    单词表的大小
- 参数9：ndrop_prob（float）    保留率