from torch import nn as nn

# encoder: 接收长度可变的序列为输入，将其转换为具有固定形状的编码状态
# decoder: 将固定形状的编码状态映射到长度可变的序列,
# # 是基于输入序列的编码信息和输出序列已经看见的或者生成的词元来预测下一个词元


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    # 只指定长度可变的序列作为编码器的输入X
    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    # 用于将编码器的输出（enc_outputs）转换为编码后的状态
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    # 在每个时间步都会将输入和编码后的状态映射成当前时间步的输出词元
    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """Encoder-Decoder 架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

