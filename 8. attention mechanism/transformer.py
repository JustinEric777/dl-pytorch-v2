import torch
import sys
import math
from torch import nn as nn
from multi_head_attention import MultiHeadAttention
from self_attention import PositionalEncoding
from d2l import torch as d2l

sys.path.append('../7. modern rnn')
from encoder_decoder import Encoder, Decoder, EncoderDecoder, train_seq2seq


# Transformer:
# 1. 完全基于注意力机制，无任何的 CNN,RNN
#
# 编码器：
# 1. 多个层叠加，sublayer 构成 - 多头注意力 + 基于位置的前馈网络
# 2. 计算编码器的自注意力时，，查询、键和值都来自前一个编码器层的输出
# 3. 每个子层都采用了残差连接，残差连接需要满足 便残差连接满足x + sublayer(x) ∈ R ** d
# 4. 应用规范层，俗称层归一化 - 基于特征维度进行规范化
# 5. 输入序列对应的每个位置，Transformer编码器都将输出一个d维表示向量
#
# 解码器：
# 1. 多个层叠加，层中使用了残差连接和层规范化，sublayer 构成 - 多头注意力 + 基于位置的前馈网络 + 编码器－解码器注意力层
# 2. 编码器 - 解码器注意力：
#    2.1 查询来自前一个解码层的输出
#    2.2 建和值来自整个编码器的输出
# 3. 解码器自注意力：查询、减、值都来自上一个解码器层的输出
# 4. 解码器中的每个位置只能考虑该位置之前的所有位置，确保预测依赖于已生成的词元输出
#


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # 改变张量的最里层维度的尺寸，会改变成基于位置的前馈网络的输出尺寸
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self,  X, valid_lens):
        # Transformer 中的任何层都不会改变输入的形状
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer Encoder"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        # 添加 EncoderBlock 块
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
            return X


class DecoderBlock(nn.Module):
    """Decoder Block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state[2][self.i] 初始化
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        # dec_valid_lens 初始化
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 编码器 - 解码器注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(Decoder):
    """Transformer Decoder"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super.__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        # Decoder Block 填充
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        # 残差网络
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    def attention_weights(self):
        return self._attention_weights


if __name__ == "__main__":
    print("\n基于位置的前馈网络：")
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    res = ffn(torch.ones((2, 3, 4)))
    # 改变张量的最里层维度的尺寸，会改变成基于位置的前馈网络的输出尺寸
    # 所以最里面输出的 size = 8
    print("输入为：", torch.ones((2, 3, 4)))
    print("输出结果为：", res)

    print("\n残差连接后的层规范化：")
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    res1 = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4)))
    print("输入为：", torch.ones((2, 3, 4)))
    print("输出的结果为：", res1)

    print("\nTransformer 编码块：")
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    res_encoder_blk = encoder_blk(X, valid_lens)
    print("输入 X ：", X)
    print("输入 valid_lens：", valid_lens)
    print("Encoder Block 输出的结果为：", res_encoder_blk)

    print("\n TransformerEncoder:")
    encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    res_encoder = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)
    print("TransformerEncoder 输出结果为：", res_encoder)

    print("\nTransfromer 解码块：")
    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24))
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    res_decoder_blk = decoder_blk(X, state)
    print("Decoder Block 输出结果为：", res_decoder_blk)

    print("\n模型训练：")
    # 超参数
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    # load 数据
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    # encoder
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    # decoder
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    # net 定义
    net = EncoderDecoder(encoder, decoder)
    # train
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    # metric
    d2l.plt.show()

