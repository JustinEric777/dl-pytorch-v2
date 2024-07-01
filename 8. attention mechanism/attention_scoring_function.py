import torch
import math
import sys
from torch import nn as nn

sys.path.append('../7. modern rnn')
from seq2seq import sequence_mask

# 将注意力汇聚的输出计算可以作为值的加权平均
# 选择不同的注意力评分函数会带来不同的注意力汇聚操作
# 当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数
# 当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高


# 掩码 softmax 操作
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来进行 softmax 操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其 softmax 输出为 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 加性注意力
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 缩放点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    print("掩码 softmax 操作：")
    res = masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
    print("输入为1：", torch.rand(2, 2, 4))
    print("输入为2：", torch.tensor([2, 3]))
    print("结果为：", res)

    print("\n加性注意力 - MLP：")
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval()
    res1 = attention(queries, keys, values, valid_lens)
    print("输入的 queries = ", queries)
    print("输入的 keys = ", keys)
    print("输入的 values = ", values)
    print("输入的 valid_lens = ", valid_lens)
    print("结果为：", res1)

    print("\n点积缩放注意力：")
    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    res2 = attention(queries, keys, values, valid_lens)
    print("结果为：", res2)
