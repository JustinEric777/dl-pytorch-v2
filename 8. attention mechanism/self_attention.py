import torch
from torch import nn as nn
from d2l import torch as d2l

# 自注意力：
# 1. 注意力机制：每个查询都会关注所有的键－值对并生成一个注意力输出
# 2. 自注意力机制：查询、键和值来自同一组输入
# 3. 每个词元都通过自注意力直接连接到任何其他词元
# 4. 相比较 CNN, RNN, 自注意力对长序列的计算复杂度比较大，因为 和序列长度 N**2 相关
# 5. 优势：并行计算、最短的最大路径长度
#
# 位置编码：
# 1. 固定位置编码：通过 sin(), cos() 函数进行相应的位置编码
# 2. 位置编码也可以通过学习得到
# 3. 在输入中添加位置编码，目的是为了并行计算，表示为 X+P
# 4. 位置编码代表的是相对的位置信息


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # 暂退法避免过拟合
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 P
        # 行代表词元在序列中的位置
        # 列代表位置编码的不同维度
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == "__main__":
    print("位置编码：")
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    d2l.plt.show()

