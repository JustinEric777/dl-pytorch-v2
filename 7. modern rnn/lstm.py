import sys
import torch
from d2l import torch as d2l
from torch import nn as nn

sys.path.append('../6. rnn')

from dataset_and_model import load_data_time_machine
from zero_rnn import train_ch8, RNNModelScratch
from simple_rnn import RNNModel

# 长短期记忆网络
# 门控记忆元：
# 输入门：何时将数据读入单元
# 忘记门：重置单元的内容
# 输出门：从单元中输出条目
# 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息
# 长短期记忆网络可以缓解梯度消失和梯度爆炸


# load 数据
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    """初始化模型参数"""
    num_inputs = num_outputs = vocab_size

    # 从标准差为0.01的高斯分布中提取权重
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 将偏置项设置为 0
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 输入门参数
    W_xi, W_hi, b_i = three()
    # 遗忘门参数
    W_xf, W_hf, b_f = three()
    # 输出门参数
    W_xo, W_ho, b_o = three()
    # 候选记忆元参数
    W_xc, W_hc, b_c = three()

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    """初始化隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    """定义模型"""
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


if __name__ == "__main__":
    print("zero - 预测和训练：")
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    print("simple - 预测和训练：")
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()
