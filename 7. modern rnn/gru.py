import torch
import sys
import d2l.torch as d2l
from torch import nn as nn

sys.path.append('../6. rnn')

from dataset_and_model import load_data_time_machine
from zero_rnn import train_ch8, RNNModelScratch
from simple_rnn import RNNModel


# 门控循环单元 - 支持隐状态的门控 - 并且可学习
# 原因：
#     1、希望有某种机制能后记忆很早期的信息，它很重要
#     2、通过某种机制跳过与当前表达无关的词元
#     3、存在逻辑中断的情况，在此情况下，可通过某种机制重置内部状态表示
# 重置门：允许我们控制“可能还想记住”的过去状态的数量 - 有助于捕获序列中的短期信息
# 更新门：允许我们控制新状态中有多少个是旧状态的副本 - 有助于捕获序列中的长期信息
# 操作：每当更新门 Zt 趋近于 1 时，模型倾向保留旧状态；当Zt接近0时，新的隐状态Ht就会接近候选隐状态H˜t
# 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列


# load data
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


# init params
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # 从标准差为0.01的高斯分布中提取权重
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    # 将偏置项设为0
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 更新门参数
    W_xz, W_hz, b_z = three()
    # 重置门参数
    W_xr, W_hr, b_r = three()
    # 候选隐状态参数
    W_xh, W_hh, b_h = three()

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    """初始化隐状态"""
    return torch.zeros((batch_size, num_hiddens), device=device),


def gru(inputs, state, params):
    """定义门控单元"""
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


if __name__ == "__main__":
    print("zero GRU 的训练与预测：")
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                init_gru_state, gru)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)


    print("simple GRU 的训练与预测：")
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()
