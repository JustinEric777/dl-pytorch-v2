import sys
from d2l import torch as d2l
from torch import nn as nn

sys.path.append('../6. rnn')
from dataset_and_model import load_data_time_machine
from simple_rnn import RNNModel
from zero_rnn import train_ch8


# 双向循环神经网络：使用序列两端的信息来估计输出, 每个时间步的隐状态由当前时间步的前后数据同时决定
# 双向循环网络使用的场景比较少，一般不能用于任何预测任务,仅使用完型填空等相关场景
# 由于梯度链更长，因此双向循环神经网络的训练代价非常高

# 错误应用 - 双向循环神经网络预测文本
# load 数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)


if __name__ == "__main__":
    print("模型训练：")
    num_epochs, lr = 500, 1
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()
