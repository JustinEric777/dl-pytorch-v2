import sys
from d2l import torch as d2l
from torch import nn as nn

sys.path.append('../6. rnn')
from dataset_and_model import load_data_time_machine
from simple_rnn import RNNModel
from zero_rnn import train_ch8

# 深度循环神经网络 - 多隐藏层的 RNN
# 在深度循环神经网络中，隐状态的信息被传递到当前层的下一时间步和下一层的当前时间步

# load 数据
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 手动设置隐藏层的 LSTM
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)


if __name__ == "__main__":
    print("模型的训练与预测：")
    num_epochs, lr = 500, 2
    train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
    d2l.plt.show()

