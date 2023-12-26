# 其目的还是为了模型的泛化性，通过注入噪声让模型更平滑一些
# dropout：在前向传播的过程中，计算每一层内部的同时并注入噪声
# 通俗来讲：表面上看是在训练的过程中丢弃了一些神经元
# 标准实现：计算下一层之前将当前层的一些节点置零
import torch
from torch import nn as nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    """dropout_layer 相关函数的实现"""
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都应该被保留
    if dropout == 0:
        return X

    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


# 定义模型参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# 定义模型
dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self,  X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)

        return out


if __name__ == "__main__":
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=[torch.device('cpu')])
