from torch import nn as nn
import torch
from d2l import torch as d2l


# dropout 相关系数定义
dropout1, dropout2 = 0.2, 0.5


# net 相关定义
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第一个全连接层之后添加一个dropout层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第二个全连接层之后添加一个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


if __name__ == "__main__":
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    d2l.train_ch13(net, train_iter, test_iter, loss,  trainer, num_epochs, devices=[torch.device('cpu')])


