from typing import Any
import torch
from torch.utils import data
import torch.nn as nn
from d2l.torch import d2l


def synthetic_data() -> (torch.Tensor, torch.Tensor):
    """数据生成"""
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    return d2l.synthetic_data(true_w, true_b, 1000)


def load_data(data_arrays: Any, batch_size: int, is_train: bool = True):
    """构造一个 Pytorch 的数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)

    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def linear_net():
    """模型的定义以及初始化"""
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    return net


def loss_func():
    """损失函数 - 平方L2范数"""
    return nn.MSELoss()


def sgd(net: Any, lr: float):
    """优化算法的定义"""
    return torch.optim.SGD(net.parameters(), lr)


def train(features: torch.Tensor, labels: torch.Tensor, num_epochs: int, loss: Any, net: Any, sgd: Any):
    data_iter = load_data((features, labels), batch_size)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            sgd.zero_grad()
            l.backward()
            sgd.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


if __name__ == "__main__":
    # 生成数据
    features, labels = synthetic_data()

    # 超参数
    num_epochs = 30
    loss = loss_func()
    batch_size = 10
    net = linear_net()
    lr = 0.03
    sgd = sgd(net, lr)

    # 训练
    train(features, labels, num_epochs, loss, net, sgd)

