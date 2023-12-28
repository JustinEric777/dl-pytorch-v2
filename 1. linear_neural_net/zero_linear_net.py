import random
import torch
from typing import Any


# 用线性模型参数w = [2, −3.4]⊤、b = 4.2
def synthetic_data(w: torch.Tensor, b: float, num_examples: int):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 矩阵相乘
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # y.reshape((-1, 1))：
    # -1 的意思是根据数组的大小自动确定这一维的长度
    # 而 1 表示希望得到的列数为 1
    return X, y.reshape((-1, 1))


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """数据迭代器，根据 batch_size 批量获取数据"""
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机获取索引
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linear_reg(X: torch.Tensor, w: torch.Tensor,  b: float) -> torch.Tensor:
    """定义线性模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params: Any, lr: float, batch_size: int):
    """小批量随机梯度下降"""
    # torch.no_grad 关闭梯度计算，上下文中的操作不会记录到梯度图中
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # 梯度清 0，将该张量中的所有元素置为 0
            param.grad.zero_()


def train(features: torch.Tensor, labels: torch.Tensor, num_epochs: int, lr: float, loss: Any, net: Any, batch_size: int, w: torch.Tensor, b: torch.Tensor):
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # X和y的小批量损失
            l = loss(net(X, w, b), y)
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            # 使用参数的梯度更新参数
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


if __name__ == "__main__":
    # 超参数
    num_epochs = 10
    lr = 0.03
    loss = squared_loss
    net = linear_reg
    batch_size = 32

    # 初始化模型
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 模型训练
    train(features, labels, num_epochs, lr, loss, net, batch_size, w, b)



