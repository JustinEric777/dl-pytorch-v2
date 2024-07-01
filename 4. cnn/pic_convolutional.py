import torch
import torch.nn as nn


# 互相关运算 - 卷积计算
def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)


if __name__ == "__main__":
    # 互相关运算 - 卷积计算
    print("\n互相关运算 - 卷积计算:")
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(f"输入: X = {X}\n卷积核: K = {K}")
    print(f"卷积计算的结果为：corr_res = {corr2d(K, X)}")

    # 学习卷积核
    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    print("\n学习卷积核：")
    X = torch.ones((6, 8))
    print("X 输入为：\n", X)
    X[:, 2:6] = 0
    print("赋值之后的 X 为：\n", X)
    K = torch.tensor([[1.0, -1.0]])
    print("卷积核 K 为：\n", K)

    Y = corr2d(X, K)
    print("互相关运算后的 X 为：\n", Y)
    X = X.reshape((1, 1, 6, 8))
    print("reshape 之后的 X 为：\n", X)
    Y = Y.reshape((1, 1, 6, 7))
    print("reshape 之后的 Y 为：\n", X)
    # 学习率
    lr = 3e-2
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')
    print(f'卷积核的权重为：\n{conv2d.weight.data.reshape((1, 2))}')