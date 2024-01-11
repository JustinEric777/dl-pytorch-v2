import torch
from torch import nn as nn

# 转置卷积：
# 目的：上采样
# 计算方式：
#       1. 输入张量中的每个元素都要乘以卷积核
#       2. 方向根据 左 -> 右, 上 -> 下 一次计算
#       3. 然后将输出的中间张量相加
#
# 我们可以使用矩阵乘法来实现卷积。转置卷积层能够交换卷积层的正向传播函数和反向传播函数
#


def trans_conv(X, K):
    """转置卷积"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K

    return Y


if __name__ == "__main__":
    print("转置卷积：")
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    res = trans_conv(X, K)
    print("转置圈卷积计算的结果为：\n", res)

    print("\n步幅 - 扩大输出:")
    X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    tconv.weight.data = K
    res1 = tconv(X)
    print("输出结果为：\n", res1)
