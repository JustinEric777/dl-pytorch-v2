import torch
from torch import nn as nn


def comp_conv2d(conv2d, X):
    """计算卷积层"""
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)

    # 忽略前两个维度，批量大小和通道
    return Y.reshape(Y.shape[2:])


if __name__ == "__main__":
    print("padding 填充：")
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, device=torch.device('cpu'))
    X = torch.rand(size=(8, 8))
    print("高度和宽度都填充1：", comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print("高度和宽度分别填充2 和 1：", comp_conv2d(conv2d, X).shape)
