import torch
from torch import nn as nn

# 1. 降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性
# 2. 汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动
# 3. 为固定形状窗口（有时称为汇聚窗口）遍历的每个位置计算一个输出
# 4: 最大汇聚层和平均汇聚层


def pool2d(X, pool_size, mode='max'):
    """汇聚层的前向传播"""
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()

    return Y


if __name__ == "__main__":
    print("汇聚层：")
    print("最大汇聚层：")
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print("输入 X 为：\n", X)
    out = pool2d(X, (2, 2))
    print("经过最大汇聚层计算后的输出为：\n", out)
    print("平均汇聚层：")
    out = pool2d(X, (2, 2), 'avg')
    print("经过平均汇聚层计算后的输出为：\n", out)

    print("填充和步幅 - 汇聚层也可改变输出形状：")
    print(torch.arange(16, dtype=torch.float32))
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print("X 输入为：\n", X)
    pool2d = nn.MaxPool2d(3)
    out = pool2d(X)
    print("未设置填充和步幅之前的汇聚层输出的结果为：\n", out)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    out = pool2d(X)
    print("padding = 1，stride = 2 设置之后的汇聚层输出的结果为：\n", out)
    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    out = pool2d(X)
    print("任意大小的矩形汇聚窗口，其输出结果为：\n", out)

    print("多个通道：")
    X = torch.cat((X, X + 1), 1)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    out = pool2d(X)
    print("多通道汇聚计算后的计算结果为：\n", out)
