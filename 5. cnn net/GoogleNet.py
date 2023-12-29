from torch import nn as nn
import torch.nn.functional as F
import torch
from d2l import torch as d2l


# 主要解决多大卷积核大小合适的问题
# Inception 块：使用不同大小的卷积核组合是有利的，由 4 条并行路径组成
# 1. 前 3 条路径使用 1×1、3×3、5×5的卷积层，从不同的空间大小中提取信息
# 2. 第四条路径使用3 × 3最大汇聚层，然后使用1 × 1卷积层来改变通道数
# Inception 块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1：1×1 的卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2：1×1 的卷积层后面接 3×3 卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3：1×1 的卷积层后面接 5×5 卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4：3x3 最大汇聚层后接 1x1 卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# 第一个模块：64 个通道， 7×7 的卷积层，最大汇聚层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第二个模块：64 个通道，1×1 的卷积层 + 通道数量×3倍的 3×3的卷积，对应 Inception 块中的第二条路径
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第三个模块：块串联两个完整的Inception块， 输出通道数如下：
# 1. 64 + 128 + 32 + 32 = 256
# 2. 128 + 192 + 96 + 64 = 480
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第四个模块：串联了5个Inception块，对应的输出通道数如下
# 1. 192 + 208 + 48 + 64 = 512
# 2. 160 + 224 + 64 + 64 = 512
# 3，128 + 256 + 64 + 64 = 512
# 4. 112 + 288 + 64 + 64 = 528
# 5. 256 + 320 + 128 + 128 = 832
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第五个模块：两个Inception 块，输出通道如下：
# 1. 256 + 320 + 128 + 128 = 832
# 2. 384 + 384 + 128 + 128 = 1024
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

# net 的定义
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

if __name__ == "__main__":
    print("模型结构为：")
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print("该层是：", layer.__class__.__name__, 'output shape:\t', X.shape)

    print("\n模型训练：")
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
