from torch import nn as nn
import torch
from d2l import torch as d2l

# NIN 的思路：在每个像素的通道上分别使用多层感知机
# NiN使用由一个卷积层和多个1 × 1卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的像素非线性
# NIN 块的组成：
# 1. 以一个普通卷积层开始
# 2. 后面是两个 1×1 的卷积层


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # 窗口大小为 3，步幅为 2 的最大汇聚层
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    # 最大汇聚层
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    # 最大汇聚层
    nn.MaxPool2d(3, stride=2),
    # 暂退法避免过拟合
    nn.Dropout(0.5),
    # 标签类别是10，即输出是10，向输出靠近
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 全局平均汇聚层
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten()
)


if __name__ == "__main__":
    print("模型结构：")
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print("该层是：", layer.__class__.__name__, 'output shape:\t', X.shape)

    print("\n模型训练：")
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
