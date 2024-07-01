from torch import nn as nn
import torch
from d2l import torch as d2l

# 模型定义
net = nn.Sequential(
    # 使用更大的 11×11 的窗口来捕获数据
    # 步幅为 4，减少输出的高度和宽度
    nn.Conv2d(1, 96, kernel_size=11, padding=1, stride=4), nn.ReLU(),
    # 3×3 的最大汇聚层
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减少卷积窗口，padding =2 使得输入输出一致，且增大通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    # 3×3 的最大汇聚层
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 3层连续卷积和较小的卷积窗口
    # 除了最后的卷积层，输出的通道数量进一步增加
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    # 最大汇聚层
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 将输入张量展平为一维张量
    nn.Flatten(),
    # 全连接层
    nn.Linear(6400, 4096), nn.ReLU(),
    # 使用 dropout 暂退法来减轻过拟合
    nn.Dropout(p=0.5),
    # 全连接层
    nn.Linear(4096, 4096), nn.ReLU(),
    # 暂退法，来减轻过拟合
    nn.Dropout(p=0.5),
    # 输出层
    nn.Linear(4096, 10)
)

# 数据读取
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

if __name__ == "__main__":
    print("模型结构为：")
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print("该层是：", layer.__class__.__name__, 'output shape:\t', X.shape)

    print("\n模型训练：")
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
