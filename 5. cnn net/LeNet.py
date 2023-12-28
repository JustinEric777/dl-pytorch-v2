from torch import nn as nn
import torch
from d2l import torch as d2l

# leNet 的组成：
# 卷积编码器：由两个卷积层组成
# 全连接层密集块：由三个全连接层组成


# net 定义
net = nn.Sequential(
    # 第一个卷积层，输入通道为 1，输出通道为 6，卷积核大小为 5，padding 填充为 2
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 平均汇聚层
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 第二个卷积层，输入通道为 6，输出通道为 16， 卷积核大小为 5
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 第二个汇聚层
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将输入张量展平为一维张量
    # 输入张量的所有维度除去批量维度（batch dimension）外的维度展平为一个一维张量
    nn.Flatten(),
    # 第一个全连接层
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    # 第二个全连接层
    nn.Linear(120, 84), nn.Sigmoid(),
    # 第三个全连接层
    nn.Linear(84, 10)
)


# load数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# 评估函数
def evaluate_accuracy_gpu(net, data_iter, device='cpu'):
    """计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        # 设置为评估模式
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    # 在这块取消梯度计算
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            animator.add(epoch + (i + 1) / num_batches,
                        (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == "__main__":
    print("检查打印模型：")
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print("该层是：", layer.__class__.__name__, 'output shape: \t', X.shape)

    print("\n模型训练：")
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
