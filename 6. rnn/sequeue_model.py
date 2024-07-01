import torch
from d2l import torch as d2l
from torch import nn as nn

# 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大
# 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型
# 对于直到时间步t的观测序列，其在时间步t + k的预测输出是“k步预测”

# 总共产生1000个点
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# 数据处理
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

# load 数据
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# loss - 平方损失
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
        l = loss(net(X), y)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


if __name__ == "__main__":
    # 数据显示
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    # 模型训练
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    # 模型预测 - 但不预测
    onestep_preds = net(features)
    d2l.plot([time, time[tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
             'x', legend=['data', '1-step preds'], xlim=[1, 1000],
             figsize=(6, 3))
    d2l.plt.show()

    # 模型预测 - 多步预测
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))
    d2l.plot([time, time[tau:], time[n_train + tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy(),
              multistep_preds[n_train + tau:].detach().numpy()], 'time',
             'x', legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    # 模型预测 - k 步预测
    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]

    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
             [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
             figsize=(6, 3))
    d2l.plt.show()


