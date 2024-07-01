import torch
import d2l.torch as d2l
import torch.nn as nn

# 加载数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# loss 函数的实现
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法的时下
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


if __name__ == "__main__":
    num_epochs = 10
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=[torch.device('cpu')])
