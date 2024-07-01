import torch
import torch.nn as nn
import d2l.torch as d2l

# net
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 超参数相关
batch_size, lr, num_epochs = 256, 0.1, 10

# 激活函数
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# loss func
loss = nn.CrossEntropyLoss(reduction='none')

# load data
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

if __name__ == "__main__":
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=[torch.device('cpu')])