import torch
import torch.nn as nn
import d2l.torch as d2l


# load data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# init model params
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


# activation function
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    """定义net
    # 这里“@”代表矩阵乘法
    """
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)


# loss func
loss = nn.CrossEntropyLoss(reduction="none")


if __name__ == "__main__":
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, updater, num_epochs)
