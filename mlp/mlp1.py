import d2l.torch as d2l
import torch
import torch.nn as nn


def load_data():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    return train_iter, test_iter


def init_param():
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    return params


def relu(x):
    a = torch.zeros_like(x)

    return torch.max(a, x)


def net(x):
    params = init_param()
    w1, b1, w2, b2 = params[0], params[1], params[2], params[3]
    x = x.reshape((-1, 784))
    h = relu(x@w1 + b1)

    return h@w2 + b2


def loss_func():
    loss = nn.CrossEntropyLoss(reduction='none')

    return loss


def train_net(num_epochs, lr):
    # data
    train_iter,  test_iter = load_data()

    # init params
    params = init_param()

    # loss
    loss = loss_func()

    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, updater, num_epochs, devices=[torch.device("cpu")])


if __name__ == "__main__":
    train_net(10, 0.1)

    train_iter,  test_iter = load_data()
    d2l.predict_sentiment(net=net, sequence=test_iter)
    

