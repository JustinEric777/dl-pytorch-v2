import torch
import d2l.torch as d2l


# ReLU函数 - 激活函数
# ReLU(x) = max(x, 0)
# 变种 pReLU：
# pReLU(x) = max(0, x) + α min(0, x)
# 计算该元素与 0 的最大值
def active_func_relu():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    d2l.plt.show()

    # Df/Dx
    # 计算相对于张量 x 的梯度，梯度的计算方式考虑了 y 对于 x 的影响，并且计算图被保留，以便后续的梯度计算
    # ReLU 求导后要么让参数消失，要么让参数通过
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(10, 5))
    d2l.plt.show()


# sigmoid 函数 - 挤压函数
# 作用：将输入变化为 [0, 1] 上的输出
# 求导：
# d sigmoid(x)/dx  = exp(−x)/(1 + exp(−x))2= sigmoid(x) (1 − sigmoid(x)).
def active_func_sigmoid():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(10, 5))
    d2l.plt.show()

    # 清除以前的梯度
    # x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(10, 5))
    d2l.plt.show()


# tanh 函数 - 挤压函数
# 将输入挤压到 [-1, 1] 这个区间内
def active_func_tanh():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.tanh(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(10, 5))
    d2l.plt.show()

    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh(x)', figsize=(10, 5))
    d2l.plt.show()


if __name__ == "__main__":
    # RelU
    # active_func_relu()

    # sigmoid
    # active_func_sigmoid()

    # tanh
    active_func_tanh()

