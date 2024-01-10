import torch
import math
from d2l import torch as d2l

# 随机梯度下降
# 作用：可降低每次迭代时的计算代价 - 在每次迭代中，对数据样本进行随机均匀采样索引，并计算梯度并更新 X

# 目标函数
def f(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


# 目标函数的梯度
def f_grad(x1, x2):
    return 2 * x1, 4 * x2


def sgd(x1, x2, s1, s2, f_grad, eta=0.1):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return x1 - eta_t * g1, x2 - eta_t * g2, 0, 0


def constant_lr():
    return 1


# 动态学习率
def exponential_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    # α = 0.5的多项式衰减
    return math.exp(-0.1 * t)


if __name__ == "__main__":
    # 随机梯度下降问题：会导致梯度下降变得十分嘈杂
    # 处理方法：在优化过程中动态降低学习率
    print("随机梯度下降")
    eta = 0.1
    lr = constant_lr
    d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
    d2l.plt.show()

    print("\n动态学习率：")
    t = 1
    lr = exponential_lr
    d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
    d2l.plt.show()

