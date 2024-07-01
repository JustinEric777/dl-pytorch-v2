import torch
import numpy as np
from d2l import torch as d2l

# 梯度下降：
# 学习率：决定目标函数能否收敛到局部最小值，以及何时收敛到最小值
#


# 梯度下降
# 目标函数 - 凸函数
def f(x):
    return x ** 2


# 目标函数的梯度 - 凸函数
def f_grad(x):
    return 2 * x


c = torch.tensor(0.15 * np.pi)
# 目标函数 - 非凸函数
def f1(x):
    return x * torch.cos(c * x)


# 目标函数的梯度 - 非凸函数
def f1_grad(x):
    return torch.cos(c * x) - c * x * torch.sin(c * x)


# 求最优解
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
        print(f'epoch 10, x: {x:f}')
    return results


# 绘制chat
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])


# 二维梯度下降
def train_2d(trainer, steps=20, f_grad=None):
    # s1和s2是稍后将使用的内部状态变量
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


# 2d 梯度下降的chat
def show_trace_2d(f, results):
    """显示优化过程中2D变量的轨迹"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


# 目标函数 - 2d
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


# 目标函数的梯度
def f_2d_grad(x1, x2):
    return 2 * x1, 4 * x2


def gd_2d(x1, x2, s1, s2, f_grad, eta = 0.1):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)


def f_hess(x): # 目标函数的Hessian
    return c**2 * torch.cosh(c * x)


def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
        print('epoch 10, x:', x)
    return results


c2 = torch.tensor(0.5)
# O目标函数
def f2(x):
    return torch.cosh(c2 * x)


# 目标函数的梯度
def f2_grad(x):
    return c * torch.sinh(c2 * x)


if __name__ == "__main__":
    print("梯度下降:")
    results = gd(0.2, f_grad)
    show_trace(results, f)
    d2l.plt.show()

    # 低学习率可能导致没有进展
    print("\n低学习率：")
    show_trace(gd(0.05, f_grad), f)
    d2l.plt.show()

    # 高学习率可能导致结果逐渐发散
    print("\n高学习率：")
    show_trace(gd(1.1, f_grad), f)
    d2l.plt.show()

    print("\n局部最小值：")
    show_trace(gd(2, f1_grad), f1)
    d2l.plt.show()

    print("\n多维梯度下降（2d）:")
    show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
    d2l.plt.show()

    # 非凸函数 如f(x) = x cos(cx) 中，牛顿法的局限性，最终需要除以 Hessian， 意味着二阶导数为负
    # 局限性处理, 1. 用 Hessian 的绝对值来修正 2. 重新引入学习率
    print("\n自使用方法 - 牛顿法：")
    show_trace(newton(), f2)
    d2l.plt.show()

    # 牛顿法需要计算整个 Hessian， 收敛速度较慢
    # 处理方法：预处理 - 仅计算 Hessian 的对角线项


