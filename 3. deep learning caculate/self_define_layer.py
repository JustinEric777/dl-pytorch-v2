from torch import nn as nn
import torch
import torch.nn.functional as F


# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X-X.mean()


# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == "__main__":
    X = torch.rand(4, 8)

    print("自定义不带参数的层:")
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    out = net(X)
    print(f"net = {net}, out_mean = {out.mean()}")

    print(f"\n自定义带参数的层：")
    linear = MyLinear(5, 3)
    net = nn.Sequential(nn.Linear(8, 128), linear)
    out = linear(torch.rand(2, 5))
    print(f"net = {net}, out_mean = {out.mean()}")
    print(f"打印出相应的参数：{linear.weight}")
