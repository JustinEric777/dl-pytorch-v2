import torch
from torch import nn as nn
import torch.nn.functional as F

# 一个块可以有由许多层组成，一个块可以由许多块组成
# 有单独的 Sequential 块处理层/块执行的有序性


# 自定义块
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        # 隐藏层
        self.hidden = nn.Linear(20, 256)
        # 输出层
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


# 有序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


# 前向传播中执行代码，进行相应层的定义
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# 混合块 - mix block
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == "__main__":
    # 自定义块
    X = torch.rand(2, 20)
    net = MLP()
    out = net(X)
    print(f"self_define_block out = {out}")

    # 有序块
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    out = net(X)
    print(f"sequential_block out = {out}")

    # 在前向传播中执行相应的网络层
    net = FixedHiddenMLP()
    out = net(X)
    print(f"fixed_hidden_mlp out = {out}")

    # 混合块的使用 - 多层快混合
    net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    out = net(X)
    print(f"nest_block out = {out}")
