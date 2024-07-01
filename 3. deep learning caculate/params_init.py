import torch
from torch import nn as nn

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


# 初始化为常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


# 延后初始化
# 直到数据第一次通过模型传递时，模型才会动态的推断出每个层的大小


# 参数绑定 - 实现多个层间共享参数
shared = nn.Linear(8, 8)
net1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                     shared, nn.ReLU(),
                     shared, nn.ReLU(),
                     nn.Linear(8, 1))

if __name__ == "__main__":
    X = torch.rand(size=(2, 4))
    out = net(X)
    print(out)

    print(f"\n参数的内置初始化：")
    net.apply(init_normal)
    print(f"初始化后的参数为：{net[0].weight.data[0]}, {net[0].bias.data[0]}")
    net[0].apply(init_xavier)
    print(f"初始化后的参数为：{net[0].weight.data[0]}, {net[0].bias.data[0]}")
    net[2].apply(init_42)
    print(f"初始化后的参数为：{net[0].weight.data[0]}, {net[0].bias.data[0]}")

    print(f"\n初始化为常数：")
    net.apply(init_constant)
    print(f"初始化后的参数为：{net[0].weight.data[0]}, {net[0].bias.data[0]}")

    print(f"\n自定义初始化参数：")
    net.apply(my_init)
    print(f"初始化后的参数为：{net[0].weight[:2]}")

    print(f"\n参数绑定 - 多个层共享参数：")
    net1(X)
    print(f"检查多层之前参数是否相同：", net1[2].weight.data[0] == net1[4].weight.data[0])
    net1[2].weight.data[0, 0] = 100
    print(f"参数值修改后都改变，确认参数为同一对象", net1[2].weight.data[0] == net1[4].weight.data[0])
