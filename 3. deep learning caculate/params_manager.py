from torch import nn as nn
import torch

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


if __name__ == "__main__":
    # 单隐藏层的MLP - 意味着拥有两个线性层， 一个激活函数
    print(f"单隐藏层的MLP：")
    X = torch.rand(size=(2, 4))
    out = net(X)
    print(out)

    # 基本的参数访问
    print(f"\n基本参数的访问：")
    print(net[0].state_dict())
    print(net[1].state_dict())
    print(net[2].state_dict())

    # 目标函数的访问
    print(f"\n目标函数的访问：")
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)

    # 一次性访问所有参数
    print(f"\n一次性访问所有参数：")
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])

    # 访问网络参数的另一种方式
    print(f"\n访问网络参数的另一种方式：")
    print(net.state_dict()['2.bias'].data)

    # 从嵌套块收集参数
    print(f"\n从嵌套块收集参数：")
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    rgnet(X)
    print(f"嵌套块的网络结构为：\n{rgnet}")
    print(f"参数打印：\n{rgnet[0][1][0].bias.data}")

