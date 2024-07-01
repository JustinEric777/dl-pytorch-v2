import torch
from torch import nn as nn
from torch.nn import functional as F


# 单隐藏层的MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == "__main__":
    # 加载和保存张量
    x = torch.arange(4)
    torch.save(x, "x-file")
    x2 = torch.load('x-file')
    print(x2)

    # 存储张量列表
    y = torch.zeros(4)
    torch.save([x, y], 'x-files')
    x2, y2 = torch.load('x-files')
    print(x2, y2)

    # 字典
    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)

    # 保存和加载模型参数
    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    print(f"\n保存前的模型参数为：\n", net.state_dict())
    torch.save(net.state_dict(), 'mlp.params')

    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()
    print("\nload 模型的参数为：\n", clone.state_dict())

    Y_clone = clone(X)
    print(Y_clone == Y)
