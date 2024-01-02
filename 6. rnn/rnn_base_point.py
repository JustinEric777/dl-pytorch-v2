import torch

# 概念：对隐状态使用循环计算的神经网络称为循环神经网络
# 评价：困惑度 - 困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”


# 计算转化: 矩阵乘 + 连接 = 矩阵连接 + 乘
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))


if __name__ == "__main__":
    print("计算转化(数学上课证明)：")
    print("X = ", X)
    print("W_xh = ", W_xh)
    print("H = ", H)
    print("W_hh = ", W_hh)
    res1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
    print("计算 X×W_xh + H × W_hh (乘积+连接)：\n", res1)
    res2 = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
    print("计算 (X+H) ×（W_xh+W_hh）(连接 + 乘积)：\n", res2)
