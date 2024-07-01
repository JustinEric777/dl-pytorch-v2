from d2l import torch as d2l
import torch

# 1. 输入的通道数为ci，那么卷积核的输入通道数也需要为ci
# 2. 对每个通道输入的二维张量和卷积核的二维张量进行互相关运算, 再对通道求和
# 3. 为了获得多个通道的输出，可以为每个输出通道创建一个形状为ci × kh × kw的卷积核张量，这样卷积核的形状是co × ci × kh × kw
# 4. 1*1 卷积的作用：计算发生在通道上, 通常用于调整网络层的通道数量和控制模型复杂性


def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == "__main__":
    print("多通道输入卷积计算：")
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print("多通道输入卷积计算的结果为：\n", corr2d_multi_in(X, K))

    print("\n多输出通道：")
    K = torch.stack((K, K + 1, K + 2), 0)
    print("3 个输出通道的卷积核：\n", K)
    print("多输出通道的计算结果为：\n", corr2d_multi_in_out(X, K))

    print("\n 1*1 卷积层：")
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    print("1×1 卷积运算的结果为：\n", Y1)
    Y2 = corr2d_multi_in_out(X, K)
    print("多通道输出卷积计算的结果为：\n", Y2)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6


