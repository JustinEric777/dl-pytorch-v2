import torch
from d2l import torch as d2l

# 注意力机制与全连接层或者汇聚层的区别源于增加的自主提示
# 注意力机制通过注意力汇聚使选择偏向于值（感官输入），
# 其中包含查询（自主性提示）和键（非自主性提示）。键和值是成对的


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图
       - 其输入matrices的形状是（要显示的行数，要显示的列数，查询的数目，键的数目）"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
            fig.colorbar(pcm, ax=axes, shrink=0.6)


if __name__ == "__main__":
    print("注意力提示热力图显示：")
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    d2l.plt.show()
