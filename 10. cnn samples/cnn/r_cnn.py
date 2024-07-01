
# R-CNN 区域神经卷积网络
# R-CNN 相关步骤：速度比较慢
#       1. 对输入图像使用选择性搜索来选取多个高质量的提议区域
#          这些提议区域通常是在多个尺度下选取的，并具有不同的形状和大小
#          每个提议区域都将被标注类别和真实边界框
#       2. 选择一个预训练的卷积神经网络，并将其在输出层之前截断
#          将每个提议区域变形为网络需要的输入尺寸，并通过前向传播输出抽取的提议区域特征
#       3. 将每个提议区域的特征连同其标注的类别作为一个样本
#          训练多个支持向量机对目标分类，其中每个支持向量机用来判断样本是否属于某一个类别
#       4. 将每个提议区域的特征连同其标注的边界框作为一个样本，训练线性回归模型来预测真实边界框
#
# Fast R-CNN：
# R-CNN 的性能瓶颈：重复计算 - 对每个提议区域，卷积神经网络前向传播是独立的，没有共享计算
# 改进：是仅在整张图象上执行卷积神经网络的前向传播，主要是加入了兴趣汇聚层
#
#
# Faster R-CNN：
# Fast R-CNN的性能瓶颈：通常需要在选择性搜索中生成大量的提议区域
# 改进：将选择性搜索替换为区域提议网络，减少提议区域的生成数量
#
#
# Mask R-CNN:
# 改进：N将兴趣区域汇聚层替换为了兴趣区域对齐层（全卷积网络），使用双线性插值来保留特征图上的空间信息
# 借助像素级位置进一步提升目标检测的精度
#
#