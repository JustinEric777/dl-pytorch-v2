import torch
from d2l import torch as d2l

# 多尺度目标检测

img = d2l.plt.imread('./imgs/dog_and_cat.jpg')
h, w = img.shape[:2]


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    d2l.plt.show()


if __name__ == "__main__":
    print("多尺度锚框：")
    display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    display_anchors(fmap_w=2, fmap_h=2, s=[0.4])

    # 利用深层神经网络在多个层次上对图像进行分层表示，从而实现多尺度目标检测
    print("\n多尺度检测")

