import torchvision
from d2l import torch as d2l


# 图像增广：生成相似但不同的数据样本，主要是数据多样性，数据集的规模

# 图片展示
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()


# load data
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])


if __name__ == "__main__":
    print("显示原始图片：")
    d2l.set_figsize()
    img = d2l.Image.open('./imgs/people.jpeg')
    d2l.plt.imshow(img)
    d2l.plt.show()

    print("\n翻转和裁剪：")
    apply(img, torchvision.transforms.RandomHorizontalFlip())
    apply(img, torchvision.transforms.RandomVerticalFlip())

    print("\n随机改变形状：")
    shape_aug = torchvision.transforms.RandomResizedCrop(
        (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)

    print("\n改变颜色：")
    apply(img, torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0, saturation=0, hue=0))

    print("\n更改色调：")
    apply(img, torchvision.transforms.ColorJitter(
        brightness=0, contrast=0, saturation=0, hue=0.5))

    print("\n随机更改图像的亮度、对比度、饱和度、色调")
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug)

    print("\n多效果随机：")
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)

