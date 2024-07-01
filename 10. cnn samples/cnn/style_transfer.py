import torch
import torchvision
from torch import nn as nn
from d2l import torch as d2l

# 风格迁移：
# 损失函数：
#       1. 内容损失使合成图像与内容图像在内容特征上接近
#       2. 风格损失使合成图像与风格图像在风格特征上接近
#       3. 全变分损失有助于减少合成图像中的噪点
#
# 越靠近输入层，越容易抽取图像的输入信息
#
# 格拉姆矩阵：
#       假设该输出的样本数为1，通道数为c，
#       高和宽分别为h和w，我们可以将此输出转换为矩阵X，其有c行和hw列。
#       这个矩阵可以被看作由c个长度为hw的向量x1, . . . , xc组合而成的。
#       其中向量xi代表了通道i上的风格特征。
#       在这些向量的格拉姆矩阵XX⊤ ∈ R c×c中，i行j列的元素xij即向量xi和xj的内积
#
#


rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    """图片的预处理 - 对输入图像 RGB 三个通道做标准化"""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)


def postprocess(img):
    """图片的后处理 - 结果变为卷积神经输入的格式"""
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


def extract_features(X, content_layers, style_layers):
    """逐层计算，保存内容层，风格层的中间输出"""
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
    if i in style_layers:
        styles.append(X)
    if i in content_layers:
        contents.append(X)
    return contents, styles


content_img = d2l.Image.open('./imgs/rainier.jpg')
style_img = d2l.Image.open('./imgs/autumn-oak.jpg')


def get_contents(image_shape, device):
    """抽取内容图片特征"""
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    """抽取风格图片特征"""
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


# 网络 net 的定义 - vgg19
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 选取网络的层
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
# 定义新的net
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])


# 损失函数相关
# 内容损失
def content_loss(Y_hat, Y):
    # 平方误差函数
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()


# 风格损失
def gram(X):
    """将格拉姆矩阵除以了矩阵中元素的个数"""
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


# 全变分损失
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 损失函数的定义 - 3 种损失的加权和
content_weight, style_weight, tv_weight = 1, 1e3, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


# init
def get_inits(X, device, lr, styles_Y):
    """创建了合成图像的实例，并将其初始化为 X """
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                        xlim=[10, num_epochs],
                        legend=['content', 'style', 'TV'],
                        ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                             float(sum(styles_l)), float(tv_l)])
    return X


if __name__ == "__main__":
    print("模型训练：")
    device, image_shape = d2l.try_gpu(), (300, 450)
    net = net.to(device)
    content_X, contents_Y = get_contents(image_shape, device)
    _, styles_Y = get_styles(image_shape, device)
    output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)