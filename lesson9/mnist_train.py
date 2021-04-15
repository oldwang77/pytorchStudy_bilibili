import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

batch_size = 512
# step1 加载数据集
train_loader = torch.utils.data.DataLoader(
    # normalize使得图片均匀分布在0附近
    # shuffle加载的时候，随机的打散
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
# torch.Size([512, 1, 28, 28]) torch.Size([512])
# 512张图片，1个通道，28*28的尺寸
print(x.shape, y.shape)


class Net(nn.Module):
    # 这里指定了每一层的构造
    def __init__(self):
        super(Net, self).__init__()

        # xw + b
        # nn.Linear表明这一层是线性层
        # in_features由输入张量的形状决定，out_features则决定了输出张量的形状
        # 这里的256和64是我们随机决定的
        # 最后分类有10个，最后一层一定是10
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b,1,28,28],一共有b张图片
        # h1 = relu ( xw + b )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 训练
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

for epoch in range(3):
    # 完成这个对整个数据集的一次迭代
    for batch_idx, (x, y) in enumerate(train_loader):
        # x的shape是[图片数量,1,28,28]
        # y的shape是[512]
        # 这是一个全连接层，它只能接受一个[图片数量,feature]，二位的tensor
        # 所以要把四维的=》二维度
        # x的shape是[图片数量,1,28,28] = > [图片数量,784]
        x = x.view(x.size(0), 28 * 28)
        # =>[b,10]
        out = net(x)

        # [b,10]
        # one_hot将类别进行编码
        y_onehot = one_hot(y)

        # 计算均方差
        loss = F.mse_loss(out, y_onehot)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # w' = w-lr*grad
        # 更新权重
        optimizer.step()

        # 一个元素张量可以用x.item()得到元素值
        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 图形化显示loss
plot_curve(train_loss)
# 这个时候我们会得到一个比较好的 [w1,b1,w2,b2,w3,b3]

# loss并不是我们衡量是否准确的指标，指标是acc

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out : [b,10]
    # 值最大的那个索引
    # out : [b,10] => pred:[b]
    # argmax返回最大值对应的坐标
    pred = out.argmax(dim=1)
    # 当前这个batch中，预测正确的总个数
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("test acc", acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test")
