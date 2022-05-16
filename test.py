import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import hiddenlayer as h
from torchviz import make_dot


class CNN2(nn.Module):  # 3 layers
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(1 * 10816, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def modeltorchviz(model, input, input2):
    # from torchviz import make_dot
    # params: model = MSDNet(args).cuda()
    # params: input = (3, 32, 32)
    # params: input2 = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
    print(model)
    summary(model, input)
    y = model(input2.cuda())  # 获取网络的预测值
    MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', input2)]))
    MyConvNetVis.format = "png"
    # 指定文件生成的文件夹
    MyConvNetVis.directory = "data"
    # 生成文件
    MyConvNetVis.view()


def main():
    model = CNN2(10).cuda()
    input = (1, 28, 28)
    input2 = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
    modeltorchviz(model, input, input2)


if __name__ == '__main__':
    main()

