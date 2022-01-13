import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_channels):
        super(InceptionBlock, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3reduce),
            nn.ReLU(True),
            nn.Conv2d(n3x3reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5reduce),
            nn.ReLU(True),
            nn.Conv2d(n5x5reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            nn.BatchNorm2d(pool_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = InceptionBlock(480, 192,  96, 208, 16,  48,  64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24,  64,  64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24,  64,  64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32,  64,  64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, out_features=num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        out = self.flatten(out)
        # out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return

def test():
    net = GoogLeNet(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    y = net(x)
    print(y.size())

test()