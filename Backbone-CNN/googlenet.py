import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_channels):
        super(InceptionBlock, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3reduce, kernel_size=1),
            nn.Conv2d(n3x3reduce, n3x3, kernel_size=3, padding=1),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5reduce, kernel_size=1),
            nn.Conv2d(n5x5reduce, n5x5, kernel_size=3, padding=1),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64 )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = InceptionBlock(480, 192,  96, 208, 16,  48,  64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24,  64,  64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24,  64,  64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32,  64,  64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, out_features=num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        # N x 192 x 28 x 28
        x = self.a3(x)
        # N x 256 x 28 x 28
        x = self.b3(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.a4(x)
        # N x 512 x 14 x 14
        x = self.b4(x)
        # N x 512 x 14 x 14
        x = self.c4(x)
        # N x 512 x 14 x 14
        x = self.d4(x)
        # N x 528 x 14 x 14
        x = self.e4(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.a5(x)
        # N x 832 x 7 x 7
        x = self.b5(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = self.flatten(x)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

# def test():
#     net = GoogLeNet(num_classes=10)
#     x = torch.randn(4, 3, 224, 224)
#     y = net(x)
#     print(y.size())

# test()